import asyncio
import json
import time
import uuid
import urllib.error

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from sensor_simulator import ROOMS, generate_sensor_reading
from hybrid_engine import (
    evaluate_rules,
    analyze_with_ai,
    parse_timestamp,
)

# -------------------------
# CONFIGURATION
# -------------------------

BASE_DIR = Path(__file__).resolve().parent

CAPTURE_DIR = BASE_DIR / "captures"
STATIC_DIR = BASE_DIR / "static"

CAPTURE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

LATEST_FILE = BASE_DIR / "latest.json"
HISTORY_FILE = BASE_DIR / "detections.jsonl"

MODEL_NAME = "yolo26n.pt"

CONFIDENCE_THRESHOLD = 0.35

# Load the model only once.
model = YOLO(MODEL_NAME)

# Prevent concurrent requests from running inference
# on the same model simultaneously.
model_lock = Lock()

app = FastAPI(title="CampusTwin Camera API")


@app.get("/")
def index_page():
    """Serve the live campus dashboard."""

    page = BASE_DIR / "index.html"

    if not page.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html has not been created yet"
        )

    return FileResponse(page)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

# Most recent observations, indexed by camera.
latest_by_camera = {}
# Latest simulated IoT readings, indexed by room.
latest_sensor_readings = {}

# Store the latest scenario selected for each room.
sensor_scenarios = {
    room_id: "normal"
    for room_id in ROOMS
}

# --------------------------------------------------
# CONTINUOUS IOT SIMULATION
# --------------------------------------------------

SENSOR_UPDATE_INTERVAL = 5


async def continuous_sensor_simulator():
    """
    Continuously generate simulated readings
    for every configured room.
    """

    while True:
        for room_id in ROOMS:
            scenario = sensor_scenarios[room_id]

            reading = generate_sensor_reading(
                room_id=room_id,
                scenario=scenario,
            )

            latest_sensor_readings[room_id] = reading

        await asyncio.sleep(SENSOR_UPDATE_INTERVAL)


@app.on_event("startup")
async def start_sensor_simulator():
    """
    Start the simulator when FastAPI starts.
    """

    app.state.sensor_simulator_task = asyncio.create_task(
        continuous_sensor_simulator()
    )

# -------------------------
# HELPERS
# -------------------------

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(file_path, data):
    """
    Append one JSON object per line.
    Useful for historical analysis.
    """

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def analyze_frame(frame):
    """
    Run YOLO and return structured detections,
    including individual counts for each object category.
    """

    height, width = frame.shape[:2]

    start_time = time.perf_counter()

    with model_lock:
        result = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False
        )[0]

    inference_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            class_name = result.names[class_id]

            detections.append({
                "class_id": class_id,
                "label": class_name,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

    # Count every detected category independently.
    object_counts = dict(
        sorted(
            Counter(
                detection["label"]
                for detection in detections
            ).items()
        )
    )

    # Count people separately.
    person_count = object_counts.get("person", 0)

    # Total detections across all categories.
    detection_count = len(detections)

    return {
        "image_width": width,
        "image_height": height,
        "person_count": person_count,
        "detection_count": detection_count,
        "object_counts": object_counts,
        "detections": detections,
        "inference_ms": inference_ms
    }


# -------------------------
# CAMERA WEBPAGE
# -------------------------

@app.get("/camera")
def camera_page():

    page = STATIC_DIR / "camera.html"

    if not page.exists():
        raise HTTPException(
            status_code=404,
            detail="camera.html has not been created yet"
        )

    return FileResponse(page)


# -------------------------
# RECEIVE CAMERA FRAME
# -------------------------

@app.post("/api/frame")
async def receive_frame(request: Request):

    camera_id = request.query_params.get(
        "camera_id",
        "camera_unknown"
    )

    room_id = request.query_params.get(
        "room_id",
        "room_unassigned"
    )

    # Keep identifiers simple and safe for file names.
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if (
        not camera_id
        or not room_id
        or len(camera_id) > 64
        or len(room_id) > 64
        or not set(camera_id).issubset(allowed)
        or not set(room_id).issubset(allowed)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid camera_id or room_id"
        )

    content_type = request.headers.get(
        "content-type", ""
    )

    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Send an image with an image content type"
        )

    image_bytes = await request.body()

    # Avoid unexpectedly large uploads.
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large"
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image received"
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

    observation_time = utc_now()
    event_id = str(uuid.uuid4())

    # Run object detection and category-wise counting.
    analysis = analyze_frame(frame)

    # Save the original image for debugging.
    image_path = (
        CAPTURE_DIR
        / f"{camera_id}_{event_id}.jpg"
    )

    cv2.imwrite(str(image_path), frame)

    record = {
        "event_id": event_id,
        "timestamp_utc": observation_time,
        "camera_id": camera_id,
        "room_id": room_id,
        "source": "iphone_browser_camera",
        "status": "connected",
        **analysis
    }

    # Update the latest observation for this camera.
    latest_by_camera[camera_id] = record

    # Save one historical record per frame.
    append_jsonl(HISTORY_FILE, record)

    # Rewrite the current-state JSON file.
    snapshot = {
        "updated_at_utc": utc_now(),
        "cameras": latest_by_camera
    }

    LATEST_FILE.write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8"
    )

    return JSONResponse(record)


# -------------------------
# JSON ENDPOINTS
# -------------------------

@app.get("/api/rooms")
def get_rooms():
    """List configured rooms and their sensor-data source."""

    return {
        "source": "simulated",
        "rooms": [
            {
                "room_id": room_id,
                "name": room["name"],
                "category": room["category"],
            }
            for room_id, room in ROOMS.items()
        ],
    }


@app.get("/api/latest")
def get_latest():

    return {
        "updated_at_utc": utc_now(),
        "cameras": latest_by_camera
    }


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "known_cameras": list(latest_by_camera.keys())
    }

# -------------------------
# SIMULATED IOT SENSOR API
# -------------------------

@app.get("/api/sensors/latest")
def get_latest_sensor_readings():
    """Return the latest simulated readings for all rooms."""

    return {
        "source": "simulated",
        "rooms": latest_sensor_readings,
    }


@app.get("/api/sensors/{room_id}")
def get_room_sensor_reading(room_id: str):
    """Return the latest reading for one room."""

    if room_id not in ROOMS:
        raise HTTPException(
            status_code=404,
            detail="Unknown room ID",
        )

    if room_id not in latest_sensor_readings:
        latest_sensor_readings[room_id] = (
            generate_sensor_reading(
                room_id,
                sensor_scenarios[room_id],
            )
        )

    return latest_sensor_readings[room_id]


@app.post("/api/sensors/{room_id}/simulate")
def simulate_room_sensor_reading(
    room_id: str,
    scenario: str = "normal",
):
    """Generate and store a new simulated sensor reading."""

    if room_id not in ROOMS:
        raise HTTPException(
            status_code=404,
            detail="Unknown room ID",
        )

    allowed_scenarios = {
        "normal",
        "empty_room_ac_on",
        "hot_room",
        "water_leak",
        "after_hours_waste",
    }

    if scenario not in allowed_scenarios:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid scenario",
                "allowed_scenarios": sorted(
                    allowed_scenarios
                ),
            },
        )

    sensor_scenarios[room_id] = scenario

    reading = generate_sensor_reading(
        room_id,
        scenario,
    )

    latest_sensor_readings[room_id] = reading

    return reading

# --------------------------------------------------
# HYBRID RULE + AI ANALYSIS API
# --------------------------------------------------

@app.get("/api/hybrid/{room_id}")
def get_hybrid_analysis(room_id: str):
    """
    Combine room sensor readings and camera observations,
    evaluate deterministic rules, then request AI analysis.
    """

    import logging

    logger = logging.getLogger(__name__)

    # Validate room.
    if room_id not in ROOMS:
        raise HTTPException(
            status_code=404,
            detail="Unknown room ID",
        )

    # Retrieve latest sensor reading.
    sensor = latest_sensor_readings.get(room_id)

    if not sensor:
        return JSONResponse(
            status_code=503,
            content={
                "room_id": room_id,
                "status": "waiting_for_sensor_data",
                "sensor_data": None,
                "camera_data": None,
                "rule_alerts": [],
                "ai_analysis": None,
                "message": (
                    "No sensor reading is available yet."
                ),
            },
        )

    # Find the most recent camera record for this room.
    camera_candidates = [
        record
        for record in latest_by_camera.values()
        if record.get("room_id") == room_id
    ]

    camera_record = None
    newest_timestamp = None

    for record in camera_candidates:
        timestamp = (
            record.get("timestamp_utc")
            or record.get("timestamp")
        )

        parsed_timestamp = (
            parse_timestamp(timestamp)
            if "parse_timestamp" in globals()
            else None
        )

        if parsed_timestamp is None:
            continue

        if (
            newest_timestamp is None
            or parsed_timestamp > newest_timestamp
        ):
            newest_timestamp = parsed_timestamp
            camera_record = record

    # Evaluate deterministic rules first.
    evidence = evaluate_rules(
        room_id=room_id,
        sensor=sensor,
        camera_record=camera_record,
    )

    # Don't call the LLM if sensor evidence is stale.
    if evidence["status"] != "analyzed":
        return {
            "room_id": room_id,
            "status": evidence["status"],
            "sensor_data": sensor,
            "camera_data": camera_record,
            "rule_alerts": [],
            "ai_analysis": None,
            "evidence": evidence,
            "message": (
                "AI analysis was withheld because "
                "the sensor evidence is stale or invalid."
            ),
        }

    # Run AI analysis.
    try:
        recommendation = analyze_with_ai(evidence)

        return {
            "room_id": room_id,
            "status": "success",
            "sensor_data": sensor,
            "camera_data": camera_record,
            "rule_alerts": evidence["alerts"],
            "ai_analysis": recommendation,
            "evidence": evidence,
            "data_sources": {
                "camera": "real_camera",
                "iot_sensors": "simulated",
                "recommendation_engine": "rules_plus_local_llm",
            },
        }

    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        logger.warning(
            "AI analysis failed for %s: %s",
            room_id,
            exc,
        )

        return {
            "room_id": room_id,
            "status": "rules_only",
            "sensor_data": sensor,
            "camera_data": camera_record,
            "rule_alerts": evidence["alerts"],
            "ai_analysis": None,
            "evidence": evidence,
            "message": (
                "AI analysis is unavailable. "
                "Deterministic rule results are still available."
            ),
        }