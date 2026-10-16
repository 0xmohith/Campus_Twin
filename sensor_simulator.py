import random
from datetime import datetime, timezone

# -------------------------
# CONFIGURATION
# -------------------------

ROOMS = {
    "classroom_101": {
        "name": "Classroom 101",
        "category": "classroom",
        "temperature_range": (24.0, 32.0),
        "humidity_range": (45.0, 75.0),
        "base_power_w": 150.0,
    },
    "computer_lab_201": {
        "name": "Computer Lab 201",
        "category": "lab",
        "temperature_range": (23.0, 30.0),
        "humidity_range": (40.0, 70.0),
        "base_power_w": 400.0,
    },
    "library_301": {
        "name": "Library 301",
        "category": "library",
        "temperature_range": (23.0, 29.0),
        "humidity_range": (40.0, 65.0),
        "base_power_w": 100.0,
    },
}

_CAMPUS_BLOCKS = (
    ("M", "Main Block", "BLLCCL"),
    ("A", "Block A", "LLLCLC"),
    ("B", "Block B", "LLBCCC"),
    ("C", "Block C", "LBLCCL"),
    ("D", "Block D", "LLLLCC"),
    ("E", "Block E", "BLLCLC"),
    ("F", "Block F", "LLLCCB"),
)

_ROOM_CATEGORIES = {
    "L": ("Lecture hall", "lecture_hall", (24.0, 32.0), (45.0, 75.0), 150.0),
    "B": ("Library", "library", (23.0, 29.0), (40.0, 65.0), 100.0),
    "C": ("Lab", "lab", (23.0, 30.0), (40.0, 70.0), 400.0),
}

for _block_id, _block_name, _layout in _CAMPUS_BLOCKS:
    _category_counts = {"L": 0, "B": 0, "C": 0}

    for _room_number, _category_code in enumerate(_layout, start=1):
        (
            _category_name,
            _category,
            _temperature_range,
            _humidity_range,
            _base_power_w,
        ) = _ROOM_CATEGORIES[_category_code]
        _category_counts[_category_code] += 1
        _room_id = f"{_block_id}-{_room_number}"

        ROOMS[_room_id] = {
            "name": (
                f"{_block_name} {_category_name} "
                f"{_category_counts[_category_code]}"
            ),
            "category": _category,
            "temperature_range": _temperature_range,
            "humidity_range": _humidity_range,
            "base_power_w": _base_power_w,
        }

# -------------------------
# SENSOR SIMULATION
# -------------------------

def generate_sensor_reading(room_id, scenario="normal"):
    """
    Generate one simulated IoT reading for a room.

    Scenarios:
        normal
        empty_room_ac_on
        hot_room
        water_leak
        after_hours_waste
    """

    if room_id not in ROOMS:
        raise ValueError(f"Unknown room: {room_id}")

    room = ROOMS[room_id]

    # Default environmental conditions.
    temperature = random.uniform(
        *room["temperature_range"]
    )

    humidity = random.uniform(
        *room["humidity_range"]
    )

    # Default equipment states.
    ac_status = random.choice(["ON", "OFF"])
    lights_status = random.choice(["ON", "OFF"])

    leak_detected = False

    # Simulated equipment power consumption.
    base_power = room["base_power_w"]

    ac_power_w = (
        random.uniform(900, 1800)
        if ac_status == "ON"
        else 0.0
    )

    lights_power_w = (
        random.uniform(100, 350)
        if lights_status == "ON"
        else 0.0
    )

    equipment_power_w = random.uniform(
        50,
        base_power
    )

    # Apply scenario-specific conditions.
    if scenario == "empty_room_ac_on":

        ac_status = "ON"
        lights_status = "OFF"

        temperature = round(
            random.uniform(24.0, 29.0), 1
        )

        ac_power_w = random.uniform(900, 1800)
        lights_power_w = 0.0

    elif scenario == "hot_room":

        temperature = round(
            random.uniform(30.0, 35.0), 1
        )

        humidity = round(
            random.uniform(55.0, 80.0), 1
        )

        ac_status = "ON"
        ac_power_w = random.uniform(1000, 2000)

    elif scenario == "water_leak":

        leak_detected = True

        humidity = round(
            random.uniform(65.0, 85.0), 1
        )

    elif scenario == "after_hours_waste":

        ac_status = "ON"
        lights_status = "ON"

        ac_power_w = random.uniform(1000, 1800)
        lights_power_w = random.uniform(100, 350)

    elif scenario != "normal":

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    total_power_w = (
        ac_power_w
        + lights_power_w
        + equipment_power_w
    )

    return {
        "room_id": room_id,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": "simulated",
        "scenario": scenario,
        "environment": {
            "temperature_c": round(temperature, 1),
            "humidity_percent": round(humidity, 1),
        },
        "equipment": {
            "ac_status": ac_status,
            "lights_status": lights_status,
            "ac_power_w": round(ac_power_w, 1),
            "lights_power_w": round(lights_power_w, 1),
            "other_equipment_power_w": round(
                equipment_power_w, 1
            ),
            "total_power_w": round(total_power_w, 1),
        },
        "water": {
            "leak_detected": leak_detected,
        },
    }


# -------------------------
# MANUAL TEST
# -------------------------

if __name__ == "__main__":

    scenarios = [
        "normal",
        "empty_room_ac_on",
        "hot_room",
        "water_leak",
        "after_hours_waste",
    ]

    for scenario in scenarios:

        reading = generate_sensor_reading(
            room_id="classroom_101",
            scenario=scenario,
        )

        print(f"\n--- Scenario: {scenario} ---")

        import json

        print(
            json.dumps(
                reading,
                indent=2
            )
        )