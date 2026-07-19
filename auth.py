"""
auth.py
--------
Handles car-key login and pulls each car's locked, read-only info
(number plate, policy type, and a LIVE-computed age) from cars_dataset.csv.

Age is never stored as a fixed number -- it's computed from purchase_year
every time it's needed, so it automatically increases every year with zero
manual updates:

    age = current_year - purchase_year

Password is fixed for every car (995565), per project decision -- it is
hardcoded here, not stored in the CSV.
"""

import os
import csv
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "cars_dataset.csv")

FIXED_PASSWORD = "995565"


def _load_cars():
    """Returns {car_key: {number_plate, purchase_year, policy_type}}"""
    cars = {}
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            cars[row["car_key"].strip()] = {
                "number_plate": row["number_plate"].strip(),
                "purchase_year": int(row["purchase_year"]),
                "policy_type": row["policy_type"].strip(),
            }
    return cars


def verify_login(car_key: str, password: str):
    """Returns the car's info dict (with live-computed age) if valid, else None."""
    if password != FIXED_PASSWORD:
        return None

    cars = _load_cars()
    car_key = car_key.strip()
    if car_key not in cars:
        return None

    info = cars[car_key]
    current_year = datetime.now().year
    age = max(0, current_year - info["purchase_year"])

    return {
        "car_key": car_key,
        "number_plate": info["number_plate"],
        "policy_type": info["policy_type"],
        "vehicle_age": age,
    }


def get_car_by_key(car_key: str):
    """Fetch a logged-in car's current info again (e.g. for re-display),
    without re-checking the password. Used after login, mid-session."""
    cars = _load_cars()
    car_key = car_key.strip()
    if car_key not in cars:
        return None
    info = cars[car_key]
    current_year = datetime.now().year
    age = max(0, current_year - info["purchase_year"])
    return {
        "car_key": car_key,
        "number_plate": info["number_plate"],
        "policy_type": info["policy_type"],
        "vehicle_age": age,
    }
