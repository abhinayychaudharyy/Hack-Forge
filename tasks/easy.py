from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    return {
        "task_name":   "easy",
        "grid_width":  10,
        "grid_height": 10,
        "max_steps":   100,

        "drones": [
            {
                "id":      "drone_0",
                "start_x": 1,
                "start_y": 0,
                "battery": 100.0,
            },
            {
                "id":      "drone_1",
                "start_x": 5,
                "start_y": 0,
                "battery": 100.0,
            },
        ],

        "tasks": [
            {
                "id":       "task_0",
                "item":     "Small Package",
                "pickup":   {"x": 1, "y": 2, "z": 0},   # shelf
                "delivery": {"x": 7, "y": 8, "z": 0},   # customer address
                "priority": 1,
            },
            {
                "id":       "task_1",
                "item":     "Documents",
                "pickup":   {"x": 3, "y": 3, "z": 0},   # shelf
                "delivery": {"x": 8, "y": 9, "z": 0},   # customer address
                "priority": 1,
            },
            {
                "id":       "task_2",
                "item":     "Medicine",
                "pickup":   {"x": 5, "y": 2, "z": 0},   # shelf
                "delivery": {"x": 1, "y": 9, "z": 0},   # customer address
                "priority": 1,
            },
            {
                "id":       "task_3",
                "item":     "Electronics",
                "pickup":   {"x": 7, "y": 3, "z": 0},   # shelf
                "delivery": {"x": 9, "y": 5, "z": 0},   # customer address
                "priority": 1,
            },
        ],

        "obstacles": [],

        "charging_stations": [
            (0, 9),   # south-west pad
            (9, 9),   # south-east pad
        ],
    }
