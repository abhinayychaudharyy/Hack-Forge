from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    return {
        "task_name":   "medium",
        "grid_width":  15,
        "grid_height": 15,
        "max_steps":   250,

        "drones": [
            {"id": "drone_0", "start_x": 0, "start_y": 0,  "battery": 100.0},
            {"id": "drone_1", "start_x": 0, "start_y": 2,  "battery":  80.0},
            {"id": "drone_2", "start_x": 9, "start_y": 13, "battery":  80.0},
        ],

        "tasks": [
            
            {
                "id":       "task_1",
                "item":     "Clothing",
                "pickup":   {"x": 3, "y": 1, "z": 0},
                "delivery": {"x": 12, "y": 5, "z": 0},
                "priority": 1,
            },
            {
                "id":       "task_2",
                "item":     "Books",
                "pickup":   {"x": 1, "y": 10, "z": 0},
                "delivery": {"x": 14, "y": 9, "z": 0},
                "priority": 1,
            },
            {
                "id":       "task_3",
                "item":     "Groceries",
                "pickup":   {"x": 4, "y": 2, "z": 0},
                "delivery": {"x": 11, "y": 13, "z": 0},
                "priority": 3,
            },
            {
                "id":       "task_4",
                "item":     "Medical Supplies",
                "pickup":   {"x": 4, "y": 4, "z": 0},
                "delivery": {"x": 12, "y": 4, "z": 0},
                "priority": 3,
            },
            {
                "id":       "task_5",
                "item":     "Tools",
                "pickup":   {"x": 1, "y": 5, "z": 0},
                "delivery": {"x": 13, "y": 5, "z": 0},
                "priority": 1,
            },
        ],

        "obstacles": [
            (6, 0), (6, 1), (6, 2),
            (6, 4), (6, 5), (6, 6),
            (6, 8), (6, 9), (6, 10), (6, 11), (6, 12),
        ],

        "charging_stations": [
            (0, 14),   # south-west
            (4, 14),   # south-centre
            (9, 14),   # south drone pad
        ],
    }
