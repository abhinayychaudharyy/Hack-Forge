from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    return {
        "task_name":   "hard",
        "grid_width":  20,
        "grid_height": 20,
        "max_steps":   500,

        "drones": [
            {"id": "drone_0", "start_x": 0,  "start_y": 0,  "battery": 100.0},
            {"id": "drone_1", "start_x": 2,  "start_y": 0,  "battery": 100.0},
            {"id": "drone_2", "start_x": 10, "start_y": 0,  "battery": 100.0},
            {"id": "drone_3", "start_x": 0,  "start_y": 5,  "battery": 100.0},
            {"id": "drone_4", "start_x": 10, "start_y": 7,  "battery":  70.0},  # degraded
            {"id": "drone_5", "start_x": 0,  "start_y": 14, "battery":  55.0},  # critically low
        ],

        "tasks": [
            
            {
                "id":       "task_1",
                "item":     "Frozen Food",
                "pickup":   {"x": 4,  "y": 3,  "z": 0},
                "delivery": {"x": 19, "y": 4,  "z": 0},
                "priority": 3,
            },
            {
                "id":       "task_2",
                "item":     "Baby Supplies",
                "pickup":   {"x": 2,  "y": 2,  "z": 0},
                "delivery": {"x": 16, "y": 19, "z": 0},
                "priority": 3,
            },

            {
                "id":       "task_3",
                "item":     "Electronics Bundle",
                "pickup":   {"x": 3,  "y": 10, "z": 0},
                "delivery": {"x": 17, "y": 6,  "z": 0},
                "priority": 2,
            },
            {
                "id":       "task_4",
                "item":     "Smart Device",
                "pickup":   {"x": 5,  "y": 1,  "z": 0},
                "delivery": {"x": 15, "y": 18, "z": 0},
                "priority": 2,
            },
            {
                "id":       "task_5",
                "item":     "Tools Kit",
                "pickup":   {"x": 5,  "y": 3,  "z": 0},
                "delivery": {"x": 18, "y": 18, "z": 0},
                "priority": 2,
            },
            {
                "id":       "task_6",
                "item":     "Fragile Glassware",
                "pickup":   {"x": 3,  "y": 6,  "z": 0},
                "delivery": {"x": 19, "y": 15, "z": 0},
                "priority": 2,
            },

            {
                "id":       "task_7",
                "item":     "Clothing Set",
                "pickup":   {"x": 4,  "y": 7,  "z": 0},
                "delivery": {"x": 16, "y": 14, "z": 0},
                "priority": 1,
            },
            {
                "id":       "task_8",
                "item":     "Books Order",
                "pickup":   {"x": 2,  "y": 7,  "z": 0},
                "delivery": {"x": 17, "y": 17, "z": 0},
                "priority": 1,
            },
            {
                "id":       "task_9",
                "item":     "Sports Equipment",
                "pickup":   {"x": 4,  "y": 13, "z": 0},
                "delivery": {"x": 18, "y": 17, "z": 0},
                "priority": 1,
            },
            {
                "id":       "task_10",
                "item":     "Furniture Part",
                "pickup":   {"x": 4,  "y": 6,  "z": 0},
                "delivery": {"x": 14, "y": 3,  "z": 0},
                "priority": 1,
            },
            {
                "id":       "task_11",
                "item":     "Office Supplies",
                "pickup":   {"x": 5,  "y": 11, "z": 0},
                "delivery": {"x": 17, "y": 12, "z": 0},
                "priority": 1,
            },
        ],

        "obstacles": [
            (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),

            (8, 0),  (8, 1),  (8, 2),  (8, 3),  (8, 4),
            (8, 5),  (8, 6),  (8, 7),  (8, 8),
            (8, 9),  (8, 10), (8, 11), (8, 12), (8, 13),
            (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19),

            (0, 12), (2, 12), (4, 12),
        ],

        "charging_stations": [
            (0,  19),   # drone pad SW
            (5,  19),   # drone pad SC
            (10,  0),   # drone pad N
            (10, 19),   # drone pad S
        ],
    }
