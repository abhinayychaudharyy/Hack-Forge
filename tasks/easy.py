"""
Easy Task — AeroSync AI (Drone-Only)
=====================================
Difficulty:  1 / 3
Agents:      2 drones
Deliveries:  2 tasks (both priority 1 — normal)
Grid:        10 × 10 (open, no obstacles)
Max steps:   120
Battery:     Both drones start at 100%

Pipeline (drone-only — no robots):
    Drone descends to z=0 at pickup_location (shelf) → pick →
    Drone ascends → flies to delivery_location →
    Drone descends to z=0 → hover → place (DELIVERED)

Layout (z=0 ground plane):
  ┌──────────────────────────────┐
  │ dr0 .  .  .  .  dr1 .  .  . │  row 0  (dr0=drone_0, dr1=drone_1 start at z=1)
  │  .  .  .  .  .  .  .  .  .  │  row 1
  │  . [P0] .  .  .  .  .  .  . │  row 2  (P0=task_0 pickup/shelf)
  │  .  .  . [P1] .  .  .  .  . │  row 3  (P1=task_1 pickup/shelf)
  │  .  .  .  .  .  .  .  .  .  │  row 4
  │  .  .  .  .  .  .  .  .  .  │  row 5
  │  .  .  .  .  .  .  .  .  .  │  row 6
  │  .  .  .  .  .  .  .  .  .  │  row 7
  │  .  .  .  .  .  .  . [T0] . │  row 8  (T0=task_0 delivery)
  │ [C] .  .  .  .  .  . [T1][C]│  row 9  (T1=task_1 delivery, C=charging)
  └──────────────────────────────┘

Charging stations: (0,9)  (9,9)
"""
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    return {
        # ── Identity ─────────────────────────────────────────────────────
        "task_name":   "easy",
        "grid_width":  10,
        "grid_height": 10,
        "max_steps":   100,

        # ── Drones ───────────────────────────────────────────────────────
        # drone_0 — starts near task_0 shelf, full battery
        # drone_1 — starts near task_1 shelf, full battery
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

        # ── Tasks ─────────────────────────────────────────────────────────
        # task_0: Small Package  — shelf at (1,2) → deliver (7,8)
        # task_1: Documents      — shelf at (3,3) → deliver (8,9)
        #
        # Drones pick directly from shelf (no robot dispatch required).
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
        ],

        # ── Map Geometry ──────────────────────────────────────────────────
        # Open grid — no obstacles, lets agent focus on learning
        # the pick-and-deliver pipeline without routing complexity.
        "obstacles": [],

        # Two charging stations along the south wall
        "charging_stations": [
            (0, 9),   # south-west pad
            (9, 9),   # south-east pad
        ],
    }
