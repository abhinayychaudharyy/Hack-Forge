from __future__ import annotations
import math
from typing import Any, Dict

from env.models import TaskStatus

_WEIGHTS = {
    "completion_weight": 0.35,
    "efficiency_weight": 0.18,
    "safety_weight":     0.22,   # safety matters most on hard
    "priority_weight":   0.15,
    "drone_weight":      0.10,
}

_COLLISION_HIT      = 0.10
_BATTERY_FAIL_HIT   = 0.15
_FORCED_RTB_HIT     = 0.05
_NEAR_MISS_HIT      = 0.03
_MOTOR_DEGRADED_HIT = 0.10



def _delivered_set(tasks: Dict[str, Any]) -> list:
    return [
        t for t in tasks.values()
        if t.get("status") in ("delivered", TaskStatus.DELIVERED)
    ]


def _priority_score(tasks: Dict[str, Any]) -> float:
    delivered = _delivered_set(tasks)
    if not delivered:
        return 0.01
    total_priority = sum(t.get("priority", 1) for t in delivered)
    max_possible   = 3 * len(delivered)
    raw = total_priority / max_possible if max_possible > 0 else 0.01
    return float(min(0.99, max(0.01, raw)))


def _drone_quality_score(state: Dict[str, Any]) -> float:
    drone_states = state.get("drone_states", {})
    if not drone_states:
        return 0.99

    total_near_misses  = 0
    total_forced_rtb   = 0
    avg_motor_health   = 0.0
    avg_stab_score     = 0.0
    total_deliveries   = 0
    total_failed_drops = 0

    for drone in drone_states.values():
        diag = drone.get("diagnostics", {}) if isinstance(drone, dict) else {}
        fl   = drone.get("flight",      {}) if isinstance(drone, dict) else {}

        total_near_misses  += diag.get("near_miss_count",        0)
        total_forced_rtb   += diag.get("forced_rtb_count",       0)
        avg_motor_health   += diag.get("motor_health",           1.0)
        avg_stab_score     += fl.get("hover_stability_score",    1.0)
        total_deliveries   += diag.get("total_deliveries",       0)
        total_failed_drops += diag.get("total_failed_deliveries", 0)

    n = len(drone_states)
    avg_motor_health /= n
    avg_stab_score   /= n

    deduction = 0.0
    deduction += min(total_near_misses * _NEAR_MISS_HIT,  0.4)
    deduction += min(total_forced_rtb  * _FORCED_RTB_HIT, 0.3)

    if avg_motor_health < 0.8:
        deduction += _MOTOR_DEGRADED_HIT * (0.8 - avg_motor_health) / 0.8

    if avg_stab_score < 0.7:
        deduction += 0.1 * (0.7 - avg_stab_score) / 0.7

    total_attempts = total_deliveries + total_failed_drops
    if total_attempts > 0:
        drop_failure_rate = total_failed_drops / total_attempts
        deduction += min(drop_failure_rate * 0.2, 0.2)

    raw = round(float(max(0.0, 1.0 - deduction)), 4)
    return float(min(0.99, max(0.01, raw)))



def grade(state: Dict[str, Any]) -> float:
    tasks = state.get("tasks", {})
    if not tasks:
        return 0.01

    total_tasks      = len(tasks)
    delivered        = len(_delivered_set(tasks))
    completion_ratio = delivered / total_tasks

    priority_score = _priority_score(tasks)

    steps_used       = state.get("step",      1)
    max_steps        = state.get("max_steps", 1)
    efficiency_ratio = max(0.0, 1.0 - (steps_used / max_steps) * 0.5) if max_steps > 0 else 0.5

    collisions       = state.get("collision_count",  0)
    battery_failures = state.get("battery_failures", 0)
    forced_rtb       = sum(
        (d.get("diagnostics", {}) if isinstance(d, dict) else {}).get("forced_rtb_count", 0)
        for d in state.get("drone_states", {}).values()
    )
    near_misses      = sum(
        (d.get("diagnostics", {}) if isinstance(d, dict) else {}).get("near_miss_count", 0)
        for d in state.get("drone_states", {}).values()
    )
    safety_deduction = (
        collisions       * _COLLISION_HIT
        + battery_failures * _BATTERY_FAIL_HIT
        + forced_rtb       * _FORCED_RTB_HIT
        + near_misses      * _NEAR_MISS_HIT
    )
    safety_factor = max(0.0, 1.0 - safety_deduction)

    drone_score = _drone_quality_score(state)

    score = (
        _WEIGHTS["completion_weight"] * completion_ratio
        + _WEIGHTS["efficiency_weight"] * efficiency_ratio
        + _WEIGHTS["safety_weight"]     * safety_factor
        + _WEIGHTS["priority_weight"]   * priority_score
        + _WEIGHTS["drone_weight"]      * drone_score
    )

    if not math.isfinite(score):
        score = 0.01
    return float(min(0.99, max(0.01, round(score, 6))))
