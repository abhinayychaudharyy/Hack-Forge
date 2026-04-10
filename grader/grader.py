from __future__ import annotations
from typing import Any, Dict

from env.models import TaskStatus



GRADE_PARAMS: Dict[str, Dict[str, float]] = {
    "easy": {
        "completion_weight": 0.60,
        "efficiency_weight": 0.15,
        "safety_weight":     0.10,
        "priority_weight":   0.05,
        "drone_weight":      0.10,   # hover stability, near-miss, delivery precision
    },
    "medium": {
        "completion_weight": 0.45,
        "efficiency_weight": 0.18,
        "safety_weight":     0.17,
        "priority_weight":   0.10,
        "drone_weight":      0.10,
    },
    "hard": {
        "completion_weight": 0.35,
        "efficiency_weight": 0.18,
        "safety_weight":     0.22,   # safety matters most on hard
        "priority_weight":   0.15,
        "drone_weight":      0.10,
    },
}

_COLLISION_HIT      = 0.10   # per collision event
_BATTERY_FAIL_HIT   = 0.15   # per battery failure (dead-mid-air)
_FORCED_RTB_HIT     = 0.05   # per forced RTB (low battery return)
_NEAR_MISS_HIT      = 0.03   # per near-miss step (obstacle < 1 cell)
_MOTOR_DEGRADED_HIT = 0.10   # per drone whose motor_health < 0.8



def _delivered_set(tasks: Dict[str, Any]) -> list:
    return [
        t for t in tasks.values()
        if t.get("status") in ("delivered", TaskStatus.DELIVERED)
    ]


def _priority_score(tasks: Dict[str, Any]) -> float:
    delivered = _delivered_set(tasks)
    if not delivered:
        return 0.0
    total_priority = sum(t.get("priority", 1) for t in delivered)
    max_possible   = 3 * len(delivered)
    return total_priority / max_possible if max_possible > 0 else 0.0


def _drone_quality_score(state: Dict[str, Any]) -> float:
    drone_states = state.get("drone_states", {})
    if not drone_states:
        return 1.0   # no drones → full score (not penalised)

    total_near_misses  = 0
    total_forced_rtb   = 0
    avg_motor_health   = 0.0
    avg_stab_score     = 0.0
    total_deliveries   = 0
    total_failed_drops = 0

    for drone in drone_states.values():
        diag  = drone.get("diagnostics", {}) if isinstance(drone, dict) else {}
        fl    = drone.get("flight",      {}) if isinstance(drone, dict) else {}

        total_near_misses  += diag.get("near_miss_count",   0)
        total_forced_rtb   += diag.get("forced_rtb_count",  0)
        avg_motor_health   += diag.get("motor_health",      1.0)
        avg_stab_score     += fl.get("hover_stability_score", 1.0)
        total_deliveries   += diag.get("total_deliveries",  0)
        total_failed_drops += diag.get("total_failed_deliveries", 0)

    n = len(drone_states)
    avg_motor_health /= n
    avg_stab_score   /= n

    deduction = 0.0
    deduction += min(total_near_misses  * _NEAR_MISS_HIT,   0.4)
    deduction += min(total_forced_rtb   * _FORCED_RTB_HIT,  0.3)

    if avg_motor_health < 0.8:          # degraded motors from collisions
        deduction += _MOTOR_DEGRADED_HIT * (0.8 - avg_motor_health) / 0.8

    if avg_stab_score < 0.7:
        deduction += 0.1 * (0.7 - avg_stab_score) / 0.7

    total_attempts = total_deliveries + total_failed_drops
    if total_attempts > 0:
        drop_failure_rate = total_failed_drops / total_attempts
        deduction += min(drop_failure_rate * 0.2, 0.2)

    return round(float(max(0.0, 1.0 - deduction)), 4)


def _count_drone_failures(state: Dict[str, Any]) -> int:
    total = 0
    for d in state.get("drone_states", {}).values():
        diag = d.get("diagnostics", {}) if isinstance(d, dict) else {}
        total += diag.get("forced_rtb_count", 0)
    return total


def _count_near_misses(state: Dict[str, Any]) -> int:
    total = 0
    for d in state.get("drone_states", {}).values():
        diag = d.get("diagnostics", {}) if isinstance(d, dict) else {}
        total += diag.get("near_miss_count", 0)
    return total



def grade(state: Dict[str, Any]) -> float:
    task_name = state.get("task_name", "easy")
    params    = GRADE_PARAMS.get(task_name, GRADE_PARAMS["easy"])
    tasks     = state.get("tasks", {})

    if not tasks:
        return 0.01

    total_tasks     = len(tasks)
    delivered       = len(_delivered_set(tasks))
    completion_ratio = delivered / total_tasks

    priority_score  = _priority_score(tasks)

    steps_used    = state.get("step",      1)
    max_steps     = state.get("max_steps", 1)
    efficiency_ratio = max(0.0, 1.0 - (steps_used / max_steps) * 0.5) if max_steps > 0 else 0.5

    collisions       = state.get("collision_count",    0)
    battery_failures = state.get("battery_failures",   0)
    forced_rtb       = _count_drone_failures(state)
    near_misses      = _count_near_misses(state)

    safety_deduction = (
        collisions       * _COLLISION_HIT
        + battery_failures * _BATTERY_FAIL_HIT
        + forced_rtb       * _FORCED_RTB_HIT
        + near_misses      * _NEAR_MISS_HIT
    )
    safety_factor = max(0.0, 1.0 - safety_deduction)

    drone_score = _drone_quality_score(state)

    score = (
        params["completion_weight"] * completion_ratio
        + params["efficiency_weight"] * efficiency_ratio
        + params["safety_weight"]     * safety_factor
        + params["priority_weight"]   * priority_score
        + params["drone_weight"]      * drone_score
    )

    return float(min(0.99, max(0.01, score)))


def detailed_report(state: Dict[str, Any]) -> Dict[str, Any]:
    task_name = state.get("task_name", "easy")
    tasks     = state.get("tasks", {})
    total     = len(tasks)
    delivered = len(_delivered_set(tasks))

    drone_states = state.get("drone_states", {})
    total_near_misses  = _count_near_misses(state)
    total_forced_rtb   = _count_drone_failures(state)
    drone_deliveries   = sum(
        (d.get("diagnostics", {}) if isinstance(d, dict) else {}).get("total_deliveries", 0)
        for d in drone_states.values()
    )
    drone_failed_drops = sum(
        (d.get("diagnostics", {}) if isinstance(d, dict) else {}).get("total_failed_deliveries", 0)
        for d in drone_states.values()
    )
    avg_motor_health = (
        sum((d.get("diagnostics", {}) if isinstance(d, dict) else {}).get("motor_health", 1.0)
            for d in drone_states.values()) / len(drone_states)
        if drone_states else 1.0
    )
    avg_stability = (
        sum((d.get("flight", {}) if isinstance(d, dict) else {}).get("hover_stability_score", 1.0)
            for d in drone_states.values()) / len(drone_states)
        if drone_states else 1.0
    )

    return {
        "task_name":            task_name,
        "steps_used":           state.get("step", 0),
        "max_steps":            state.get("max_steps", 0),

        "total_tasks":          total,
        "delivered":            delivered,
        "completion_ratio":     round(delivered / total, 4) if total > 0 else 0.0,
        "task_statuses":        {tid: t.get("status") for tid, t in tasks.items()},

        "collisions":           state.get("collision_count",  0),
        "battery_failures":     state.get("battery_failures", 0),
        "forced_rtb_events":    total_forced_rtb,
        "obstacle_near_misses": total_near_misses,

        "num_drones":           len(drone_states),
        "drone_deliveries":     drone_deliveries,
        "drone_failed_drops":   drone_failed_drops,
        "avg_motor_health":     round(avg_motor_health, 3),
        "avg_hover_stability":  round(avg_stability, 3),
        "drone_quality_score":  _drone_quality_score(state),

        "final_score":          grade(state),
    }
