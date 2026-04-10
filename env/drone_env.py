from __future__ import annotations
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    AgentState, DroneAgentState, AgentType,
    AeroSyncAction, AeroSyncObservation, AeroSyncReward,
    ActionType, Direction, EpisodeInfo,
    Position, TaskState, TaskStatus,
    FlightMode, WindCondition,
    DroneFlightParams, DroneDiagnostics, DroneTiltState,
    FlightWaypoint, DroneFlightPath, GridCell,
)

from typing import cast

R_DELIVERY         =  1000.0  # Goal reached
R_PICKUP           =   200.0  # Significant partial progress signal
R_DISPATCH         =   100.0  
P_COLLISION        = -1000.0  # Safety first
P_BATTERY          =  -500.0  

P_DELAY            =    -0.5  # Per-step penalty
P_IDLE             =    -1.0  # Discourage idling
B_PRECISION_DROP   =    50.0  
B_EFFICIENT_RTB    =    50.0  
B_CLEAN_PATH       =    50.0  
B_OPTIMAL_PATH     =    50.0  
B_WAYPOINT         =     5.0  

P_FORCED_LANDING   =  -200.0  
P_HOVER_DRIFT      =    -2.0  
P_OBSTACLE_NEAR    =    -5.0  
P_OBSTACLE_MISS    =   -20.0  
P_TTC_CRITICAL     =   -20.0  
P_TTC_WARNING      =    -5.0  

B_HOVER_STABILITY  =     0.0  
B_SAFE_CLEAR       =     0.0  
B_TTC_SAFE         =     0.0  
B_SMOOTH_YAW       =     0.0  
B_COLLISION_AVOID  =     0.0
P_YAW_THRASH       =    -5.0
P_YAW_OVERSHOOT    =    -2.0

class DroneEnv:

    def __init__(self, task_config: Dict[str, Any]):
        self.config = task_config
        self._initial_config = copy.deepcopy(task_config)

        self.W = task_config["grid_width"]
        self.H = task_config["grid_height"]
        self.max_steps = task_config["max_steps"]
        self.task_name = task_config.get("task_name", "unknown")

        self.obstacles: set = set(tuple(o) for o in task_config.get("obstacles", []))
        self.dispatch_zones: set = set(tuple(d) for d in task_config.get("dispatch_zones", []))
        self.charging_positions: List[Position] = [
            Position(x=c[0], y=c[1], z=0)
            for c in task_config.get("charging_stations", [])
        ]
        self.wind_condition: WindCondition = WindCondition.MODERATE

        self._grid_map: Dict[str, GridCell] = self._build_grid_map()

        self._step: int = 0
        self._agents: Dict[str, AgentState] = {}
        self._drone_states: Dict[str, DroneAgentState] = {}
        self._tasks: Dict[str, TaskState] = {}
        self._dispatch_queue: List[str] = []
        self._episode_rewards: List[float] = []
        self._collision_count: int = 0
        self._battery_failures: int = 0
        self._drone_near_miss_steps: Dict[str, int] = {}

    # OpenEnv API

    def reset(self) -> AeroSyncObservation:
        cfg = copy.deepcopy(self._initial_config)
        self._step = 0
        self._dispatch_queue = []
        self._episode_rewards = []
        self._collision_count = 0
        self._battery_failures = 0

        # Per-episode variability so repeated runs don't converge to identical scores.
        # This changes only flight conditions (wind/drag), not the task layout.
        # Strong stochasticity: wind is sampled per episode and can shift during the episode.
        # This is intentional so repeated runs produce clearly different outcomes/scores.
        self.wind_condition = random.choices(
            population=[WindCondition.CALM, WindCondition.LIGHT, WindCondition.MODERATE, WindCondition.STRONG],
            weights=[0.05, 0.15, 0.35, 0.45],
            k=1,
        )[0]

        self._agents = {}
        for r in cfg.get("robots", []):
            self._agents[r["id"]] = AgentState(
                agent_id=r["id"],
                agent_type=AgentType.ROBOT,
                position=Position(x=r["start_x"], y=r["start_y"], z=0),
                battery=r.get("battery", 100.0),
            )

        self._drone_states: Dict[str, DroneAgentState] = {}
        self._drone_near_miss_steps: Dict[str, int] = {}
        self._drone_prev_yaw: Dict[str, float] = {}
        self._prev_dist_dict: Dict[str, float] = {}

        for d in cfg.get("drones", []):
            drag = {
                WindCondition.CALM: 1.00,
                WindCondition.LIGHT: 1.20,
                WindCondition.MODERATE: 1.55,
                WindCondition.STRONG: 2.10,
            }.get(self.wind_condition, 1.55)
            fp = DroneFlightParams(
                battery_capacity=100.0,
                max_altitude=5,
                wind_condition=self.wind_condition,
                wind_drag_factor=drag,
                tilt=DroneTiltState(),
            )
            diag = DroneDiagnostics(drone_id=d["id"])
            self._drone_states[d["id"]] = DroneAgentState(
                agent_id=d["id"],
                position=Position(x=d["start_x"], y=d["start_y"], z=1),
                battery=d.get("battery", 100.0),
                flight=fp,
                diagnostics=diag,
            )
            self._drone_near_miss_steps[d["id"]] = 0
            self._drone_prev_yaw[d["id"]] = 0.0
            self._prev_dist_dict[d["id"]] = 999.0

        self._tasks = {}
        for t in cfg.get("tasks", []):
            pickup_pos   = Position(**t["pickup"])
            delivery_pos = Position(**t["delivery"])
            dispatch_raw = t.get("dispatch", t["pickup"])
            dispatch_pos = Position(**dispatch_raw)
            task = TaskState(
                task_id=t["id"],
                item_name=t["item"],
                pickup_location=pickup_pos,
                dispatch_location=dispatch_pos,
                delivery_location=delivery_pos,
                priority=t.get("priority", 1),
                status=TaskStatus.DISPATCHED,
            )
            self._tasks[t["id"]] = task
            self._dispatch_queue.append(t["id"])

        return self._build_observation(reward=0.0)

    def step(self, action: AeroSyncAction) -> Tuple[AeroSyncObservation, float, bool, EpisodeInfo]:
        rb = AeroSyncReward()
        info = EpisodeInfo()
        self._step += 1

        agent = self._agents.get(action.agent_id) or self._drone_states.get(action.agent_id)
        if agent is None:
            info.message = f"Unknown agent: {action.agent_id}"
            return self._build_observation(0.0), 0.0, self._is_done(), info

        agent.steps_taken += 1
        is_drone = agent.agent_type == AgentType.DRONE

        if agent.battery <= 0.0 and action.action_type != ActionType.CHARGE:
            if is_drone and agent.position.z > 0:
                rb.forced_landing_penalty += P_FORCED_LANDING
                agent.diagnostics.motor_health = max(0.0, agent.diagnostics.motor_health - 0.5)
                agent.position.z = 0
            rb.battery_penalty += P_BATTERY
            info.battery_failures.append(action.agent_id)
            self._battery_failures += 1
            agent.is_idle = True
            agent.carrying_task_id = None
            total = P_BATTERY + P_FORCED_LANDING if is_drone and agent.position.z >= 0 else P_BATTERY
            self._episode_rewards.append(total)
            return self._build_observation(total), total, self._is_done(), info

        _waypoints = getattr(action, "waypoints", None)
        if is_drone and _waypoints:
            self._accept_flight_plan(agent, _waypoints)

        act = action.action_type
        if act == ActionType.MOVE:
            rb, info = self._do_move(agent, action.direction, rb, info)
        elif act == ActionType.HOVER and is_drone:
            rb, info = self._do_hover(agent, rb, info)
        elif act == ActionType.ASCEND and is_drone:
            rb, info = self._do_ascend(agent, action.target_altitude, rb, info)
        elif act == ActionType.DESCEND and is_drone:
            rb, info = self._do_descend(agent, action.target_altitude, rb, info)
        elif act == ActionType.RETURN_TO_BASE and is_drone:
            rb, info = self._do_rtb(agent, rb, info)
        elif act == ActionType.PICK:
            rb = self._do_pick(agent, action.task_id, rb, info)
        elif act == ActionType.PLACE:
            rb = self._do_place(agent, rb, info)
        elif act == ActionType.CHARGE:
            self._do_charge(agent)
        elif act == ActionType.ASSIGN_TASK:
            self._do_assign(agent, action.task_id)

        if is_drone:
            rb, info = self._update_drone_step(agent, rb, info)

        self._apply_battery_decay(agent)

        if is_drone:
            self._advance_waypoints(agent, info)

        pending = sum(1 for t in self._tasks.values()
                      if t.status not in (TaskStatus.DELIVERED, TaskStatus.FAILED))
        rb.delay_penalty += P_DELAY * pending

        all_agents = list(self._agents.values()) + list(self._drone_states.values())
        idle_agents = sum(1 for a in all_agents if a.is_idle and a.battery > 0)
        rb.idle_penalty += P_IDLE * idle_agents

        total_reward = sum([
            rb.delivery_bonus, rb.dispatch_bonus, rb.pickup_bonus,
            rb.collision_penalty, rb.battery_penalty, rb.delay_penalty, rb.idle_penalty,
            rb.hover_stability_bonus, rb.hover_stability_loss_penalty, rb.hover_drift_penalty,
            rb.tilt_efficiency_bonus, rb.over_tilt_penalty, rb.unnecessary_banking_penalty,
            rb.tilt_stability_loss_penalty, rb.smooth_yaw_bonus,
            rb.precision_landing_bonus, rb.imprecise_drop_penalty, rb.repeated_attempt_penalty,
            rb.efficient_rtb_bonus, rb.forced_landing_penalty,
            rb.battery_conservation_bonus, rb.unnecessary_hover_penalty,
            rb.optimal_path_bonus, rb.replanning_penalty,
            rb.waypoint_reached_bonus, rb.speed_efficiency_bonus,
            rb.obstacle_proximity_penalty, rb.obstacle_near_miss_penalty,
            rb.safe_clearance_bonus, rb.collision_avoidance_bonus,
            rb.ttc_critical_penalty, rb.ttc_warning_penalty, rb.ttc_safe_bonus,
            rb.high_speed_near_obstacle_penalty, rb.speed_reduction_near_obstacle_bonus,
            rb.waypoint_obstacle_clear_bonus, rb.waypoint_obstacle_penalty,
            rb.blocked_waypoint_penalty, rb.obstacle_replan_penalty,
            rb.clean_path_completion_bonus,
        ])
        rb.total = total_reward
        info.reward_breakdown = rb

        self._episode_rewards.append(total_reward)
        return self._build_observation(total_reward), total_reward, self._is_done(), info

    def state(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "max_steps": self.max_steps,
            "task_name": self.task_name,
            "agents": {k: v.model_dump() for k, v in self._agents.items()},
            "drone_states": {k: v.model_dump() for k, v in self._drone_states.items()},
            "tasks": {k: v.model_dump() for k, v in self._tasks.items()},
            "dispatch_queue": list(self._dispatch_queue),
            "collision_count": self._collision_count,
            "battery_failures": self._battery_failures,
            "episode_rewards": list(self._episode_rewards),
            "grid": {"width": self.W, "height": self.H},
        }


    def _do_move(self, agent: AgentState, direction: Optional[str],
             rb: AeroSyncReward, info: EpisodeInfo) -> Tuple[AeroSyncReward, EpisodeInfo]:

        if direction is None:
            return rb, info

        dx, dy, dz = 0, 0, 0
        if direction in (Direction.NORTH, "north"):   dy = -1
        elif direction in (Direction.SOUTH, "south"): dy =  1
        elif direction in (Direction.EAST,  "east"):  dx =  1
        elif direction in (Direction.WEST,  "west"):  dx = -1
        elif direction in (Direction.UP,    "up"):    dz =  1
        elif direction in (Direction.DOWN,  "down"):  dz = -1

        nx, ny, nz = agent.position.x + dx, agent.position.y + dy, agent.position.z + dz

        if not (0 <= nx < self.W and 0 <= ny < self.H):
            return rb, info

        if agent.agent_type == AgentType.ROBOT:
            if nz != 0:
                return rb, info
            if (nx, ny) in self.obstacles:
                return rb, info

        all_agents = list(self._agents.values()) + list(self._drone_states.values())
        for other in all_agents:
            if other.agent_id == agent.agent_id:
                continue
            if other.position.x == nx and other.position.y == ny and other.position.z == nz:
                rb.collision_penalty += P_COLLISION
                info.collision_events.append(f"{agent.agent_id} collided with {other.agent_id} at ({nx},{ny},{nz})")
                self._collision_count += 1

                if agent.agent_type == AgentType.DRONE:
                    drone = cast(DroneAgentState, agent)
                    drone.diagnostics.motor_health = max(0.0, drone.diagnostics.motor_health - 0.2)

                if other.agent_type == AgentType.DRONE:
                    other_drone = cast(DroneAgentState, other)
                    other_drone.diagnostics.motor_health = max(0.0, other_drone.diagnostics.motor_health - 0.2)

                return rb, info

        agent.position = Position(x=nx, y=ny, z=nz)
        agent.is_idle = False

        if agent.agent_type == AgentType.DRONE:
            drone = cast(DroneAgentState, agent)
            drone.flight.flight_mode = FlightMode.CRUISE
            drone.diagnostics.total_distance_flown += 1.0
            drone.flight.current_speed = drone.flight.max_speed

        return rb, info

    def _do_hover(self, agent: DroneAgentState,
                  rb: AeroSyncReward, info: EpisodeInfo) -> Tuple[AeroSyncReward, EpisodeInfo]:
        agent.flight.flight_mode = FlightMode.HOVER
        agent.is_idle = False
        wind_penalty = {
            WindCondition.CALM: 0.05,
            WindCondition.LIGHT: 0.25,
            WindCondition.MODERATE: 0.55,
            WindCondition.STRONG: 0.85,
        }.get(self.wind_condition, 0.55)
        tilt_cost = float(getattr(agent.flight.tilt, "tilt_stability_cost", 0.0) or 0.0)
        gust = random.uniform(-0.25, 0.25)
        # Stability reflects wind + attitude. Avoid coupling to map centre distance.
        agent.flight.hover_stability_score = max(
            0.0, min(1.0, 1.0 - wind_penalty - 0.50 * tilt_cost + gust)
        )

        wf = {WindCondition.CALM: 0, WindCondition.LIGHT: 0.05,
              WindCondition.MODERATE: 0.1, WindCondition.STRONG: 0.2}.get(self.wind_condition, 0)
        agent.flight.hover_drift_x = random.uniform(-wf, wf)
        agent.flight.hover_drift_y = random.uniform(-wf, wf)
        agent.flight.hover_drift_z = random.uniform(-wf * 0.5, wf * 0.5)

        if agent.flight.hover_stability_score >= 0.8:
            rb.hover_stability_bonus += B_HOVER_STABILITY
        elif agent.flight.hover_stability_score < agent.flight.stability_threshold:
            rb.hover_stability_loss_penalty += -1.0
            info.drone_stability_warnings.append(agent.agent_id)

        total_drift = (abs(agent.flight.hover_drift_x) + abs(agent.flight.hover_drift_y) +
                       abs(agent.flight.hover_drift_z))
        rb.hover_drift_penalty += P_HOVER_DRIFT * total_drift
        return rb, info

    def _do_ascend(self, agent: DroneAgentState, target_altitude: Optional[int],
                   rb: AeroSyncReward, info: EpisodeInfo) -> Tuple[AeroSyncReward, EpisodeInfo]:
        agent.flight.flight_mode = FlightMode.TAKEOFF
        agent.is_idle = False
        target_z = min(target_altitude if target_altitude is not None else agent.flight.max_altitude,
                       agent.flight.max_altitude)
        if agent.position.z < target_z:
            agent.position.z += 1
        return rb, info

    def _do_descend(self, agent: DroneAgentState, target_altitude: Optional[int],
                    rb: AeroSyncReward, info: EpisodeInfo) -> Tuple[AeroSyncReward, EpisodeInfo]:
        agent.flight.flight_mode = FlightMode.LANDING
        agent.is_idle = False
        target_z = max(target_altitude if target_altitude is not None else 0, 0)
        if agent.position.z > target_z:
            agent.position.z -= 1
        return rb, info

    def _do_rtb(self, agent: DroneAgentState,
                rb: AeroSyncReward, info: EpisodeInfo) -> Tuple[AeroSyncReward, EpisodeInfo]:
        agent.flight.flight_mode = FlightMode.RETURN
        info.drone_rtb_events.append(agent.agent_id)
        agent.diagnostics.forced_rtb_count += 1
        if agent.battery > agent.flight.critical_battery_threshold:
            rb.efficient_rtb_bonus += B_EFFICIENT_RTB
        if self.charging_positions:
            closest = min(self.charging_positions,
                          key=lambda c: abs(c.x - agent.position.x) + abs(c.y - agent.position.y))
            dx = 1 if closest.x > agent.position.x else (-1 if closest.x < agent.position.x else 0)
            dy = 1 if closest.y > agent.position.y else (-1 if closest.y < agent.position.y else 0)
            if dx != 0:
                d = "east" if dx == 1 else "west"
            elif dy != 0:
                d = "south" if dy == 1 else "north"
            else:
                d = None
            if d:
                rb, info = self._do_move(agent, d, rb, info)
        return rb, info

    def _do_pick(self, agent: AgentState, task_id: Optional[str],
                 rb: AeroSyncReward, info: EpisodeInfo) -> AeroSyncReward:
        if task_id is None or agent.carrying_task_id is not None:
            return rb
        task = self._tasks.get(task_id)
        if task is None:
            return rb
        ax, ay, az = agent.position.x, agent.position.y, agent.position.z

        if agent.agent_type == AgentType.ROBOT:
            if (task.status == TaskStatus.PENDING and
                    task.assigned_robot == agent.agent_id and
                    ax == task.pickup_location.x and ay == task.pickup_location.y):
                task.status = TaskStatus.PICKED
                agent.carrying_task_id = task_id
                agent.is_idle = False
                rb.pickup_bonus += R_PICKUP
                info.message = f"{agent.agent_id} picked task {task_id}"

        elif agent.agent_type == AgentType.DRONE:
            at_dispatch_zone   = (ax, ay) in self.dispatch_zones
            at_task_dispatch   = (ax == task.dispatch_location.x and ay == task.dispatch_location.y)
            at_task_pickup     = (ax == task.pickup_location.x   and ay == task.pickup_location.y)
            can_pick = at_dispatch_zone or at_task_dispatch or at_task_pickup
            if (task.status == TaskStatus.DISPATCHED and task_id in self._dispatch_queue and
                    can_pick and az == 0):
                task.status = TaskStatus.IN_FLIGHT
                task.assigned_drone = agent.agent_id
                agent.carrying_task_id = task_id
                agent.is_idle = False
                agent.flight.current_payload_kg = 1.0
                agent.flight.payload_drag_factor = 1.2
                self._dispatch_queue.remove(task_id)
                rb.pickup_bonus += R_PICKUP
        return rb

    def _do_place(self, agent: AgentState,
                  rb: AeroSyncReward, info: EpisodeInfo) -> AeroSyncReward:
        if agent.carrying_task_id is None:
            return rb
        task = self._tasks.get(agent.carrying_task_id)
        if task is None:
            return rb
        ax, ay, az = agent.position.x, agent.position.y, agent.position.z

        if agent.agent_type == AgentType.ROBOT:
            if task.status == TaskStatus.PICKED and (ax, ay) in self.dispatch_zones:
                task.status = TaskStatus.DISPATCHED
                self._dispatch_queue.append(task.task_id)
                agent.carrying_task_id = None
                agent.is_idle = True
                rb.dispatch_bonus += R_DISPATCH
                info.message = f"Task {task.task_id} dispatched"

        elif agent.agent_type == AgentType.DRONE:
            if (task.status == TaskStatus.IN_FLIGHT and
                    ax == task.delivery_location.x and ay == task.delivery_location.y):
                if az == 0:
                    prec = agent.flight.hover_stability_score
                    if prec >= agent.flight.delivery_precision_threshold:
                        task.status = TaskStatus.DELIVERED
                        task.completed_at_step = self._step
                        agent.carrying_task_id = None
                        agent.is_idle = True
                        agent.flight.current_payload_kg = 0.0
                        agent.flight.payload_drag_factor = 1.0
                        agent.diagnostics.total_deliveries += 1
                        rb.delivery_bonus += R_DELIVERY
                        rb.precision_landing_bonus += B_PRECISION_DROP * prec
                        if self._drone_near_miss_steps.get(agent.agent_id, 0) == 0:
                            rb.clean_path_completion_bonus += B_CLEAN_PATH
                            info.clean_path_drones.append(agent.agent_id)
                        info.completed_tasks.append(task.task_id)
                        info.message = f"Task {task.task_id} delivered! Precision={prec:.2f}"
                    else:
                        agent.flight.delivery_attempts += 1
                        rb.imprecise_drop_penalty += -3.0
                        if agent.flight.delivery_attempts > 1:
                            rb.repeated_attempt_penalty += -2.0 * (agent.flight.delivery_attempts - 1)
                        info.drone_precision_failures.append(agent.agent_id)
                        if agent.flight.delivery_attempts >= agent.flight.max_delivery_attempts:
                            task.status = TaskStatus.FAILED
                            agent.diagnostics.total_failed_deliveries += 1
                            agent.is_idle = True
                            agent.carrying_task_id = None
        return rb

    def _do_charge(self, agent: AgentState):
        ax, ay, az = agent.position.x, agent.position.y, agent.position.z
        at_station = any(c.x == ax and c.y == ay for c in self.charging_positions)
        if at_station and az == 0:
            if agent.agent_type == AgentType.DRONE:
                agent.battery = min(agent.flight.battery_capacity,
                                    agent.battery + agent.flight.charge_rate_per_step)
                agent.flight.flight_mode = FlightMode.LANDING
                agent.diagnostics.last_recharge_step = self._step
            else:
                agent.battery = min(100.0, agent.battery + 20.0)
            agent.is_charging = True
            agent.is_idle = True
        else:
            agent.is_charging = False

    def _do_assign(self, agent: AgentState, task_id: Optional[str]):
        if task_id is None or agent.agent_type != AgentType.ROBOT:
            return
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING and task.assigned_robot is None:
            task.assigned_robot = agent.agent_id
            agent.is_idle = False


    def _update_drone_step(self, drone: DroneAgentState,
                       rb: AeroSyncReward, info: EpisodeInfo) -> Tuple[AeroSyncReward, EpisodeInfo]:
        diag = drone.diagnostics
        fp = drone.flight

        # Update stability each step so grading reflects flight conditions even if the
        # controller doesn't explicitly call HOVER.
        # Occasionally shift wind condition mid-episode (strong randomness).
        if random.random() < 0.08:
            if self.wind_condition == WindCondition.CALM:
                self.wind_condition = WindCondition.LIGHT
            elif self.wind_condition == WindCondition.LIGHT:
                self.wind_condition = random.choice([WindCondition.CALM, WindCondition.MODERATE])
            elif self.wind_condition == WindCondition.MODERATE:
                self.wind_condition = random.choice([WindCondition.LIGHT, WindCondition.STRONG])
            else:
                self.wind_condition = random.choice([WindCondition.MODERATE, WindCondition.STRONG])
            fp.wind_condition = self.wind_condition
            fp.wind_drag_factor = {
                WindCondition.CALM: 1.00,
                WindCondition.LIGHT: 1.20,
                WindCondition.MODERATE: 1.55,
                WindCondition.STRONG: 2.10,
            }.get(self.wind_condition, 1.55)

        wind_penalty = {
            WindCondition.CALM: 0.05,
            WindCondition.LIGHT: 0.25,
            WindCondition.MODERATE: 0.55,
            WindCondition.STRONG: 0.85,
        }.get(self.wind_condition, 0.55)
        tilt_cost = float(getattr(fp.tilt, "tilt_stability_cost", 0.0) or 0.0)
        gust = random.uniform(-0.20, 0.20)
        fp.hover_stability_score = max(0.0, min(1.0, 1.0 - wind_penalty - 0.40 * tilt_cost + gust))

        min_dist = 999.0
        for ox, oy in self.obstacles:
            d = abs(ox - drone.position.x) + abs(oy - drone.position.y)
            if d < min_dist:
                min_dist = d
        diag.nearest_obstacle_dist = float(min_dist)
        diag.obstacle_detected = min_dist <= 1.5
        diag.collision_risk = max(0.0, min(1.0, 1.0 - min_dist / 5.0))

        if min_dist < 1.0:
            rb.obstacle_near_miss_penalty += P_OBSTACLE_MISS
            if drone.agent_id not in info.obstacle_near_misses:
                info.obstacle_near_misses.append(drone.agent_id)
            self._drone_near_miss_steps[drone.agent_id] = \
                self._drone_near_miss_steps.get(drone.agent_id, 0) + 1
            diag.near_miss_count += 1
        elif min_dist < 2.0:
            rb.obstacle_proximity_penalty += P_OBSTACLE_NEAR
        elif min_dist > 3.0:
            rb.safe_clearance_bonus += B_SAFE_CLEAR

        prev_dist = self._prev_dist_dict.get(drone.agent_id, 999.0)
        closing_speed = prev_dist - min_dist
        self._prev_dist_dict[drone.agent_id] = min_dist

        if closing_speed > 0:
            ttc = min_dist / closing_speed
            inv_ttc = min(1.0 / ttc, 5.0)

            if ttc < 1.0 and min_dist >= 1.0:
                rb.ttc_critical_penalty += P_TTC_CRITICAL * inv_ttc
                if drone.agent_id not in info.obstacle_near_misses:
                    info.obstacle_near_misses.append(drone.agent_id)
            elif ttc < 2.0:
                rb.ttc_warning_penalty += P_TTC_WARNING * inv_ttc
            else:
                rb.ttc_safe_bonus += B_TTC_SAFE  # 0.0, safe to keep
        else:
            if prev_dist < 3.0:
                rb.collision_avoidance_bonus += B_COLLISION_AVOID

        if fp.current_speed > 1.5 and min_dist < 3.0:
            rb.high_speed_near_obstacle_penalty += -2.0

        tilt = getattr(fp, "tilt", None) or DroneTiltState()
        if abs(tilt.pitch) > tilt.max_pitch or abs(tilt.roll) > tilt.max_roll:
            rb.over_tilt_penalty += -2.0
        if tilt.tilt_stability_cost > 0:
            rb.tilt_stability_loss_penalty += -tilt.tilt_stability_cost
        if tilt.is_banking and fp.flight_mode != FlightMode.CRUISE:
            rb.unnecessary_banking_penalty += -1.0

        prev_yaw = self._drone_prev_yaw.get(drone.agent_id, tilt.yaw)
        yaw_delta = abs(tilt.yaw - prev_yaw)
        yaw_delta = min(yaw_delta, 360.0 - yaw_delta) if yaw_delta > 180 else yaw_delta

        if yaw_delta > 0:
            if fp.flight_mode == FlightMode.CRUISE:
                if yaw_delta > tilt.yaw_rate * 3:
                    rb.smooth_yaw_bonus += P_YAW_THRASH
            elif fp.flight_mode in (FlightMode.HOVER, FlightMode.LANDING):
                if yaw_delta > tilt.yaw_rate:
                    rb.smooth_yaw_bonus += P_YAW_OVERSHOOT

        self._drone_prev_yaw[drone.agent_id] = tilt.yaw

        if drone.battery <= fp.low_battery_threshold and fp.flight_mode != FlightMode.RETURN:
            info.drone_rtb_events.append(f"{drone.agent_id}:low_battery_warning")

        return rb, info


    def _apply_battery_decay(self, agent: AgentState):
        if agent.is_charging:
            return
        if agent.agent_type == AgentType.DRONE:
            f = agent.flight
            mode = f.flight_mode
            if mode == FlightMode.CRUISE:
                drain = f.battery_drain_per_move * f.payload_drag_factor * f.wind_drag_factor
            elif mode == FlightMode.HOVER:
                drain = f.battery_drain_hover * f.wind_drag_factor
            elif mode in (FlightMode.TAKEOFF, FlightMode.RETURN):
                drain = f.battery_drain_ascend
            elif mode == FlightMode.LANDING:
                drain = f.battery_drain_ascend * 0.8
            else:
                drain = f.battery_drain_idle
            agent.battery = max(0.0, agent.battery - drain)
        else:
            agent.battery = max(0.0, agent.battery - 0.3)


    def _accept_flight_plan(self, drone: DroneAgentState, waypoints: List[FlightWaypoint]):
        plan = DroneFlightPath(
            drone_id=drone.agent_id,
            waypoints=waypoints,
            total_estimated_battery=sum(w.estimated_battery_cost for w in waypoints),
            total_estimated_steps=len(waypoints),
        )
        for wp in plan.waypoints:
            if (wp.position.x, wp.position.y) in self.obstacles:
                wp.was_blocked = True
                plan.path_is_valid = False
        if not plan.path_is_valid:
            plan.replanned_count += 1
        drone.flight.flight_plan = plan

    def _advance_waypoints(self, drone: DroneAgentState, info: EpisodeInfo):
        plan = getattr(drone.flight, "flight_plan", None)
        if plan is None or plan.current_waypoint_idx >= len(plan.waypoints):
            return
        wp = plan.waypoints[plan.current_waypoint_idx]
        pos = drone.position
        if pos.x == wp.position.x and pos.y == wp.position.y:
            wp.is_reached = True
            wp.reached_at_step = self._step
            wp.obstacle_dist_on_arrival = drone.diagnostics.nearest_obstacle_dist
            wp.arrival_was_clean = wp.obstacle_dist_on_arrival > 2.0 and not wp.was_blocked
            plan.current_waypoint_idx += 1


    def _is_done(self) -> bool:
        if self._step >= self.max_steps:
            return True
        return all(t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
                   for t in self._tasks.values())

    def _build_grid_map(self) -> Dict[str, GridCell]:
        grid: Dict[str, GridCell] = {}
        dispatch_set = set(tuple(d) for d in self._initial_config.get("dispatch_zones", []))
        charging_set = set(tuple(c) for c in self._initial_config.get("charging_stations", []))
        shelves = set()
        for t in self._initial_config.get("tasks", []):
            p = t.get("pickup", {})
            shelves.add((p.get("x", 0), p.get("y", 0)))

        for y in range(self._initial_config.get("grid_height", 1)):
            for x in range(self._initial_config.get("grid_width", 1)):
                for z in (0, 1):
                    key = f"{x},{y},{z}"
                    grid[key] = GridCell(
                        x=x, y=y, z=z,
                        is_obstacle=(x, y) in self.obstacles and z == 0,
                        is_dispatch=(x, y) in dispatch_set and z == 0,
                        is_charging=(x, y) in charging_set and z == 0,
                        is_shelf=(x, y) in shelves and z == 0,
                    )
        return grid

    def _build_observation(self, reward: float) -> AeroSyncObservation:
        delivered = sum(1 for t in self._tasks.values() if t.status == TaskStatus.DELIVERED)
        total_tasks = len(self._tasks)

        metrics = {
            "completion_rate": round(delivered / total_tasks, 3) if total_tasks else 0.0,
            "collisions": float(self._collision_count),
            "battery_failures": float(self._battery_failures),
            "steps_used": float(self._step),
            "tasks_delivered": float(delivered),
            "tasks_total": float(total_tasks),
        }

        obs_agents = {k: AgentState(**v.model_dump()) for k, v in self._agents.items()}
        obs_drones = {k: DroneAgentState(**v.model_dump()) for k, v in self._drone_states.items()}

        grid_snap = copy.copy(self._grid_map)
        for cell in grid_snap.values():
            cell.occupant_id = None
        for a in list(self._agents.values()) + list(self._drone_states.values()):
            key = f"{a.position.x},{a.position.y},{a.position.z}"
            if key in grid_snap:
                grid_snap[key].occupant_id = a.agent_id

        return AeroSyncObservation(
            step=self._step,
            max_steps=self.max_steps,
            agents=obs_agents,
            drone_states=obs_drones,
            grid=grid_snap,
            tasks=copy.deepcopy(self._tasks),
            dispatch_queue=list(self._dispatch_queue),
            charging_stations=list(self.charging_positions),
            grid_width=self.W,
            grid_height=self.H,
            reward=reward,
            done=self._is_done(),
            task_name=self.task_name,
            metrics=metrics,
        )

    def bfs_path(self, start: Position, goal: Position, agent_type: AgentType) -> List[str]:
        from collections import deque
        if start.x == goal.x and start.y == goal.y:
            return []
        visited = {(start.x, start.y)}
        queue: deque = deque([((start.x, start.y), [])])
        dirs = [("north", 0, -1), ("south", 0, 1), ("east", 1, 0), ("west", -1, 0)]
        while queue:
            (cx, cy), path = queue.popleft()
            for d_name, dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.W and 0 <= ny < self.H):
                    continue
                if (nx, ny) in visited:
                    continue
                if agent_type == AgentType.ROBOT and (nx, ny) in self.obstacles:
                    continue
                new_path = path + [d_name]
                if nx == goal.x and ny == goal.y:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
        return []