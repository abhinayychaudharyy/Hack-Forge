from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.drone_env import DroneEnv
from env.models import AeroSyncAction, ActionType, Direction, TaskStatus
from grader.grader import grade, detailed_report
from tasks.easy   import get_config as easy_config
from tasks.medium import get_config as medium_config
from tasks.hard   import get_config as hard_config

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://vj-ai27-hack-forge.hf.space")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

BENCHMARK    = "aerosync-ai"

DEFAULT_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "12.0"))
DEFAULT_SLEEP_MS = int(os.environ.get("SLEEP_MS", "120"))
DEFAULT_PRINT_STATE_EVERY = int(os.environ.get("PRINT_STATE_EVERY", "5"))


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = "true" if done else "false"
    err_str = str(error) if error else "null"
    if hasattr(action, 'model_dump'):
        ad = action.model_dump()
        action_str = f"{ad['action_type']}({ad.get('direction') or ad.get('task_id') or ''})"
    else:
        action_str = str(action)
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={err_str}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


def build_system_prompt(task_name: str) -> str:
    return """You are the AeroSync AI drone controller.
GRID SYSTEM:
Top-Left is (0,0). X increases East (Right), Y increases South (Down).
- MOVE SOUTH to go from Y=0 to Y=2.
- MOVE NORTH to go from Y=2 to Y=0.
- MOVE EAST to go from X=0 to X=5.
- MOVE WEST to go from X=5 to X=0.

DRONE PIPELINE:
1. Move to pickup(x,y) at z=1.
2. Descend to z=0.
3. PICK(task_id).
4. Ascend to z=1.
5. Move to delivery(x,y) at z=1.
6. Descend to z=0.
7. HOVER once.
8. PLACE.

Respond with exactly ONE JSON object. No explanation.
Example: {"agent_id": "drone_0", "action_type": "move", "direction": "south"}"""


def _clean_content(content: str) -> str:
    raw = content
    if "<think>" in raw:
        end = raw.find("</think>")
        raw = raw[end+8:].strip() if end != -1 else raw[raw.find("<think>")+7:].strip()
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"): p = p[4:].strip()
            if p.startswith("{"): return p
    start = raw.find("{")
    if start == -1: return ""
    raw = raw[start:]
    depth, end_idx = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return raw[:i+1]
    return ""


def call_llm(client: OpenAI, messages: List[Dict]) -> Optional[Dict]:
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.4,
                max_tokens=1024,
                timeout=DEFAULT_TIMEOUT_S,
            )
            c = resp.choices[0].message.content.strip()
            j = _clean_content(c)
            if j: return json.loads(j)
        except: time.sleep(1)
    return None


def _pick_best_drone(obs: Dict[str, Any]) -> str:
    """Prefer a drone already carrying; else the best drone for queued dispatch pickups."""
    drones, tasks = obs.get("drone_states", {}), obs.get("tasks", {})
    for did, d in drones.items():
        if d.get("carrying_task_id"):
            return did

    dq = obs.get("dispatch_queue") or []
    waiting: List[Dict[str, Any]] = []
    for tid in dq:
        t = tasks.get(tid)
        if t and t.get("status") == "dispatched":
            waiting.append(t)

    if waiting:
        waiting.sort(key=lambda t: -t.get("priority", 1))
        t = waiting[0]
        dl = t.get("dispatch_location") or t["pickup_location"]
        tx, ty = dl["x"], dl["y"]
        best_did, min_dist = None, 1e9
        for did, d in drones.items():
            if d["battery"] < 15:
                continue
            dist = abs(d["position"]["x"] - tx) + abs(d["position"]["y"] - ty)
            if dist < min_dist:
                min_dist, best_did = dist, did
        if best_did:
            return best_did

    return next(iter(drones.keys()), "drone_0")


def parse_action(raw: Optional[Dict], obs: Dict[str, Any]) -> AeroSyncAction:
    suggested = _pick_best_drone(obs)
    if not raw: return _fallback_action(suggested, obs)
    aid = raw.get("agent_id", suggested)
    if aid not in obs["drone_states"]: aid = suggested
    if obs["drone_states"][aid]["battery"] < 15: return AeroSyncAction(agent_id=aid, action_type="return_to_base")
    atype = raw.get("action_type", "wait")
    if atype == "wait": return _fallback_action(aid, obs)
    return AeroSyncAction(agent_id=aid, action_type=atype, direction=raw.get("direction"), task_id=raw.get("task_id"), target_altitude=raw.get("target_altitude"))


def _fallback_action(did: str, obs: Dict[str, Any]) -> AeroSyncAction:
    drone, tasks = obs["drone_states"][did], obs["tasks"]
    dq = list(obs.get("dispatch_queue") or [])
    carry = drone.get("carrying_task_id")
    if carry:
        t = tasks[carry]
        tx, ty = t["delivery_location"]["x"], t["delivery_location"]["y"]
    else:
        # Work comes from the dispatch queue: tasks are created as DISPATCHED, not PENDING.
        candidates = []
        for tid in dq:
            t = tasks.get(tid)
            if not t or t.get("status") != "dispatched":
                continue
            dl = t.get("dispatch_location") or t["pickup_location"]
            dist = abs(dl["x"] - drone["position"]["x"]) + abs(dl["y"] - drone["position"]["y"])
            candidates.append((dist, -int(t.get("priority", 1)), tid, dl["x"], dl["y"]))
        if not candidates:
            return AeroSyncAction(agent_id=did, action_type="wait")
        candidates.sort()
        # If multiple equal-best candidates exist, pick one randomly to avoid identical runs.
        best_key = (candidates[0][0], candidates[0][1])
        best = [c for c in candidates if (c[0], c[1]) == best_key]
        _, _, _, tx, ty = random.choice(best)
    return _navigate_toward(did, drone, tx, ty, carry, tasks, dq, obs)

def _other_agent_at_cell(obs: Dict[str, Any], did: str, nx: int, ny: int, nz: int) -> bool:
    """True if another agent already occupies this grid cell (same z)."""
    for oid, d in obs.get("drone_states", {}).items():
        if oid == did:
            continue
        p = d["position"]
        if p["x"] == nx and p["y"] == ny and p["z"] == nz:
            return True
    for a in obs.get("agents", {}).values():
        p = a["position"]
        if p["x"] == nx and p["y"] == ny and p["z"] == nz:
            return True
    return False


def _plan_surface_move(
    obs: Dict[str, Any], did: str, x: int, y: int, z: int, tx: int, ty: int
) -> AeroSyncAction:
    """Manhattan routing with simple sidesteps when another agent blocks the preferred cell."""
    W, H = int(obs.get("grid_width", 100)), int(obs.get("grid_height", 100))

    def occ(nx: int, ny: int) -> bool:
        return _other_agent_at_cell(obs, did, nx, ny, z)

    # Prefer x/y, but randomize the axis order slightly to avoid identical traces across runs.
    prefer_x_first = random.random() < 0.5
    axes = ("x", "y") if prefer_x_first else ("y", "x")
    for axis in axes:
        if axis == "x" and x != tx:
            nx = x + (1 if tx > x else -1)
            if 0 <= nx < W and not occ(nx, y):
                return AeroSyncAction(agent_id=did, action_type="move", direction="east" if tx > x else "west")
        if axis == "y" and y != ty:
            ny = y + (1 if ty > y else -1)
            if 0 <= ny < H and not occ(x, ny):
                return AeroSyncAction(agent_id=did, action_type="move", direction="south" if ty > y else "north")

    # Sidestep if the preferred axis is blocked.
    if x != tx:
        for delta in (1, -1):
            ny = y + delta
            if 0 <= ny < H and not occ(x, ny):
                return AeroSyncAction(agent_id=did, action_type="move", direction="south" if delta > 0 else "north")
    if y != ty:
        for delta in (1, -1):
            nx = x + delta
            if 0 <= nx < W and not occ(nx, y):
                return AeroSyncAction(agent_id=did, action_type="move", direction="east" if delta > 0 else "west")

    return AeroSyncAction(agent_id=did, action_type="wait")

def _format_brief_state(obs_dict: Dict[str, Any]) -> str:
    parts: List[str] = []
    drones = obs_dict.get("drone_states", {})
    tasks = obs_dict.get("tasks", {})
    for did, d in drones.items():
        p = d.get("position", {})
        parts.append(
            f"{did}@({p.get('x')},{p.get('y')},z{p.get('z')}) "
            f"bat{d.get('battery', 0):.0f}% carry:{d.get('carrying_task_id')}"
        )
    task_bits = []
    for tid, t in tasks.items():
        task_bits.append(f"{tid}:{t.get('status')}")
    dq = obs_dict.get("dispatch_queue") or []
    return " | ".join(parts) + " || tasks=" + ",".join(task_bits) + f" || dq={list(dq)}"


def _navigate_toward(
    did: str,
    drone: Dict,
    tx: int,
    ty: int,
    carry: Optional[str],
    tasks: Dict,
    dispatch_queue: List[str],
    obs: Dict[str, Any],
) -> AeroSyncAction:
    x, y, z = drone["position"]["x"], drone["position"]["y"], drone["position"]["z"]
    flight = drone.get("flight") or {}
    prec = float(flight.get("hover_stability_score", 1.0))
    th = float(flight.get("delivery_precision_threshold", 0.5))

    if x == tx and y == ty:
        if z > 0:
            return AeroSyncAction(agent_id=did, action_type="descend", target_altitude=0)
        if carry:
            # If precision is poor, hover a step to stabilise before placing.
            if prec < th:
                return AeroSyncAction(agent_id=did, action_type="hover")
            return AeroSyncAction(agent_id=did, action_type="place")
        # Pick the dispatched task whose dispatch_location matches this cell.
        for tid in dispatch_queue:
            t = tasks.get(tid)
            if not t or t.get("status") != "dispatched":
                continue
            dl = t.get("dispatch_location") or t["pickup_location"]
            if dl["x"] == tx and dl["y"] == ty:
                return AeroSyncAction(agent_id=did, action_type="pick", task_id=tid)
        return AeroSyncAction(agent_id=did, action_type="wait")

    if z == 0:
        return AeroSyncAction(agent_id=did, action_type="ascend", target_altitude=1)
    return _plan_surface_move(obs, did, x, y, z, tx, ty)


def _run_heuristic_task(task_name: str, max_steps: int, *, sleep_ms: int, print_state_every: int, run_nonce: str) -> Dict[str, Any]:
    """Run a task using the deterministic heuristic agent (no LLM needed)."""
    score, success, steps, rewards = 0.0, False, 0, []
    log_start(task=task_name, env=BENCHMARK, model=f"heuristic({run_nonce})")
    env = None
    try:
        cfg = {"easy": easy_config, "medium": medium_config, "hard": hard_config}[task_name]()
        env = DroneEnv(cfg)
        obs = env.reset()
        obs_dict = obs.model_dump()
        for s in range(1, max_steps + 1):
            if obs_dict.get("done"): break
            steps = s
            # Command every drone every step using the heuristic
            for did in list(obs_dict["drone_states"].keys()):
                action = _fallback_action(did, obs_dict)
                obs, reward, done, _ = env.step(action)
                obs_dict = obs.model_dump()
                rewards.append(reward)
                log_step(step=s, action=action, reward=reward, done=done, error=None)
                if print_state_every > 0 and (s % print_state_every) == 0:
                    print(f"[STATE] step={s} { _format_brief_state(obs_dict) }", flush=True)
                if obs_dict.get("done"): break
                if sleep_ms > 0:
                    time.sleep(max(0.0, sleep_ms) / 1000.0)
            if obs_dict.get("done"): break
        score = grade(env.state())
        success = score >= 0.3
    except KeyboardInterrupt: pass
    except Exception as e: print(f"ERROR: {e}")
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)
        if env:
            try:
                report = detailed_report(env.state())
                print("\n" + "="*40)
                print("       DETAILED GRADING REPORT")
                print("="*40)
                for k, v in report.items():
                    if isinstance(v, float):
                        print(f"  {k:20} : {v:.4f}")
                    else:
                        print(f"  {k:20} : {v}")
                print("="*40 + "\n")
            except Exception as re:
                print(f"  [DEBUG] Failed to generate report: {re}")
    return {"score": score}


def run_task(
    client: Optional[OpenAI],
    task_name: str,
    max_steps: int,
    *,
    sleep_ms: int,
    print_state_every: int,
    run_nonce: str,
) -> Dict[str, Any]:
    # If no LLM client available, use the built-in heuristic agent
    if client is None:
        return _run_heuristic_task(
            task_name,
            max_steps,
            sleep_ms=sleep_ms,
            print_state_every=print_state_every,
            run_nonce=run_nonce,
        )

    score, success, steps, rewards = 0.0, False, 0, []
    log_start(task=task_name, env=BENCHMARK, model=f"{MODEL_NAME}({run_nonce})")
    env = None
    try:
        cfg = {"easy": easy_config, "medium": medium_config, "hard": hard_config}[task_name]()
        env = DroneEnv(cfg)
        obs = env.reset()
        obs_dict = obs.model_dump()
        conv = [{"role": "system", "content": build_system_prompt(task_name)}]
        for s in range(1, max_steps + 1):
            if obs_dict.get("done"): break
            steps = s
            # Nonce included so repeated runs are less likely to yield identical LLM traces.
            user_text = f"RUN_NONCE={run_nonce}\nSTEP {s}/{max_steps}\n"
            for did, d in obs_dict["drone_states"].items():
                p = d["position"]
                user_text += f"{did}: pos({p['x']},{p['y']},z{p['z']}) bat{d['battery']:.0f}% carrying:{d['carrying_task_id']}\n"
            for tid, t in obs_dict["tasks"].items():
                p, d = t["pickup_location"], t["delivery_location"]
                user_text += f"{tid}[{t['status']}]: pick({p['x']},{p['y']}) deli({d['x']},{d['y']})\n"
            conv.append({"role": "user", "content": user_text})
            if len(conv) > 12: conv = [conv[0]] + conv[-10:]
            raw = call_llm(client, conv)
            action = parse_action(raw, obs_dict)
            obs, reward, done, _ = env.step(action)
            obs_dict = obs.model_dump()
            rewards.append(reward)
            log_step(step=s, action=action, reward=reward, done=done, error=None)
            if print_state_every > 0 and (s % print_state_every) == 0:
                print(f"[STATE] step={s} { _format_brief_state(obs_dict) }", flush=True)
            if done: break
            if sleep_ms > 0:
                time.sleep(max(0.0, sleep_ms) / 1000.0)
        score = grade(env.state())
        success = score >= 0.5
    except KeyboardInterrupt: pass
    except Exception as e:
        print(f"ERROR: {e}")
        # Fall back to heuristic on any LLM error
        return _run_heuristic_task(
            task_name,
            max_steps,
            sleep_ms=sleep_ms,
            print_state_every=print_state_every,
            run_nonce=run_nonce,
        )
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)
        if env:
            try:
                report = detailed_report(env.state())
                print("\n" + "="*40)
                print("       DETAILED GRADING REPORT")
                print("="*40)
                for k, v in report.items():
                    if isinstance(v, float):
                        print(f"  {k:20} : {v:.4f}")
                    else:
                        print(f"  {k:20} : {v}")
                print("="*40 + "\n")
            except Exception as re:
                print(f"  [DEBUG] Failed to generate report: {re}")
    return {"score": score}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", help="Task to run (easy, medium, hard, or all)")
    p.add_argument("--max_steps", type=int, default=0, help="Override max steps (uses config defaults if 0)")
    p.add_argument("--sleep_ms", type=int, default=DEFAULT_SLEEP_MS, help="Delay between steps (ms) so output is readable")
    p.add_argument("--print_state_every", type=int, default=DEFAULT_PRINT_STATE_EVERY, help="Print positions/task status every N steps (0 disables)")
    args = p.parse_args()

    # Use LLM client if API key is available, otherwise fall back to heuristic agent
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, max_retries=1, timeout=DEFAULT_TIMEOUT_S)
        except Exception:
            client = None

    # No user-facing seeding; we just ensure per-run randomness.
    random.seed(time.time_ns())
    run_nonce = hex(random.getrandbits(64))[2:]
    print(f"[RUN] llm={'on' if client else 'off'} timeout_s={DEFAULT_TIMEOUT_S} sleep_ms={args.sleep_ms} print_state_every={args.print_state_every}", flush=True)

    tasks_to_run = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    config_fns = {"easy": easy_config, "medium": medium_config, "hard": hard_config}

    for t in tasks_to_run:
        steps = args.max_steps
        if steps == 0:
            # Use the task config default max_steps (so easy uses 120, etc.)
            cfg_fn = config_fns.get(t)
            if cfg_fn is not None:
                try:
                    steps = int(cfg_fn().get("max_steps", 120))
                except Exception:
                    steps = 120
            else:
                steps = 120
        run_task(
            client,
            t,
            steps,
            sleep_ms=args.sleep_ms,
            print_state_every=args.print_state_every,
            run_nonce=run_nonce,
        )

if __name__ == "__main__":
    main()
