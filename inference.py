from __future__ import annotations
import argparse
import json
import os
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
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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
            resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.4, max_tokens=1024, timeout=10.0)
            c = resp.choices[0].message.content.strip()
            j = _clean_content(c)
            if j: return json.loads(j)
        except: time.sleep(1)
    return None


def _pick_best_drone(obs: Dict[str, Any]) -> str:
    drones, tasks = obs.get("drone_states", {}), obs.get("tasks", {})
    for did, d in drones.items():
        if d.get("carrying_task_id"): return did
    pending = [t for t in tasks.values() if t.get("status") == "pending" and t.get("assigned_drone") is None]
    if pending:
        pending.sort(key=lambda t: -t.get("priority", 1))
        t = pending[0]
        tx, ty = t["pickup_location"]["x"], t["pickup_location"]["y"]
        best_did, min_dist = None, 1e9
        for did, d in drones.items():
            if d["battery"] < 15: continue
            dist = abs(d["position"]["x"]-tx) + abs(d["position"]["y"]-ty)
            if dist < min_dist: min_dist, best_did = dist, did
        if best_did: return best_did
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
    carry = drone.get("carrying_task_id")
    if carry:
        t = tasks[carry]
        tx, ty = t["delivery_location"]["x"], t["delivery_location"]["y"]
    else:
        pending = [t for t in tasks.values() if t.get("status") == "pending"]
        if not pending: return AeroSyncAction(agent_id=did, action_type="hover")
        pending.sort(key=lambda t: abs(t["pickup_location"]["x"]-drone["position"]["x"]) + abs(t["pickup_location"]["y"]-drone["position"]["y"]))
        t = pending[0]
        tx, ty = t["pickup_location"]["x"], t["pickup_location"]["y"]
    return _navigate_toward(did, drone, tx, ty, carry, tasks)


def _navigate_toward(did: str, drone: Dict, tx: int, ty: int, carry: Optional[str], tasks: Dict) -> AeroSyncAction:
    x, y, z = drone["position"]["x"], drone["position"]["y"], drone["position"]["z"]
    if x == tx and y == ty:
        if z > 0: return AeroSyncAction(agent_id=did, action_type="descend", target_altitude=0)
        if carry: return AeroSyncAction(agent_id=did, action_type="place")
        for tid, t in tasks.items():
            if t.get("status") == "pending" and t["pickup_location"]["x"] == tx and t["pickup_location"]["y"] == ty:
                return AeroSyncAction(agent_id=did, action_type="pick", task_id=tid)
        return AeroSyncAction(agent_id=did, action_type="hover")
    if z == 0: return AeroSyncAction(agent_id=did, action_type="ascend", target_altitude=1)
    if x != tx: return AeroSyncAction(agent_id=did, action_type="move", direction="east" if tx > x else "west")
    return AeroSyncAction(agent_id=did, action_type="move", direction="south" if ty > y else "north")


def _run_heuristic_task(task_name: str, max_steps: int) -> Dict[str, Any]:
    """Run a task using the deterministic heuristic agent (no LLM needed)."""
    score, success, steps, rewards = 0.0, False, 0, []
    log_start(task=task_name, env=BENCHMARK, model="heuristic")
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
                if obs_dict.get("done"): break
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


def run_task(client: Optional[OpenAI], task_name: str, max_steps: int) -> Dict[str, Any]:
    # If no LLM client available, use the built-in heuristic agent
    if client is None:
        return _run_heuristic_task(task_name, max_steps)

    score, success, steps, rewards = 0.0, False, 0, []
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
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
            user_text = f"STEP {s}/{max_steps}\n"
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
            if done: break
            time.sleep(0.1)
        score = grade(env.state())
        success = score >= 0.5
    except KeyboardInterrupt: pass
    except Exception as e:
        print(f"ERROR: {e}")
        # Fall back to heuristic on any LLM error
        return _run_heuristic_task(task_name, max_steps)
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
    args = p.parse_args()

    # Use LLM client if API key is available, otherwise fall back to heuristic agent
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=10.0, max_retries=1)
        except Exception:
            client = None

    tasks_to_run = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    for t in tasks_to_run:
        steps = args.max_steps
        if steps == 0:
            if t == "easy": steps = 120
            elif t == "medium": steps = 250
            elif t == "hard": steps = 500
            else: steps = 120
        run_task(client, t, steps)

if __name__ == "__main__":
    main()
