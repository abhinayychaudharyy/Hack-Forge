from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from env.aerosync_env import AeroSyncEnv
from env.models import (
    AeroSyncAction, AeroSyncObservation,
    AgentType, EpisodeInfo, Position,
)
from grader.grader import grade, detailed_report, GRADE_PARAMS
from tasks.easy   import get_config as easy_config
from tasks.medium import get_config as medium_config
from tasks.hard   import get_config as hard_config

_VERSION = "1.2.0"
_ENV_NAME = "aerosync-ai"

TASK_CONFIGS: Dict[str, Any] = {
    "easy":   easy_config,
    "medium": medium_config,
    "hard":   hard_config,
}

_TASK_METADATA: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build task metadata cache at startup."""
    for task_name, config_fn in TASK_CONFIGS.items():
        cfg = config_fn()
        params = GRADE_PARAMS.get(task_name, GRADE_PARAMS["easy"])
        _TASK_METADATA[task_name] = {
            "name":        task_name,
            "description": _task_description(task_name, cfg),
            "difficulty":  {"easy": 1, "medium": 2, "hard": 3}[task_name],
            "agents": {
                "robots": len(cfg.get("robots", [])),
                "drones": len(cfg.get("drones", [])),
            },
            "task_count":  len(cfg.get("tasks", [])),
            "max_steps":   cfg.get("max_steps", 0),
            "grid":        {"width": cfg["grid_width"], "height": cfg["grid_height"]},
            "dispatch_zones": len(cfg.get("dispatch_zones", [])),
            "grading_weights": {
                "completion":    params["completion_weight"],
                "efficiency":    params["efficiency_weight"],
                "safety":        params["safety_weight"],
                "priority":      params["priority_weight"],
                "drone_quality": params["drone_weight"],
            },
        }
    yield

def _task_description(name: str, cfg: Dict[str, Any]) -> str:
    robots = len(cfg.get("robots", []))
    drones = len(cfg.get("drones", []))
    tasks  = len(cfg.get("tasks",  []))
    w, h   = cfg["grid_width"], cfg["grid_height"]
    obs    = len(cfg.get("obstacles", []))
    descriptions = {
        "easy":   f"{robots} robot + {drones} drone, {tasks} deliveries. "
                  f"Open {w}×{h} grid, no battery pressure.",
        "medium": f"{robots} robots + {drones} drones, {tasks} deliveries. "
                  f"Battery management, dual dispatch zones, {obs} obstacles.",
        "hard":   f"{robots} robots + {drones} drones, {tasks} deliveries "
                  f"(3 urgent priority-3 tasks). "
                  f"{w}×{h} grid with {obs} obstacles, congestion, mixed priorities.",
    }
    return descriptions.get(name, f"{name} task")

app = FastAPI(
    title="AeroSync AI",
    description=(
        "OpenEnv-compliant logistics simulation with warehouse robots and "
        "autonomous delivery drones. Drones feature advanced flight physics: "
        "altitude layers, hover stability, battery drain, obstacle avoidance, "
        "TTC collision risk, and precision landing."
    ),
    version=_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[AeroSyncEnv] = None


def _get_env() -> AeroSyncEnv:
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first."
        )
    return _env

class ResetRequest(BaseModel):
    task_name: str = Field("easy", description="Task difficulty: easy | medium | hard")


class StepResponse(BaseModel):
    observation: AeroSyncObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any]  


class GradeResponse(BaseModel):
    score:  float = Field(..., ge=0.0, le=1.0, description="Final score in [0.0, 1.0]")
    report: Dict[str, Any]


class BFSRequest(BaseModel):
    start:      Dict[str, int] = Field(..., description="Start position: {x, y, z}")
    goal:       Dict[str, int] = Field(..., description="Goal position: {x, y, z}")
    agent_type: str            = Field("robot", description="'robot' or 'drone'")


class BFSResponse(BaseModel):
    path:  List[str] = Field(..., description="Ordered list of directions: north/south/east/west")
    steps: int       = Field(..., description="Path length in steps")


class MetricsResponse(BaseModel):
    step:            int
    max_steps:       int
    task_name:       str
    completion_rate: float
    tasks_delivered: int
    tasks_total:     int
    collisions:      int
    battery_failures: int
    current_score:   float

@app.get("/")
def root():
    """Redirect to the dashboard."""
    return RedirectResponse(url="/web")

@app.get("/health")
def health():
   
    env_status = "uninitialised"
    current_task = None
    if _env is not None:
        s = _env.state()
        env_status = "running" if not all(
            t.get("status") in ("delivered", "failed")
            for t in s.get("tasks", {}).values()
        ) else "episode_ended"
        current_task = s.get("task_name")

    return {
        "status":       "ok",
        "env":          _ENV_NAME,
        "version":      _VERSION,
        "env_status":   env_status,
        "current_task": current_task,
    }


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata (weights pulled from grader.py)."""
    if _TASK_METADATA:
        return {"tasks": list(_TASK_METADATA.values())}
    tasks_out = []
    for name, config_fn in TASK_CONFIGS.items():
        cfg    = config_fn()
        params = GRADE_PARAMS.get(name, GRADE_PARAMS["easy"])
        tasks_out.append({
            "name":        name,
            "description": _task_description(name, cfg),
            "difficulty":  {"easy": 1, "medium": 2, "hard": 3}[name],
            "agents": {
                "robots": len(cfg.get("robots", [])),
                "drones": len(cfg.get("drones", [])),
            },
            "task_count":  len(cfg.get("tasks", [])),
            "max_steps":   cfg.get("max_steps", 0),
            "grid":        {"width": cfg["grid_width"], "height": cfg["grid_height"]},
            "dispatch_zones": len(cfg.get("dispatch_zones", [])),
            "grading_weights": {
                "completion":    params["completion_weight"],
                "efficiency":    params["efficiency_weight"],
                "safety":        params["safety_weight"],
                "priority":      params["priority_weight"],
                "drone_quality": params["drone_weight"],
            },
        })
    return {"tasks": tasks_out}


@app.post("/reset", response_model=AeroSyncObservation)
def reset(request: Optional[ResetRequest] = None):
    global _env
    if request is None:
        request = ResetRequest()
    task_name = request.task_name.lower().strip()
    if task_name not in TASK_CONFIGS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{task_name}'. Choose from: {list(TASK_CONFIGS.keys())}"
        )
    config = TASK_CONFIGS[task_name]()
    _env  = AeroSyncEnv(config)
    obs   = _env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step(action: AeroSyncAction):
    env = _get_env()
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info.model_dump(),
    )


@app.get("/state")
def state():
    env = _get_env()
    return env.state()


@app.get("/grade", response_model=GradeResponse)
def get_grade():
    env = _get_env()
    s   = env.state()
    return GradeResponse(
        score=grade(s),
        report=detailed_report(s),
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    env = _get_env()
    s   = env.state()
    tasks = s.get("tasks", {})
    delivered = sum(1 for t in tasks.values() if t.get("status") == "delivered")
    total     = len(tasks)
    return MetricsResponse(
        step=s.get("step", 0),
        max_steps=s.get("max_steps", 0),
        task_name=s.get("task_name", ""),
        completion_rate=round(delivered / total, 4) if total else 0.0,
        tasks_delivered=delivered,
        tasks_total=total,
        collisions=s.get("collision_count", 0),
        battery_failures=s.get("battery_failures", 0),
        current_score=grade(s),
    )


@app.post("/bfs", response_model=BFSResponse)
def bfs_path(request: BFSRequest):

    env = _get_env()
    try:
        atype = AgentType.ROBOT if request.agent_type.lower() == "robot" else AgentType.DRONE
        start = Position(**request.start)
        goal  = Position(**request.goal)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid positions: {e}")

    path = env.bfs_path(start, goal)
    return BFSResponse(path=path, steps=len(path))


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def get_openenv_yaml():
    """Serve the openenv.yaml spec file."""
    yaml_path = Path(__file__).parent.parent / "openenv.yaml"
    if yaml_path.exists():
        return yaml_path.read_text(encoding="utf-8")
    # Fallback to root or current dir
    root_yaml = Path("openenv.yaml")
    if root_yaml.exists():
        return root_yaml.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ── Advanced OpenEnv Features ────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session for OpenEnv agents.
    Handles a full mission lifecycle (reset -> steps -> done).
    """
    await websocket.accept()
    local_env: Optional[AeroSyncEnv] = None
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # 1. Handle Reset
            if data.get("type") == "reset":
                task_name = data.get("task_name", "easy")
                config = TASK_CONFIGS.get(task_name, easy_config)()
                local_env = AeroSyncEnv(config)
                obs = local_env.reset()
                await websocket.send_json({
                    "type": "observation",
                    "observation": obs.model_dump()
                })
                
            # 2. Handle Step
            elif data.get("type") == "step":
                if local_env is None:
                    await websocket.send_json({"type": "error", "message": "Call reset first"})
                    continue
                
                action_data = data.get("action")
                if not action_data:
                    await websocket.send_json({"type": "error", "message": "Missing action"})
                    continue
                    
                action = AeroSyncAction(**action_data)
                obs, reward, done, info = local_env.step(action)
                
                await websocket.send_json({
                    "type": "result",
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info.model_dump()
                })
                
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {data.get('type')}"})
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


@app.get("/web", response_class=HTMLResponse)
def dashboard():
    """Enhanced dashboard with drone telemetry and mission progress."""
    env_summary = "Environment Uninitialized"
    metrics_block = ""
    
    if _env is not None:
        s = _env.state()
        task_name = s.get("task_name", "unknown")
        step = s.get("step", 0)
        max_steps = s.get("max_steps", 0)
        delivered = sum(1 for t in s.get("tasks", {}).values() if t.get("status") == "delivered")
        total = len(s.get("tasks", {}))
        
        env_summary = f"Mission: {task_name.upper()} | Step: {step}/{max_steps}"
        metrics_block = f"""
            <div class="stat">Deliveries: {delivered}/{total}</div>
            <div class="stat">Collisions: {s.get('collision_count', 0)}</div>
            <div class="stat">Battery Fails: {s.get('battery_failures', 0)}</div>
            <div class="stat">Score: {grade(s):.4f}</div>
        """

    return f"""
    <html>
        <head>
            <title>AeroSync AI Dashboard</title>
            <style>
                body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: white; padding: 2rem; }}
                .card {{ background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155; max-width: 600px; margin: auto; }}
                h1 {{ color: #38bdf8; margin-top: 0; }}
                .summary {{ font-size: 1.2rem; margin-bottom: 1rem; color: #94a3b8; }}
                .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
                .stat {{ background: #0f172a; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #1e293b; }}
                .badge {{ background: #0ea5e9; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }}
            </style>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <div class="card">
                <h1>🚁 AeroSync AI <span class="badge">v{_VERSION}</span></h1>
                <div class="summary">{env_summary}</div>
                <div class="metrics">
                    {metrics_block}
                </div>
                <hr style="border: 0; border-top: 1px solid #334155; margin: 1.5rem 0;">
                <div style="font-size: 0.9rem; color: #64748b;">
                    API Live | WebSocket active on /ws | Auto-refreshing every 5s
                </div>
            </div>
        </body>
    </html>
    """

def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
