import asyncio
import json
import httpx
import websockets
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager

from env.models import AeroSyncAction, AeroSyncObservation, AgentType

class DroneEnv:
   
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws = None
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self._ws:
            await self._ws.close()
        await self._http.aclose()

    async def reset(self, task_name: str = "easy") -> AeroSyncObservation:
        response = await self._http.post("/reset", json={"task_name": task_name})
        response.raise_for_status()
        return AeroSyncObservation(**response.json())

    async def step(self, action: AeroSyncAction) -> tuple[AeroSyncObservation, float, bool, Dict[str, Any]]:
        response = await self._http.post("/step", json=action.model_dump())
        response.raise_for_status()
        data = response.json()
        return (
            AeroSyncObservation(**data["observation"]),
            data["reward"],
            data["done"],
            data["info"]
        )

    async def state(self) -> Dict[str, Any]:
        response = await self._http.get("/state")
        response.raise_for_status()
        return response.json()

    async def grade(self) -> Dict[str, Any]:
        response = await self._http.get("/grade")
        response.raise_for_status()
        return response.json()

    @asynccontextmanager
    async def session(self):
        async with websockets.connect(self.ws_url) as ws:
            self._ws = ws
            yield self
            self._ws = None

    async def ws_reset(self, task_name: str = "easy") -> AeroSyncObservation:
        if not self._ws:
            raise RuntimeError("Not in a WebSocket session. Use 'async with env.session():'")
        await self._ws.send(json.dumps({"type": "reset", "task_name": task_name}))
        resp = json.loads(await self._ws.recv())
        return AeroSyncObservation(**resp["observation"])

    async def ws_step(self, action: AeroSyncAction) -> tuple[AeroSyncObservation, float, bool, Dict[str, Any]]:
        if not self._ws:
            raise RuntimeError("Not in a WebSocket session. Use 'async with env.session():'")
        await self._ws.send(json.dumps({"type": "step", "action": action.model_dump()}))
        resp = json.loads(await self._ws.recv())
        return (
            AeroSyncObservation(**resp["observation"]),
            resp["reward"],
            resp["done"],
            resp["info"]
        )

    def sync(self):
        return AeroSyncSyncClient(self)


class AeroSyncSyncClient:
    def __init__(self, async_client: DroneEnv):
        self._async = async_client
        self._loop = asyncio.get_event_loop()

    def __enter__(self): return self
    def __exit__(self, *args): self._loop.run_until_complete(self._async.close())

    def reset(self, task_name: str = "easy"):
        return self._loop.run_until_complete(self._async.reset(task_name))

    def step(self, action: AeroSyncAction):
        return self._loop.run_until_complete(self._async.step(action))

    def state(self):
        return self._loop.run_until_complete(self._async.state())

    def grade(self):
        return self._loop.run_until_complete(self._async.grade())
