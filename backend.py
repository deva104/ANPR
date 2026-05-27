import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

import database

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead: List[WebSocket] = []
        payload = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()


class VehicleCreate(BaseModel):
    plate_number: str
    owner_name: str
    vehicle_type: str
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    purpose: Optional[str] = None


class VehicleUpdate(BaseModel):
    owner_name: str
    vehicle_type: str
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    purpose: Optional[str] = None


class VerifyRequest(BaseModel):
    plate_number: str


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/vehicles")
async def vehicles_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "vehicles.html"))


@app.get("/api/vehicles")
async def api_get_vehicles():
    return database.get_all_vehicles()


@app.post("/api/vehicles")
async def api_add_vehicle(body: VehicleCreate):
    return database.add_vehicle(
        body.plate_number,
        body.owner_name,
        body.vehicle_type,
        body.valid_from,
        body.valid_until,
        body.purpose,
    )


@app.delete("/api/vehicles/{plate_number}")
async def api_delete_vehicle(plate_number: str):
    database.delete_vehicle(plate_number)
    return {"success": True}


@app.put("/api/vehicles/{plate_number}")
async def api_update_vehicle(plate_number: str, body: VehicleUpdate):
    return database.update_vehicle(
        plate_number,
        body.owner_name,
        body.vehicle_type,
        body.valid_from,
        body.valid_until,
        body.purpose,
    )


@app.get("/api/logs")
async def api_get_logs(limit: int = 50):
    return database.get_logs(limit)


@app.get("/api/logs/search/{plate_number}")
async def api_search_logs(plate_number: str):
    return database.search_logs(plate_number)


@app.get("/api/stats")
async def api_get_stats():
    return database.get_today_stats()


@app.post("/api/verify")
async def api_verify(body: VerifyRequest):
    result = database.verify_plate(body.plate_number)
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
    await manager.broadcast(
        {
            "type": "detection",
            "plate_number": result.get("plate_number"),
            "owner_name": result.get("owner_name"),
            "vehicle_type": result.get("vehicle_type"),
            "granted": result.get("granted"),
            "reason": result.get("reason"),
            "time": now_str,
        }
    )
    return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
