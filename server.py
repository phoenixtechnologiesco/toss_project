import json
import time
import cv2
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
from typing import List

app = FastAPI()

# YOLO Model load
model = YOLO(".best.pt")

# Camera stream
CAMERA_STREAM = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# Global session variables
clients: List[WebSocket] = []
session_running = False
bullet_count = 0
fire_distances = []
results_data = []

ACTUAL_BULLET_WIDTH = 0.009
FOCAL_LENGTH = 800


# CLIENT CONNECTION HANDLING
async def broadcast(message: dict):
    dead_clients = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except:
            dead_clients.append(ws)

    for dc in dead_clients:
        clients.remove(dc)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)

    await websocket.send_json({"status": "connected"})

    try:
        while True:
            await websocket.receive_text()  # client messages if needed
    except WebSocketDisconnect:
        clients.remove(websocket)


#        START SESSION
@app.get("/start_session")
async def start_session():
    global session_running, bullet_count, fire_distances, results_data
    session_running = True
    bullet_count = 0
    fire_distances = []
    results_data = []

    asyncio.create_task(run_detection())

    await broadcast({"event": "session_started"})
    return {"status": "session started"}


#       DETECTION LOOP
async def run_detection():
    global session_running, bullet_count

    cap = cv2.VideoCapture(CAMERA_STREAM)

    while session_running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, conf=0.3, iou=0.45, verbose=False)

        for box in results[0].boxes:
            # box coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            bullet_count += 1
            box_width = x2 - x1
            distance = (ACTUAL_BULLET_WIDTH * FOCAL_LENGTH) / box_width if box_width > 0 else 0

            fire_distances.append(distance)
            results_data.append({
                "time": time.time(),
                "confidence": conf,
                "distance": distance
            })

            # Send live update to ALL tabs
            await broadcast({
                "event": "fire_detected",
                "bullet_count": bullet_count,
                "last_distance": distance
            })

        await asyncio.sleep(0.05)

    cap.release()


#        END SESSION
@app.get("/stop_session")
async def stop_session():
    global session_running

    session_running = False

    if not fire_distances:
        return {"error": "no fires detected"}

    max_dist = max(fire_distances)
    fire_no = fire_distances.index(max_dist) + 1

    result = {
        "total_bullets": bullet_count,
        "longest_fire_number": fire_no,
        "longest_fire_distance": max_dist,
        "detections": results_data
    }

    # send final result to every tab
    await broadcast({
        "event": "session_ended",
        "result": result
    })

    return {"status": "session ended", "result": result}
