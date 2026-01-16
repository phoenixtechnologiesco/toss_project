# main.py
import cv2
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import os
from services import ShootingService
from typing import List

app = FastAPI(title="TOS - Target Observation System")

# Mount static files
app.mount("/static", StaticFiles(directory="shooting_sessions"), name="static")

# Global service instance
shooting_service = ShootingService()

# Global WebSocket connections for broadcasting
websocket_connections: List[WebSocket] = []

class StartSessionRequest(BaseModel):
    name: str
    service_no: str
    service_year: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard")
async def get_dashboard():
    return shooting_service.get_session_data()

@app.post("/start_session")
async def start_session(request: StartSessionRequest):
    result = shooting_service.start_session(
        request.name, 
        request.service_no, 
        request.service_year
    )
    
    # Broadcast session update to all connected clients
    session_info = {
        'current_session': shooting_service.current_session_index,
        'total_sessions': len(shooting_service.all_sessions),
        'active_hits': 0
    }
    
    # Send to all connected websockets
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_text(json.dumps({"type": "session_update", "data": session_info}))
        except:
            disconnected.append(ws)
    
    # Remove disconnected websockets
    for ws in disconnected:
        if ws in websocket_connections:
            websocket_connections.remove(ws)
    
    return result

@app.post("/stop_session")
async def stop_session():
    result = shooting_service.stop_session()
    
    # Broadcast session update to all connected clients
    session_info = {
        'current_session': shooting_service.current_session_index,
        'total_sessions': len(shooting_service.all_sessions),
        'active_hits': 0
    }
    
    # Send to all connected websockets
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_text(json.dumps({"type": "session_update", "data": session_info}))
        except:
            disconnected.append(ws)
    
    # Remove disconnected websockets
    for ws in disconnected:
        if ws in websocket_connections:
            websocket_connections.remove(ws)
    
    return result

@app.get("/report/{session_id}")
async def show_report(session_id: int):
    sessions = shooting_service.get_all_sessions()
    if session_id >= len(sessions):
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    filename = f"{session['service_no']}_{session['name']}_{session['service_year']}.jpg"
    image_path = os.path.join("shooting_sessions", filename)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Session Report</title></head>
    <body style="background:#1e2833; color:white; padding:20px;">
        <h2>🎯 Session Report</h2>
        <p><strong>Shooter:</strong> {session.get('name', 'N/A')} (SN: {session.get('service_no', 'N/A')}, SY: {session.get('service_year', 'N/A')})</p>
        <p><strong>Total Hits:</strong> {session.get('hits', 0)}</p>
        <p><strong>Group Size:</strong> {session.get('group_size', 0.0):.2f} inches</p>
        <p><strong>Start Time:</strong> {session.get('timestamp', 'N/A').replace('_', ' ')}</p>
        <hr>
        <h3>Target Image:</h3>
        {'<img src="/static/' + filename + '" style="max-width:100%; height:auto;" />' if os.path.exists(image_path) else '<p>No image available.</p>'}
        <br><br>
        <a href="/" style="color:#4CAF50;">← Back to Dashboard</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Add to global connections list
    websocket_connections.append(websocket)
    
    try:
        # Send initial session data
        session_info = {
            'current_session': shooting_service.current_session_index,
            'total_sessions': len(shooting_service.all_sessions),
            'active_hits': 0
        }
        await websocket.send_text(json.dumps({"type": "session_update", "data": session_info}))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()  # Wait for messages (we don't expect any)
    except WebSocketDisconnect:
        # Remove from connections list when disconnected
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# Background task for video streaming
async def video_streaming_task():
    """Background task to capture and stream video frames"""
    print("Starting video streaming task...")
    
    # Initialize camera first
    print("Initializing camera...")
    if shooting_service._grab_frame() is not None:
        print("✅ Camera initialized successfully!")
        await asyncio.sleep(2)
    else:
        print("❌ Camera initialization failed!")
        return
    
    while True:
# In video_streaming_task(), replace the frame processing part with:

        try:
            # Only process frames if there's an active session
            if shooting_service.current_session_index != -1:
                shooting_service.process_frame()  # Process frame for detection
            
            frame = shooting_service.get_display_frame()
            if frame is not None and frame.size > 0:  # Check if frame is valid
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Convert to base64 for WebSocket transmission
                    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                    
                    # Broadcast frame to all connected clients
                    disconnected = []
                    for ws in websocket_connections:
                        try:
                            await ws.send_text(json.dumps({"type": "video_frame", "data": frame_base64}))
                        except:
                            disconnected.append(ws)
                    
                    # Remove disconnected websockets
                    for ws in disconnected:
                        if ws in websocket_connections:
                            websocket_connections.remove(ws)
                    
                    print(f"✅ Frame sent to {len(websocket_connections)} clients ({len(frame_base64)} bytes)")
                else:
                    print("❌ Failed to encode frame")
            else:
                print("⚠️ No frame available")
            
            await asyncio.sleep(0.067)  # ~15 FPS
            
        except Exception as e:
            print(f"❌ Video streaming error: {e}")
            await asyncio.sleep(1)  # Wait 1 second before retrying        

# Start video streaming task
import threading
def start_video_streaming():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(video_streaming_task())

# Start the video streaming thread when the app starts
video_thread = threading.Thread(target=start_video_streaming, daemon=True)
video_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
