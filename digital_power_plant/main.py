#!/usr/bin/env python3
"""
Digital Power Plant - Main Application Entry Point
World's First Digital Power Plant System
"""

import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from contextlib import asynccontextmanager

from core.power_plant_manager import PowerPlantManager
from core.ai_controller import AIController
from core.physics_simulator import PhysicsSimulator
from core.safety_system import SafetySystem
from api.websocket_handler import WebSocketManager
from api.rest_api import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
power_plant_manager = None
ai_controller = None
physics_simulator = None
safety_system = None
websocket_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global power_plant_manager, ai_controller, physics_simulator, safety_system, websocket_manager
    
    logger.info("🚀 Starting Digital Power Plant System...")
    
    # Initialize core systems
    power_plant_manager = PowerPlantManager()
    ai_controller = AIController()
    physics_simulator = PhysicsSimulator()
    safety_system = SafetySystem()
    websocket_manager = WebSocketManager()
    
    # Start background tasks
    asyncio.create_task(power_plant_manager.start())
    asyncio.create_task(ai_controller.start())
    asyncio.create_task(physics_simulator.start())
    asyncio.create_task(safety_system.start())
    
    logger.info("✅ Digital Power Plant System Started Successfully!")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Digital Power Plant System...")
    await power_plant_manager.stop()
    await ai_controller.stop()
    await physics_simulator.stop()
    await safety_system.stop()
    logger.info("✅ Shutdown Complete")

# Create FastAPI application
app = FastAPI(
    title="Digital Power Plant",
    description="World's First Digital Power Plant System",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Inject core systems into API
@app.on_event("startup")
async def inject_core_systems():
    from api.rest_api import set_core_systems
    set_core_systems(power_plant_manager, ai_controller, physics_simulator, safety_system)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main dashboard"""
    return HTMLResponse(open("static/dashboard.html").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "power_plant": power_plant_manager.get_status() if power_plant_manager else "not_initialized",
        "ai_controller": ai_controller.get_status() if ai_controller else "not_initialized",
        "physics_simulator": physics_simulator.get_status() if physics_simulator else "not_initialized",
        "safety_system": safety_system.get_status() if safety_system else "not_initialized"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )