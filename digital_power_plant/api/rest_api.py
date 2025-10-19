"""
REST API - RESTful endpoints for the digital power plant
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class PowerPlantStatusResponse(BaseModel):
    status: str
    units: List[Dict]
    metrics: Optional[Dict]

class SafetyEventsResponse(BaseModel):
    active_events: List[Dict]
    event_history: List[Dict]
    safety_level: str

class PhysicsDataResponse(BaseModel):
    turbines: Dict[str, Dict]
    power_grid: Dict[str, Dict]
    simulation_time: float

class AIStatusResponse(BaseModel):
    status: str
    network_parameters: int
    control_actions_today: int
    average_performance: float
    learning_rate: float

class CommandRequest(BaseModel):
    command: str
    parameters: Dict

class CommandResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

# Create router
router = APIRouter()

# Global references to core systems (will be injected by main.py)
power_plant_manager = None
ai_controller = None
physics_simulator = None
safety_system = None

def set_core_systems(ppm, ai, physics, safety):
    """Set references to core systems"""
    global power_plant_manager, ai_controller, physics_simulator, safety_system
    power_plant_manager = ppm
    ai_controller = ai
    physics_simulator = physics
    safety_system = safety

@router.get("/status", response_model=PowerPlantStatusResponse)
async def get_power_plant_status():
    """Get current power plant status"""
    if not power_plant_manager:
        raise HTTPException(status_code=503, detail="Power plant manager not available")
    
    status = power_plant_manager.get_status()
    return PowerPlantStatusResponse(**status)

@router.get("/metrics", response_model=List[Dict])
async def get_metrics_history(limit: int = Query(100, ge=1, le=1000)):
    """Get power plant metrics history"""
    if not power_plant_manager:
        raise HTTPException(status_code=503, detail="Power plant manager not available")
    
    return power_plant_manager.get_metrics_history(limit)

@router.post("/units/{unit_id}/start")
async def start_unit(unit_id: str):
    """Start a power generation unit"""
    if not power_plant_manager:
        raise HTTPException(status_code=503, detail="Power plant manager not available")
    
    success = power_plant_manager.start_unit(unit_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to start unit {unit_id}")
    
    return {"message": f"Unit {unit_id} started successfully"}

@router.post("/units/{unit_id}/stop")
async def stop_unit(unit_id: str):
    """Stop a power generation unit"""
    if not power_plant_manager:
        raise HTTPException(status_code=503, detail="Power plant manager not available")
    
    success = power_plant_manager.stop_unit(unit_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to stop unit {unit_id}")
    
    return {"message": f"Unit {unit_id} stopped successfully"}

@router.get("/safety", response_model=SafetyEventsResponse)
async def get_safety_status():
    """Get safety system status and events"""
    if not safety_system:
        raise HTTPException(status_code=503, detail="Safety system not available")
    
    status = safety_system.get_status()
    active_events = safety_system.get_active_events()
    event_history = safety_system.get_event_history()
    
    return SafetyEventsResponse(
        active_events=active_events,
        event_history=event_history,
        safety_level=status["current_safety_level"]
    )

@router.get("/safety/events", response_model=List[Dict])
async def get_safety_events(limit: int = Query(50, ge=1, le=500)):
    """Get safety event history"""
    if not safety_system:
        raise HTTPException(status_code=503, detail="Safety system not available")
    
    return safety_system.get_event_history(limit)

@router.post("/safety/emergency-shutdown")
async def emergency_shutdown():
    """Trigger emergency shutdown"""
    if not safety_system:
        raise HTTPException(status_code=503, detail="Safety system not available")
    
    safety_system.manual_emergency_shutdown()
    return {"message": "Emergency shutdown initiated"}

@router.post("/safety/reset")
async def reset_safety_system():
    """Reset safety system"""
    if not safety_system:
        raise HTTPException(status_code=503, detail="Safety system not available")
    
    safety_system.reset_emergency_shutdown()
    return {"message": "Safety system reset"}

@router.get("/physics", response_model=PhysicsDataResponse)
async def get_physics_data():
    """Get physics simulation data"""
    if not physics_simulator:
        raise HTTPException(status_code=503, detail="Physics simulator not available")
    
    status = physics_simulator.get_status()
    turbines = physics_simulator.get_all_turbines_status()
    power_grid = physics_simulator.get_power_grid_status()
    
    return PhysicsDataResponse(
        turbines=turbines,
        power_grid=power_grid,
        simulation_time=status["simulation_time"]
    )

@router.get("/physics/turbines/{turbine_id}")
async def get_turbine_status(turbine_id: str):
    """Get specific turbine status"""
    if not physics_simulator:
        raise HTTPException(status_code=503, detail="Physics simulator not available")
    
    turbine_status = physics_simulator.get_turbine_status(turbine_id)
    if not turbine_status:
        raise HTTPException(status_code=404, detail=f"Turbine {turbine_id} not found")
    
    return turbine_status

@router.post("/physics/turbines/{turbine_id}/adjust")
async def adjust_turbine_parameters(turbine_id: str, parameters: Dict):
    """Adjust turbine simulation parameters"""
    if not physics_simulator:
        raise HTTPException(status_code=503, detail="Physics simulator not available")
    
    success = physics_simulator.adjust_turbine_parameters(turbine_id, **parameters)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to adjust turbine {turbine_id}")
    
    return {"message": f"Turbine {turbine_id} parameters adjusted successfully"}

@router.get("/ai", response_model=AIStatusResponse)
async def get_ai_status():
    """Get AI controller status"""
    if not ai_controller:
        raise HTTPException(status_code=503, detail="AI controller not available")
    
    status = ai_controller.get_status()
    return AIStatusResponse(**status)

@router.get("/ai/history", response_model=List[Dict])
async def get_ai_control_history(limit: int = Query(50, ge=1, le=500)):
    """Get AI control action history"""
    if not ai_controller:
        raise HTTPException(status_code=503, detail="AI controller not available")
    
    return ai_controller.get_control_history(limit)

@router.post("/ai/learning-rate")
async def adjust_learning_rate(learning_rate: float = Query(..., ge=0.0001, le=1.0)):
    """Adjust AI learning rate"""
    if not ai_controller:
        raise HTTPException(status_code=503, detail="AI controller not available")
    
    ai_controller.adjust_learning_rate(learning_rate)
    return {"message": f"Learning rate adjusted to {learning_rate}"}

@router.post("/ai/reset")
async def reset_ai_network():
    """Reset AI neural network"""
    if not ai_controller:
        raise HTTPException(status_code=503, detail="AI controller not available")
    
    ai_controller.reset_network()
    return {"message": "AI neural network reset"}

@router.post("/command", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """Execute a command"""
    # This would normally be handled by the WebSocket manager
    # For REST API, we'll implement basic command handling
    
    command = request.command
    parameters = request.parameters
    
    if command == "start_unit":
        unit_id = parameters.get("unit_id")
        if not unit_id:
            return CommandResponse(success=False, message="Missing unit_id parameter")
        
        if power_plant_manager:
            success = power_plant_manager.start_unit(unit_id)
            if success:
                return CommandResponse(success=True, message=f"Unit {unit_id} started")
            else:
                return CommandResponse(success=False, message=f"Failed to start unit {unit_id}")
        else:
            return CommandResponse(success=False, message="Power plant manager not available")
    
    elif command == "stop_unit":
        unit_id = parameters.get("unit_id")
        if not unit_id:
            return CommandResponse(success=False, message="Missing unit_id parameter")
        
        if power_plant_manager:
            success = power_plant_manager.stop_unit(unit_id)
            if success:
                return CommandResponse(success=True, message=f"Unit {unit_id} stopped")
            else:
                return CommandResponse(success=False, message=f"Failed to stop unit {unit_id}")
        else:
            return CommandResponse(success=False, message="Power plant manager not available")
    
    elif command == "emergency_shutdown":
        if safety_system:
            safety_system.manual_emergency_shutdown()
            return CommandResponse(success=True, message="Emergency shutdown initiated")
        else:
            return CommandResponse(success=False, message="Safety system not available")
    
    else:
        return CommandResponse(success=False, message=f"Unknown command: {command}")

@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "components": {}
    }
    
    # Check power plant manager
    if power_plant_manager:
        ppm_status = power_plant_manager.get_status()
        health_status["components"]["power_plant_manager"] = {
            "status": "healthy",
            "plant_status": ppm_status["status"]
        }
    else:
        health_status["components"]["power_plant_manager"] = {"status": "unavailable"}
        health_status["overall_status"] = "degraded"
    
    # Check AI controller
    if ai_controller:
        ai_status = ai_controller.get_status()
        health_status["components"]["ai_controller"] = {
            "status": "healthy",
            "ai_status": ai_status["status"]
        }
    else:
        health_status["components"]["ai_controller"] = {"status": "unavailable"}
        health_status["overall_status"] = "degraded"
    
    # Check physics simulator
    if physics_simulator:
        physics_status = physics_simulator.get_status()
        health_status["components"]["physics_simulator"] = {
            "status": "healthy",
            "simulation_time": physics_status["simulation_time"]
        }
    else:
        health_status["components"]["physics_simulator"] = {"status": "unavailable"}
        health_status["overall_status"] = "degraded"
    
    # Check safety system
    if safety_system:
        safety_status = safety_system.get_status()
        health_status["components"]["safety_system"] = {
            "status": "healthy",
            "safety_level": safety_status["current_safety_level"]
        }
    else:
        health_status["components"]["safety_system"] = {"status": "unavailable"}
        health_status["overall_status"] = "degraded"
    
    return health_status