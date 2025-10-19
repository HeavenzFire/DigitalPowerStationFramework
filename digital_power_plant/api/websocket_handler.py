"""
WebSocket Handler - Real-time communication for the digital power plant
"""

import asyncio
import json
import logging
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, Dict] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_data[websocket] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "subscriptions": set()
        }
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.connection_data:
                del self.connection_data[websocket]
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
            
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            # Update last activity
            if websocket in self.connection_data:
                self.connection_data[websocket]["last_activity"] = datetime.now()
                
            if message_type == "subscribe":
                await self._handle_subscription(websocket, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(websocket, data)
            elif message_type == "command":
                await self._handle_command(websocket, data)
            else:
                await self._send_error(websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, f"Error processing message: {str(e)}")
            
    async def _handle_subscription(self, websocket: WebSocket, data: Dict):
        """Handle subscription request"""
        subscription_type = data.get("subscription_type")
        if not subscription_type:
            await self._send_error(websocket, "Missing subscription_type")
            return
            
        if websocket in self.connection_data:
            self.connection_data[websocket]["subscriptions"].add(subscription_type)
            
        await self._send_success(websocket, f"Subscribed to {subscription_type}")
        logger.info(f"WebSocket subscribed to {subscription_type}")
        
    async def _handle_unsubscription(self, websocket: WebSocket, data: Dict):
        """Handle unsubscription request"""
        subscription_type = data.get("subscription_type")
        if not subscription_type:
            await self._send_error(websocket, "Missing subscription_type")
            return
            
        if websocket in self.connection_data:
            self.connection_data[websocket]["subscriptions"].discard(subscription_type)
            
        await self._send_success(websocket, f"Unsubscribed from {subscription_type}")
        logger.info(f"WebSocket unsubscribed from {subscription_type}")
        
    async def _handle_command(self, websocket: WebSocket, data: Dict):
        """Handle command request"""
        command = data.get("command")
        parameters = data.get("parameters", {})
        
        if not command:
            await self._send_error(websocket, "Missing command")
            return
            
        # Process command (in real implementation, would interface with core systems)
        result = await self._process_command(command, parameters)
        
        if result["success"]:
            await self._send_success(websocket, f"Command {command} executed successfully", result)
        else:
            await self._send_error(websocket, f"Command {command} failed: {result['error']}")
            
    async def _process_command(self, command: str, parameters: Dict) -> Dict:
        """Process a command (placeholder implementation)"""
        # In real implementation, would interface with PowerPlantManager, AIController, etc.
        command_handlers = {
            "start_unit": self._handle_start_unit,
            "stop_unit": self._handle_stop_unit,
            "adjust_power": self._handle_adjust_power,
            "emergency_shutdown": self._handle_emergency_shutdown,
            "reset_safety": self._handle_reset_safety
        }
        
        handler = command_handlers.get(command)
        if handler:
            return await handler(parameters)
        else:
            return {"success": False, "error": f"Unknown command: {command}"}
            
    async def _handle_start_unit(self, parameters: Dict) -> Dict:
        """Handle start unit command"""
        unit_id = parameters.get("unit_id")
        if not unit_id:
            return {"success": False, "error": "Missing unit_id"}
            
        # In real implementation, would call PowerPlantManager.start_unit()
        logger.info(f"Starting unit {unit_id}")
        return {"success": True, "message": f"Unit {unit_id} started"}
        
    async def _handle_stop_unit(self, parameters: Dict) -> Dict:
        """Handle stop unit command"""
        unit_id = parameters.get("unit_id")
        if not unit_id:
            return {"success": False, "error": "Missing unit_id"}
            
        # In real implementation, would call PowerPlantManager.stop_unit()
        logger.info(f"Stopping unit {unit_id}")
        return {"success": True, "message": f"Unit {unit_id} stopped"}
        
    async def _handle_adjust_power(self, parameters: Dict) -> Dict:
        """Handle adjust power command"""
        power_level = parameters.get("power_level")
        if power_level is None:
            return {"success": False, "error": "Missing power_level"}
            
        # In real implementation, would call PowerPlantManager.adjust_power()
        logger.info(f"Adjusting power to {power_level} MW")
        return {"success": True, "message": f"Power adjusted to {power_level} MW"}
        
    async def _handle_emergency_shutdown(self, parameters: Dict) -> Dict:
        """Handle emergency shutdown command"""
        # In real implementation, would call SafetySystem.manual_emergency_shutdown()
        logger.critical("Emergency shutdown initiated via WebSocket")
        return {"success": True, "message": "Emergency shutdown initiated"}
        
    async def _handle_reset_safety(self, parameters: Dict) -> Dict:
        """Handle reset safety command"""
        # In real implementation, would call SafetySystem.reset_emergency_shutdown()
        logger.info("Safety system reset via WebSocket")
        return {"success": True, "message": "Safety system reset"}
        
    async def _send_success(self, websocket: WebSocket, message: str, data: Dict = None):
        """Send success response"""
        response = {
            "type": "success",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if data:
            response["data"] = data
        await websocket.send_text(json.dumps(response))
        
    async def _send_error(self, websocket: WebSocket, message: str):
        """Send error response"""
        response = {
            "type": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(response))
        
    async def broadcast_to_subscribers(self, subscription_type: str, data: Dict):
        """Broadcast data to all subscribers of a specific type"""
        if not self.active_connections:
            return
            
        message = {
            "type": "broadcast",
            "subscription_type": subscription_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        message_text = json.dumps(message)
        disconnected = set()
        
        for websocket in self.active_connections:
            try:
                if (websocket in self.connection_data and 
                    subscription_type in self.connection_data[websocket]["subscriptions"]):
                    await websocket.send_text(message_text)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)
                
        # Remove disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)
            
    async def broadcast_power_plant_status(self, status: Dict):
        """Broadcast power plant status to subscribers"""
        await self.broadcast_to_subscribers("power_plant_status", status)
        
    async def broadcast_safety_events(self, events: List[Dict]):
        """Broadcast safety events to subscribers"""
        await self.broadcast_to_subscribers("safety_events", {"events": events})
        
    async def broadcast_physics_data(self, physics_data: Dict):
        """Broadcast physics simulation data to subscribers"""
        await self.broadcast_to_subscribers("physics_data", physics_data)
        
    async def broadcast_ai_control_data(self, ai_data: Dict):
        """Broadcast AI control data to subscribers"""
        await self.broadcast_to_subscribers("ai_control", ai_data)
        
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "connections": [
                {
                    "connected_at": data["connected_at"].isoformat(),
                    "last_activity": data["last_activity"].isoformat(),
                    "subscriptions": list(data["subscriptions"])
                }
                for data in self.connection_data.values()
            ]
        }