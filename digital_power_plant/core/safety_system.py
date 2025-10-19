"""
Safety System - Automated safety monitoring and emergency response
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyEventType(Enum):
    TEMPERATURE_HIGH = "temperature_high"
    PRESSURE_HIGH = "pressure_high"
    POWER_OVERLOAD = "power_overload"
    EFFICIENCY_LOW = "efficiency_low"
    GRID_FREQUENCY_DEVIATION = "grid_frequency_deviation"
    EQUIPMENT_FAILURE = "equipment_failure"
    FIRE_DETECTED = "fire_detected"
    GAS_LEAK = "gas_leak"

@dataclass
class SafetyEvent:
    """Represents a safety event"""
    id: str
    event_type: SafetyEventType
    level: SafetyLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SafetyThreshold:
    """Safety threshold configuration"""
    parameter: str
    warning_value: float
    critical_value: float
    emergency_value: float
    unit: str

class SafetySystem:
    """Automated safety monitoring and emergency response system"""
    
    def __init__(self):
        self.running = False
        self._task = None
        
        # Safety events and history
        self.active_events: List[SafetyEvent] = []
        self.event_history: List[SafetyEvent] = []
        self.event_counter = 0
        
        # Safety thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Emergency procedures
        self.emergency_procedures: Dict[SafetyEventType, Callable] = {
            SafetyEventType.TEMPERATURE_HIGH: self._handle_temperature_emergency,
            SafetyEventType.PRESSURE_HIGH: self._handle_pressure_emergency,
            SafetyEventType.POWER_OVERLOAD: self._handle_power_overload,
            SafetyEventType.EFFICIENCY_LOW: self._handle_efficiency_low,
            SafetyEventType.GRID_FREQUENCY_DEVIATION: self._handle_frequency_deviation,
            SafetyEventType.EQUIPMENT_FAILURE: self._handle_equipment_failure,
            SafetyEventType.FIRE_DETECTED: self._handle_fire_emergency,
            SafetyEventType.GAS_LEAK: self._handle_gas_leak
        }
        
        # Safety status
        self.current_safety_level = SafetyLevel.NORMAL
        self.emergency_shutdown_active = False
        
    def _initialize_thresholds(self) -> Dict[str, SafetyThreshold]:
        """Initialize safety thresholds"""
        return {
            "temperature": SafetyThreshold("temperature", 80.0, 100.0, 120.0, "°C"),
            "pressure": SafetyThreshold("pressure", 20.0, 25.0, 30.0, "Bar"),
            "power_output": SafetyThreshold("power_output", 2000.0, 2500.0, 3000.0, "MW"),
            "efficiency": SafetyThreshold("efficiency", 0.7, 0.6, 0.5, "ratio"),
            "grid_frequency": SafetyThreshold("grid_frequency", 49.5, 49.0, 48.5, "Hz"),
            "voltage": SafetyThreshold("voltage", 480.0, 450.0, 400.0, "kV"),
            "current": SafetyThreshold("current", 4000.0, 4500.0, 5000.0, "A")
        }
        
    async def start(self):
        """Start the safety system"""
        self.running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Safety System started")
        
    async def stop(self):
        """Stop the safety system"""
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("Safety System stopped")
        
    async def _monitoring_loop(self):
        """Main safety monitoring loop"""
        while self.running:
            try:
                await self._check_safety_conditions()
                await self._process_active_events()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in safety monitoring: {e}")
                await asyncio.sleep(5)
                
    async def _check_safety_conditions(self):
        """Check all safety conditions"""
        # This would normally receive data from PowerPlantManager and PhysicsSimulator
        # For now, we'll simulate some monitoring
        
        # Simulate temperature monitoring
        simulated_temperature = 75.0 + (datetime.now().second * 0.5)  # Vary with time
        await self._check_parameter("temperature", simulated_temperature)
        
        # Simulate pressure monitoring
        simulated_pressure = 15.0 + (datetime.now().second * 0.1)
        await self._check_parameter("pressure", simulated_pressure)
        
        # Simulate power output monitoring
        simulated_power = 1500.0 + (datetime.now().second * 10)
        await self._check_parameter("power_output", simulated_power)
        
        # Simulate efficiency monitoring
        simulated_efficiency = 0.85 + (datetime.now().second * 0.001)
        await self._check_parameter("efficiency", simulated_efficiency)
        
    async def _check_parameter(self, parameter: str, value: float):
        """Check a specific parameter against safety thresholds"""
        if parameter not in self.thresholds:
            return
            
        threshold = self.thresholds[parameter]
        level = None
        event_type = None
        
        # Determine safety level
        if value >= threshold.emergency_value:
            level = SafetyLevel.EMERGENCY
        elif value >= threshold.critical_value:
            level = SafetyLevel.CRITICAL
        elif value >= threshold.warning_value:
            level = SafetyLevel.WARNING
        else:
            return  # Normal operation
            
        # Map parameter to event type
        event_type_map = {
            "temperature": SafetyEventType.TEMPERATURE_HIGH,
            "pressure": SafetyEventType.PRESSURE_HIGH,
            "power_output": SafetyEventType.POWER_OVERLOAD,
            "efficiency": SafetyEventType.EFFICIENCY_LOW,
            "grid_frequency": SafetyEventType.GRID_FREQUENCY_DEVIATION
        }
        
        event_type = event_type_map.get(parameter)
        if not event_type:
            return
            
        # Check if this event is already active
        active_event = next((e for e in self.active_events 
                           if e.event_type == event_type and not e.resolved), None)
        
        if not active_event:
            # Create new safety event
            await self._create_safety_event(event_type, level, value, threshold.unit)
        else:
            # Update existing event if level increased
            if level.value > active_event.level.value:
                active_event.level = level
                active_event.message = f"{parameter} is {value:.2f} {threshold.unit} - {level.value.upper()}"
                
    async def _create_safety_event(self, event_type: SafetyEventType, level: SafetyLevel, 
                                 value: float, unit: str):
        """Create a new safety event"""
        self.event_counter += 1
        event_id = f"SE_{self.event_counter:06d}"
        
        message = f"{event_type.value.replace('_', ' ').title()}: {value:.2f} {unit} - {level.value.upper()}"
        
        event = SafetyEvent(
            id=event_id,
            event_type=event_type,
            level=level,
            message=message,
            timestamp=datetime.now()
        )
        
        self.active_events.append(event)
        self.event_history.append(event)
        
        # Update overall safety level
        if level.value > self.current_safety_level.value:
            self.current_safety_level = level
            
        logger.warning(f"Safety Event Created: {message}")
        
        # Execute emergency procedure if critical or emergency
        if level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            await self._execute_emergency_procedure(event)
            
    async def _process_active_events(self):
        """Process active safety events"""
        for event in self.active_events[:]:  # Copy list to avoid modification during iteration
            if not event.resolved:
                # Check if event should be resolved (simplified logic)
                if self._should_resolve_event(event):
                    await self._resolve_event(event)
                    
    def _should_resolve_event(self, event: SafetyEvent) -> bool:
        """Determine if an event should be resolved"""
        # Simplified resolution logic - in practice, would check actual conditions
        time_since_creation = datetime.now() - event.timestamp
        return time_since_creation > timedelta(minutes=5)  # Auto-resolve after 5 minutes
        
    async def _resolve_event(self, event: SafetyEvent):
        """Resolve a safety event"""
        event.resolved = True
        event.resolution_time = datetime.now()
        
        # Remove from active events
        if event in self.active_events:
            self.active_events.remove(event)
            
        # Update overall safety level
        if not self.active_events:
            self.current_safety_level = SafetyLevel.NORMAL
        else:
            # Set to highest level of active events
            max_level = max(event.level for event in self.active_events)
            self.current_safety_level = max_level
            
        logger.info(f"Safety Event Resolved: {event.id} - {event.message}")
        
    async def _execute_emergency_procedure(self, event: SafetyEvent):
        """Execute emergency procedure for a safety event"""
        procedure = self.emergency_procedures.get(event.event_type)
        if procedure:
            try:
                await procedure(event)
            except Exception as e:
                logger.error(f"Error executing emergency procedure for {event.event_type}: {e}")
                
    async def _handle_temperature_emergency(self, event: SafetyEvent):
        """Handle temperature emergency"""
        logger.critical(f"TEMPERATURE EMERGENCY: {event.message}")
        # In real implementation, would trigger cooling systems, reduce power, etc.
        
    async def _handle_pressure_emergency(self, event: SafetyEvent):
        """Handle pressure emergency"""
        logger.critical(f"PRESSURE EMERGENCY: {event.message}")
        # In real implementation, would trigger pressure relief systems
        
    async def _handle_power_overload(self, event: SafetyEvent):
        """Handle power overload emergency"""
        logger.critical(f"POWER OVERLOAD: {event.message}")
        # In real implementation, would reduce power output, disconnect loads
        
    async def _handle_efficiency_low(self, event: SafetyEvent):
        """Handle low efficiency warning"""
        logger.warning(f"LOW EFFICIENCY: {event.message}")
        # In real implementation, would trigger maintenance procedures
        
    async def _handle_frequency_deviation(self, event: SafetyEvent):
        """Handle grid frequency deviation"""
        logger.critical(f"FREQUENCY DEVIATION: {event.message}")
        # In real implementation, would adjust power output to stabilize frequency
        
    async def _handle_equipment_failure(self, event: SafetyEvent):
        """Handle equipment failure"""
        logger.critical(f"EQUIPMENT FAILURE: {event.message}")
        # In real implementation, would isolate failed equipment, start backup systems
        
    async def _handle_fire_emergency(self, event: SafetyEvent):
        """Handle fire emergency"""
        logger.critical(f"FIRE DETECTED: {event.message}")
        # In real implementation, would trigger fire suppression, evacuation procedures
        
    async def _handle_gas_leak(self, event: SafetyEvent):
        """Handle gas leak emergency"""
        logger.critical(f"GAS LEAK DETECTED: {event.message}")
        # In real implementation, would trigger gas shutoff, ventilation, evacuation
        
    def get_status(self) -> Dict:
        """Get safety system status"""
        return {
            "status": "running" if self.running else "stopped",
            "current_safety_level": self.current_safety_level.value,
            "active_events": len(self.active_events),
            "total_events_today": len([e for e in self.event_history 
                                     if e.timestamp.date() == datetime.now().date()]),
            "emergency_shutdown_active": self.emergency_shutdown_active
        }
        
    def get_active_events(self) -> List[Dict]:
        """Get active safety events"""
        return [
            {
                "id": event.id,
                "event_type": event.event_type.value,
                "level": event.level.value,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
                "resolved": event.resolved
            }
            for event in self.active_events
        ]
        
    def get_event_history(self, limit: int = 100) -> List[Dict]:
        """Get safety event history"""
        return [
            {
                "id": event.id,
                "event_type": event.event_type.value,
                "level": event.level.value,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
                "resolved": event.resolved,
                "resolution_time": event.resolution_time.isoformat() if event.resolution_time else None
            }
            for event in self.event_history[-limit:]
        ]
        
    def update_threshold(self, parameter: str, warning: float, critical: float, emergency: float):
        """Update safety threshold for a parameter"""
        if parameter in self.thresholds:
            self.thresholds[parameter].warning_value = warning
            self.thresholds[parameter].critical_value = critical
            self.thresholds[parameter].emergency_value = emergency
            logger.info(f"Updated threshold for {parameter}: W={warning}, C={critical}, E={emergency}")
            
    def manual_emergency_shutdown(self):
        """Manually trigger emergency shutdown"""
        self.emergency_shutdown_active = True
        self.current_safety_level = SafetyLevel.EMERGENCY
        
        # Create emergency shutdown event
        event = SafetyEvent(
            id=f"SE_{self.event_counter + 1:06d}",
            event_type=SafetyEventType.EQUIPMENT_FAILURE,
            level=SafetyLevel.EMERGENCY,
            message="MANUAL EMERGENCY SHUTDOWN INITIATED",
            timestamp=datetime.now()
        )
        
        self.active_events.append(event)
        self.event_history.append(event)
        
        logger.critical("MANUAL EMERGENCY SHUTDOWN INITIATED")
        
    def reset_emergency_shutdown(self):
        """Reset emergency shutdown status"""
        self.emergency_shutdown_active = False
        self.current_safety_level = SafetyLevel.NORMAL
        logger.info("Emergency shutdown reset")