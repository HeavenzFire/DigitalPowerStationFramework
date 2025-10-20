"""
Power Plant Manager - Central control system for the digital power plant
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PowerPlantStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

@dataclass
class PowerGenerationUnit:
    """Represents a power generation unit"""
    id: str
    name: str
    power_output: float  # MW
    max_capacity: float  # MW
    efficiency: float    # 0.0 to 1.0
    status: str
    last_maintenance: datetime
    next_maintenance: datetime

@dataclass
class PowerPlantMetrics:
    """Power plant operational metrics"""
    timestamp: datetime
    total_power_output: float  # MW
    total_demand: float        # MW
    efficiency: float          # Overall efficiency
    temperature: float         # °C
    pressure: float           # Bar
    fuel_consumption: float   # Units per hour
    co2_emissions: float      # Tons per hour

class PowerPlantManager:
    """Central management system for the digital power plant"""
    
    def __init__(self):
        self.status = PowerPlantStatus.STARTING
        self.generation_units: List[PowerPlantGenerationUnit] = []
        self.metrics_history: List[PowerPlantMetrics] = []
        self.running = False
        self._task = None
        
        # Initialize generation units
        self._initialize_generation_units()
        
    def _initialize_generation_units(self):
        """Initialize power generation units"""
        units = [
            PowerGenerationUnit(
                id="unit_001",
                name="Quantum Turbine Alpha",
                power_output=0.0,
                max_capacity=500.0,
                efficiency=0.95,
                status="offline",
                last_maintenance=datetime.now(),
                next_maintenance=datetime.now()
            ),
            PowerGenerationUnit(
                id="unit_002", 
                name="Neural Generator Beta",
                power_output=0.0,
                max_capacity=750.0,
                efficiency=0.92,
                status="offline",
                last_maintenance=datetime.now(),
                next_maintenance=datetime.now()
            ),
            PowerGenerationUnit(
                id="unit_003",
                name="Vortex Reactor Gamma", 
                power_output=0.0,
                max_capacity=1000.0,
                efficiency=0.88,
                status="offline",
                last_maintenance=datetime.now(),
                next_maintenance=datetime.now()
            )
        ]
        self.generation_units = units
        
    async def start(self):
        """Start the power plant manager"""
        self.running = True
        self.status = PowerPlantStatus.STARTING
        self._task = asyncio.create_task(self._main_loop())
        logger.info("Power Plant Manager started")
        
    async def stop(self):
        """Stop the power plant manager"""
        self.running = False
        self.status = PowerPlantStatus.SHUTDOWN
        if self._task:
            self._task.cancel()
        logger.info("Power Plant Manager stopped")
        
    async def _main_loop(self):
        """Main operational loop"""
        while self.running:
            try:
                await self._update_metrics()
                await self._monitor_units()
                await self._optimize_operations()
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
                
    async def _update_metrics(self):
        """Update power plant metrics"""
        total_output = sum(unit.power_output for unit in self.generation_units)
        total_demand = total_output * 1.1  # Simulate 10% higher demand
        
        metrics = PowerPlantMetrics(
            timestamp=datetime.now(),
            total_power_output=total_output,
            total_demand=total_demand,
            efficiency=sum(unit.efficiency for unit in self.generation_units) / len(self.generation_units),
            temperature=45.0 + (total_output / 1000) * 10,  # Simulate temperature based on output
            pressure=15.0 + (total_output / 1000) * 2,      # Simulate pressure
            fuel_consumption=total_output * 0.8,             # Simulate fuel consumption
            co2_emissions=total_output * 0.5                 # Simulate CO2 emissions
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
    async def _monitor_units(self):
        """Monitor generation units for issues"""
        for unit in self.generation_units:
            if unit.status == "online" and unit.efficiency < 0.8:
                logger.warning(f"Unit {unit.id} efficiency below threshold: {unit.efficiency}")
                
    async def _optimize_operations(self):
        """Optimize power plant operations"""
        # Simple optimization: adjust output based on demand
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            if latest_metrics.total_demand > latest_metrics.total_power_output:
                await self._increase_power_output()
            elif latest_metrics.total_demand < latest_metrics.total_power_output * 0.9:
                await self._decrease_power_output()
                
    async def _increase_power_output(self):
        """Increase power output"""
        for unit in self.generation_units:
            if unit.status == "online" and unit.power_output < unit.max_capacity:
                unit.power_output = min(unit.max_capacity, unit.power_output + 10.0)
                
    async def _decrease_power_output(self):
        """Decrease power output"""
        for unit in self.generation_units:
            if unit.status == "online" and unit.power_output > 0:
                unit.power_output = max(0, unit.power_output - 10.0)
                
    def start_unit(self, unit_id: str) -> bool:
        """Start a generation unit"""
        unit = next((u for u in self.generation_units if u.id == unit_id), None)
        if unit and unit.status == "offline":
            unit.status = "online"
            unit.power_output = unit.max_capacity * 0.5  # Start at 50% capacity
            logger.info(f"Started unit {unit_id}")
            return True
        return False
        
    def stop_unit(self, unit_id: str) -> bool:
        """Stop a generation unit"""
        unit = next((u for u in self.generation_units if u.id == unit_id), None)
        if unit and unit.status == "online":
            unit.status = "offline"
            unit.power_output = 0.0
            logger.info(f"Stopped unit {unit_id}")
            return True
        return False
        
    def get_status(self) -> Dict:
        """Get current power plant status"""
        return {
            "status": self.status.value,
            "units": [
                {
                    "id": unit.id,
                    "name": unit.name,
                    "power_output": unit.power_output,
                    "max_capacity": unit.max_capacity,
                    "efficiency": unit.efficiency,
                    "status": unit.status
                }
                for unit in self.generation_units
            ],
            "metrics": self.metrics_history[-1].__dict__ if self.metrics_history else None
        }
        
    def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Get metrics history"""
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "total_power_output": m.total_power_output,
                "total_demand": m.total_demand,
                "efficiency": m.efficiency,
                "temperature": m.temperature,
                "pressure": m.pressure,
                "fuel_consumption": m.fuel_consumption,
                "co2_emissions": m.co2_emissions
            }
            for m in self.metrics_history[-limit:]
        ]