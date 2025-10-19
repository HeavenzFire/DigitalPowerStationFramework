"""
Physics Simulator - Real-time physics simulation for power generation and distribution
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class TurbineState:
    """State of a power generation turbine"""
    position: float  # Angular position (radians)
    velocity: float  # Angular velocity (rad/s)
    acceleration: float  # Angular acceleration (rad/s²)
    power_output: float  # Power output (MW)
    efficiency: float  # Current efficiency (0-1)
    temperature: float  # Temperature (°C)
    pressure: float  # Pressure (Bar)

@dataclass
class PowerGridNode:
    """Node in the power distribution grid"""
    id: str
    voltage: float  # Voltage (kV)
    current: float  # Current (A)
    power: float  # Power (MW)
    frequency: float  # Frequency (Hz)
    phase: float  # Phase angle (radians)

class DampedHarmonicOscillator:
    """Physics simulation for damped harmonic oscillator (turbine dynamics)"""
    
    def __init__(self, mass: float = 1.0, k: float = 1.0, b: float = 0.1, 
                 x0: float = 1.0, v0: float = 0.0):
        self.mass = mass
        self.k = k
        self.b = b
        self.x0 = x0
        self.v0 = v0
        
        self.omega0 = np.sqrt(k / mass)  # Natural frequency
        self.gamma = b / (2 * mass)     # Damping ratio
        
        # Determine damping regime
        damping_factor = self.gamma**2 - self.omega0**2
        if np.isclose(damping_factor, 0):
            self.regime = "critically_damped"
        elif damping_factor > 0:
            self.regime = "overdamped"
            self.r1 = -self.gamma + np.sqrt(damping_factor)
            self.r2 = -self.gamma - np.sqrt(damping_factor)
        else:
            self.regime = "underdamped"
            self.omega_d = np.sqrt(-damping_factor)
            
    def position(self, t: float) -> float:
        """Calculate position at time t"""
        if self.regime == "underdamped":
            A = self.x0
            B = (self.v0 + self.gamma * self.x0) / self.omega_d
            return np.exp(-self.gamma * t) * (A * np.cos(self.omega_d * t) + B * np.sin(self.omega_d * t))
        elif self.regime == "critically_damped":
            A = self.x0
            B = self.v0 + self.gamma * self.x0
            return np.exp(-self.gamma * t) * (A + B * t)
        else:  # overdamped
            denom = self.r1 - self.r2
            C1 = (self.v0 - self.r2 * self.x0) / denom
            C2 = (self.r1 * self.x0 - self.v0) / denom
            return C1 * np.exp(self.r1 * t) + C2 * np.exp(self.r2 * t)
            
    def velocity(self, t: float) -> float:
        """Calculate velocity at time t"""
        if self.regime == "underdamped":
            A = self.x0
            B = (self.v0 + self.gamma * self.x0) / self.omega_d
            term1 = -self.gamma * self.position(t)
            term2 = np.exp(-self.gamma * t) * (-A * self.omega_d * np.sin(self.omega_d * t) + 
                                              B * self.omega_d * np.cos(self.omega_d * t))
            return term1 + term2
        elif self.regime == "critically_damped":
            A = self.x0
            B = self.v0 + self.gamma * self.x0
            term1 = -self.gamma * self.position(t)
            term2 = np.exp(-self.gamma * t) * B
            return term1 + term2
        else:  # overdamped
            denom = self.r1 - self.r2
            C1 = (self.v0 - self.r2 * self.x0) / denom
            C2 = (self.r1 * self.x0 - self.v0) / denom
            return C1 * self.r1 * np.exp(self.r1 * t) + C2 * self.r2 * np.exp(self.r2 * t)

class PhysicsSimulator:
    """Real-time physics simulation for power plant operations"""
    
    def __init__(self):
        self.running = False
        self._task = None
        self.simulation_time = 0.0
        self.dt = 0.01  # Time step (seconds)
        
        # Turbine simulations
        self.turbines: Dict[str, TurbineState] = {}
        self.oscillators: Dict[str, DampedHarmonicOscillator] = {}
        
        # Power grid simulation
        self.grid_nodes: Dict[str, PowerGridNode] = {}
        
        # Initialize simulation components
        self._initialize_turbines()
        self._initialize_power_grid()
        
    def _initialize_turbines(self):
        """Initialize turbine simulations"""
        turbine_configs = [
            {"id": "turbine_001", "mass": 1000.0, "k": 5000.0, "b": 50.0},
            {"id": "turbine_002", "mass": 1200.0, "k": 6000.0, "b": 60.0},
            {"id": "turbine_003", "mass": 800.0, "k": 4000.0, "b": 40.0}
        ]
        
        for config in turbine_configs:
            # Create oscillator for turbine dynamics
            oscillator = DampedHarmonicOscillator(
                mass=config["mass"],
                k=config["k"],
                b=config["b"],
                x0=1.0,
                v0=0.0
            )
            self.oscillators[config["id"]] = oscillator
            
            # Create turbine state
            turbine = TurbineState(
                position=oscillator.position(0),
                velocity=oscillator.velocity(0),
                acceleration=0.0,
                power_output=0.0,
                efficiency=0.9,
                temperature=25.0,
                pressure=1.0
            )
            self.turbines[config["id"]] = turbine
            
    def _initialize_power_grid(self):
        """Initialize power grid nodes"""
        grid_configs = [
            {"id": "node_001", "voltage": 500.0, "frequency": 50.0},
            {"id": "node_002", "voltage": 220.0, "frequency": 50.0},
            {"id": "node_003", "voltage": 110.0, "frequency": 50.0}
        ]
        
        for config in grid_configs:
            node = PowerGridNode(
                id=config["id"],
                voltage=config["voltage"],
                current=0.0,
                power=0.0,
                frequency=config["frequency"],
                phase=0.0
            )
            self.grid_nodes[config["id"]] = node
            
    async def start(self):
        """Start the physics simulator"""
        self.running = True
        self._task = asyncio.create_task(self._simulation_loop())
        logger.info("Physics Simulator started")
        
    async def stop(self):
        """Stop the physics simulator"""
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("Physics Simulator stopped")
        
    async def _simulation_loop(self):
        """Main physics simulation loop"""
        while self.running:
            try:
                await self._update_simulation()
                self.simulation_time += self.dt
                await asyncio.sleep(self.dt)
            except Exception as e:
                logger.error(f"Error in physics simulation: {e}")
                await asyncio.sleep(0.1)
                
    async def _update_simulation(self):
        """Update one simulation step"""
        # Update turbine dynamics
        for turbine_id, turbine in self.turbines.items():
            oscillator = self.oscillators[turbine_id]
            
            # Update position and velocity from oscillator
            turbine.position = oscillator.position(self.simulation_time)
            turbine.velocity = oscillator.velocity(self.simulation_time)
            
            # Calculate acceleration
            turbine.acceleration = (turbine.velocity - self._get_previous_velocity(turbine_id)) / self.dt
            
            # Calculate power output based on velocity and efficiency
            turbine.power_output = self._calculate_power_output(turbine)
            
            # Update temperature and pressure based on power output
            turbine.temperature = 25.0 + (turbine.power_output / 100.0) * 20.0
            turbine.pressure = 1.0 + (turbine.power_output / 100.0) * 5.0
            
        # Update power grid
        await self._update_power_grid()
        
    def _get_previous_velocity(self, turbine_id: str) -> float:
        """Get previous velocity for acceleration calculation"""
        # Simplified - in practice, would store previous state
        return self.turbines[turbine_id].velocity * 0.99
        
    def _calculate_power_output(self, turbine: TurbineState) -> float:
        """Calculate power output from turbine state"""
        # Power is proportional to velocity squared and efficiency
        base_power = (turbine.velocity ** 2) * 100.0  # Scale factor
        return max(0, base_power * turbine.efficiency)
        
    async def _update_power_grid(self):
        """Update power grid state"""
        total_power = sum(turbine.power_output for turbine in self.turbines.values())
        
        for node in self.grid_nodes.values():
            # Distribute power across grid nodes
            node.power = total_power / len(self.grid_nodes)
            node.current = node.power * 1000 / node.voltage  # P = VI, convert MW to kW
            
            # Update frequency based on power balance
            frequency_deviation = (node.power - 100.0) / 1000.0  # 100 MW reference
            node.frequency = 50.0 + frequency_deviation
            
            # Update phase
            node.phase = (node.phase + 2 * math.pi * node.frequency * self.dt) % (2 * math.pi)
            
    def get_turbine_status(self, turbine_id: str) -> Optional[Dict]:
        """Get status of a specific turbine"""
        if turbine_id not in self.turbines:
            return None
            
        turbine = self.turbines[turbine_id]
        return {
            "id": turbine_id,
            "position": turbine.position,
            "velocity": turbine.velocity,
            "acceleration": turbine.acceleration,
            "power_output": turbine.power_output,
            "efficiency": turbine.efficiency,
            "temperature": turbine.temperature,
            "pressure": turbine.pressure
        }
        
    def get_all_turbines_status(self) -> Dict[str, Dict]:
        """Get status of all turbines"""
        return {turbine_id: self.get_turbine_status(turbine_id) 
                for turbine_id in self.turbines.keys()}
        
    def get_power_grid_status(self) -> Dict[str, Dict]:
        """Get power grid status"""
        return {
            node_id: {
                "id": node_id,
                "voltage": node.voltage,
                "current": node.current,
                "power": node.power,
                "frequency": node.frequency,
                "phase": node.phase
            }
            for node_id, node in self.grid_nodes.items()
        }
        
    def get_status(self) -> Dict:
        """Get overall physics simulator status"""
        total_power = sum(turbine.power_output for turbine in self.turbines.values())
        avg_efficiency = np.mean([turbine.efficiency for turbine in self.turbines.values()])
        
        return {
            "status": "running" if self.running else "stopped",
            "simulation_time": self.simulation_time,
            "total_power_output": total_power,
            "average_efficiency": avg_efficiency,
            "turbine_count": len(self.turbines),
            "grid_node_count": len(self.grid_nodes)
        }
        
    def adjust_turbine_parameters(self, turbine_id: str, **kwargs):
        """Adjust turbine simulation parameters"""
        if turbine_id not in self.oscillators:
            return False
            
        oscillator = self.oscillators[turbine_id]
        
        if "mass" in kwargs:
            oscillator.mass = kwargs["mass"]
        if "k" in kwargs:
            oscillator.k = kwargs["k"]
        if "b" in kwargs:
            oscillator.b = kwargs["b"]
            
        # Recalculate oscillator parameters
        oscillator.omega0 = np.sqrt(oscillator.k / oscillator.mass)
        oscillator.gamma = oscillator.b / (2 * oscillator.mass)
        
        logger.info(f"Adjusted parameters for turbine {turbine_id}: {kwargs}")
        return True