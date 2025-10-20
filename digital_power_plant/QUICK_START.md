# Digital Power Plant - Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

### Step 1: Start the System
```bash
cd digital_power_plant
python main.py
```

### Step 2: Open the Dashboard
Navigate to `http://localhost:8000` in your browser

### Step 3: Explore the Interface

#### Main Dashboard Features:
- **Power Overview**: Real-time power output and efficiency
- **Unit Controls**: Start/stop power generation units
- **Safety Monitor**: View active safety events
- **Physics Data**: Real-time turbine and grid simulation
- **AI Status**: Neural network control system status

#### Energy Visualization:
- Navigate to `http://localhost:8000/static/energy_visualization.html`
- Interactive quantum energy matrix
- Click cells to add energy
- Adjust grid size and energy levels

### Step 4: Basic Operations

#### Starting a Unit:
1. Go to the Generation Units panel
2. Click "Start" on any offline unit
3. Watch the power output increase

#### Monitoring Safety:
1. Check the Safety System panel
2. View any active safety events
3. Monitor safety level indicators

#### AI Control:
1. View AI Controller panel
2. Monitor learning performance
3. Adjust learning rate if needed

### Step 5: Emergency Procedures

#### Emergency Shutdown:
- Click the red "EMERGENCY SHUTDOWN" button
- Confirm the action
- All systems will shut down safely

#### Safety Reset:
- Use the API endpoint: `POST /api/v1/safety/reset`
- Or use the WebSocket command

## 🔧 Quick Configuration

### Adjust Power Plant Settings:
```python
# In core/power_plant_manager.py
units = [
    {
        "id": "unit_001",
        "name": "Your Custom Unit",
        "max_capacity": 1000.0,  # MW
        "efficiency": 0.95
    }
]
```

### Modify Safety Thresholds:
```python
# In core/safety_system.py
thresholds = {
    "temperature": {"warning": 85.0, "critical": 105.0, "emergency": 125.0}
}
```

### Adjust AI Learning:
```python
# Via API
POST /api/v1/ai/learning-rate?learning_rate=0.01
```

## 📊 Key Metrics to Watch

1. **Total Power Output**: Should match demand
2. **System Efficiency**: Higher is better (target: >90%)
3. **Safety Level**: Should remain "normal"
4. **AI Performance**: Learning loss should decrease over time
5. **Temperature**: Should stay below warning threshold

## 🆘 Troubleshooting

### System Won't Start:
- Check Python version (3.8+ required)
- Install dependencies: `pip install -r requirements.txt`
- Check port 8000 is available

### Dashboard Not Loading:
- Verify server is running
- Check browser console for errors
- Try refreshing the page

### Units Not Responding:
- Check safety system status
- Verify no active safety events
- Check unit configuration

### WebSocket Connection Issues:
- Check firewall settings
- Verify WebSocket support in browser
- Check server logs for errors

## 🎯 Next Steps

1. **Explore the API**: Use the REST endpoints to integrate with other systems
2. **Customize the Interface**: Modify the dashboard for your needs
3. **Add New Units**: Configure additional power generation units
4. **Implement Monitoring**: Set up external monitoring systems
5. **Scale the System**: Deploy multiple power plants

## 📚 Additional Resources

- **Full Documentation**: See `DOCUMENTATION.md`
- **API Reference**: Available at `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **WebSocket Test**: Use browser developer tools

---

**Ready to revolutionize power generation? Your digital power plant is now online!** ⚡