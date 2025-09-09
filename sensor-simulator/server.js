// sensor-simulator/server.js
const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Sensor simulation configuration
const SIMULATION_CONFIG = {
  nodeCount: 6,
  zones: ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D'],
  updateInterval: 3000, // 3 seconds
  modes: {
    NORMAL: 'normal',
    LEAK_SIMULATION: 'leak',
    MAINTENANCE: 'maintenance',
    EMERGENCY: 'emergency'
  }
};

let currentMode = SIMULATION_CONFIG.modes.NORMAL;
let leakZone = null;
let emergencyActive = false;

class SensorNode {
  constructor(nodeId, zone, position) {
    this.nodeId = nodeId;
    this.zone = zone;
    this.position = position;
    this.isOnline = true;
    this.lastReading = null;
    this.calibrationOffset = {
      methane: Math.random() * 10 - 5,
      lpg: Math.random() * 8 - 4,
      carbonMonoxide: Math.random() * 5 - 2.5,
      hydrogenSulfide: Math.random() * 3 - 1.5
    };
  }

  generateReading() {
    if (!this.isOnline) {
      return null;
    }

    const timestamp = new Date().toISOString();
    let gases = this.generateGasReadings();
    let environmental = this.generateEnvironmentalReadings();

    const reading = {
      nodeId: this.nodeId,
      timestamp,
      gases,
      environmental,
      location: {
        x: this.position.x,
        y: this.position.y,
        zone: this.zone
      },
      status: 'online',
      batteryLevel: Math.max(20, 100 - Math.random() * 2), // Slow battery drain
      signalStrength: -45 - Math.random() * 20 // dBm
    };

    this.lastReading = reading;
    return reading;
  }

  generateGasReadings() {
    let baseValues = {
      methane: 245,
      lpg: 156,
      carbonMonoxide: 89,
      hydrogenSulfide: 23
    };

    // Apply calibration offsets
    Object.keys(baseValues).forEach(gas => {
      baseValues[gas] += this.calibrationOffset[gas];
    });

    // Simulate different modes
    switch (currentMode) {
      case SIMULATION_CONFIG.modes.LEAK_SIMULATION:
        if (leakZone === this.zone || leakZone === 'all') {
          return this.generateLeakScenario(baseValues);
        }
        break;
      
      case SIMULATION_CONFIG.modes.EMERGENCY:
        return this.generateEmergencyScenario(baseValues);
      
      case SIMULATION_CONFIG.modes.MAINTENANCE:
        return this.generateMaintenanceScenario(baseValues);
      
      default: // NORMAL
        return this.generateNormalReadings(baseValues);
    }

    return this.generateNormalReadings(baseValues);
  }

  generateNormalReadings(baseValues) {
    // Add small random variations to simulate normal sensor drift
    return {
      methane: Math.max(0, baseValues.methane + (Math.random() * 20 - 10)),
      lpg: Math.max(0, baseValues.lpg + (Math.random() * 15 - 7.5)),
      carbonMonoxide: Math.max(0, baseValues.carbonMonoxide + (Math.random() * 10 - 5)),
      hydrogenSulfide: Math.max(0, baseValues.hydrogenSulfide + (Math.random() * 6 - 3))
    };
  }

  generateLeakScenario(baseValues) {
    // Simulate gas leak with gradually increasing concentrations
    const leakIntensity = 1.5 + Math.random() * 2; // 1.5x to 3.5x normal levels
    const timeVariation = Math.sin(Date.now() / 10000) * 0.3 + 1; // Time-based variation
    
    return {
      methane: baseValues.methane * leakIntensity * timeVariation + (Math.random() * 50),
      lpg: baseValues.lpg * leakIntensity * timeVariation + (Math.random() * 40),
      carbonMonoxide: baseValues.carbonMonoxide * leakIntensity * timeVariation + (Math.random() * 30),
      hydrogenSulfide: baseValues.hydrogenSulfide * leakIntensity * timeVariation + (Math.random() * 20)
    };
  }

  generateEmergencyScenario(baseValues) {
    // Simulate critical gas levels
    const criticalMultiplier = 3 + Math.random() * 2; // 3x to 5x normal levels
    
    return {
      methane: baseValues.methane * criticalMultiplier + (Math.random() * 100),
      lpg: baseValues.lpg * criticalMultiplier + (Math.random() * 80),
      carbonMonoxide: baseValues.carbonMonoxide * criticalMultiplier + (Math.random() * 60),
      hydrogenSulfide: baseValues.hydrogenSulfide * criticalMultiplier + (Math.random() * 40)
    };
  }

  generateMaintenanceScenario(baseValues) {
    // Simulate erratic readings during maintenance
    const maintenanceVariation = 0.5 + Math.random() * 1.5; // 0.5x to 2x variation
    
    return {
      methane: baseValues.methane * maintenanceVariation,
      lpg: baseValues.lpg * maintenanceVariation,
      carbonMonoxide: baseValues.carbonMonoxide * maintenanceVariation,
      hydrogenSulfide: baseValues.hydrogenSulfide * maintenanceVariation
    };
  }

  generateEnvironmentalReadings() {
    // Simulate environmental conditions
    const baseTemp = 23.5;
    const baseHumidity = 65.2;
    const basePressure = 1013.25;

    // Add seasonal and daily variations
    const timeOfDay = new Date().getHours();
    const tempVariation = Math.sin((timeOfDay - 6) * Math.PI / 12) * 3; // Daily temperature cycle
    
    return {
      temperature: baseTemp + tempVariation + (Math.random() * 4 - 2),
      humidity: Math.max(30, Math.min(95, baseHumidity + (Math.random() * 10 - 5))),
      pressure: basePressure + (Math.random() * 20 - 10)
    };
  }

  setOnline(status) {
    this.isOnline = status;
  }
}

// Initialize sensor nodes
const sensorNodes = [];
const zones = SIMULATION_CONFIG.zones;

for (let i = 0; i < SIMULATION_CONFIG.nodeCount; i++) {
  const zone = zones[i % zones.length];
  const position = {
    x: Math.random() * 1000, // meters
    y: Math.random() * 1000
  };
  
  const nodeId = `ESP32_${String(i + 1).padStart(3, '0')}`;
  sensorNodes.push(new SensorNode(nodeId, zone, position));
}

console.log(`Initialized ${sensorNodes.length} sensor nodes across ${zones.length} zones`);

// Simulation control functions
function startSimulation() {
  setInterval(() => {
    // Generate readings from all online sensors
    sensorNodes.forEach(node => {
      const reading = node.generateReading();
      if (reading) {
        // Emit to connected clients
        io.emit('sensor_data', reading);
        
        // Send to backend API
        sendToBackend(reading);
      }
    });
    
    // Emit system status
    io.emit('system_status', {
      mode: currentMode,
      activeSensors: sensorNodes.filter(n => n.isOnline).length,
      totalSensors: sensorNodes.length,
      leakZone: leakZone,
      timestamp: new Date().toISOString()
    });
    
  }, SIMULATION_CONFIG.updateInterval);
}

async function sendToBackend(reading) {
  try {
    const response = await fetch('http://localhost:3001/api/sensor-data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(reading)
    });
    
    if (!response.ok) {
      console.error('Failed to send data to backend:', response.statusText);
    }
  } catch (error) {
    // Backend might not be running, continue simulation
    // console.error('Backend connection error:', error.message);
  }
}

// Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'online',
    service: 'Sensor Data Simulator',
    sensors: {
      total: sensorNodes.length,
      online: sensorNodes.filter(n => n.isOnline).length,
      zones: zones
    },
    simulation: {
      mode: currentMode,
      leakZone: leakZone,
      updateInterval: SIMULATION_CONFIG.updateInterval
    },
    timestamp: new Date().toISOString()
  });
});

app.get('/sensors', (req, res) => {
  const sensorStatus = sensorNodes.map(node => ({
    nodeId: node.nodeId,
    zone: node.zone,
    position: node.position,
    online: node.isOnline,
    lastReading: node.lastReading ? {
      timestamp: node.lastReading.timestamp,
      gases: node.lastReading.gases,
      batteryLevel: node.lastReading.batteryLevel
    } : null
  }));
  
  res.json({
    sensors: sensorStatus,
    summary: {
      total: sensorNodes.length,
      online: sensorNodes.filter(n => n.isOnline).length,
      zones: zones
    }
  });
});

app.post('/simulation/mode', (req, res) => {
  const { mode, zone } = req.body;
  
  if (!Object.values(SIMULATION_CONFIG.modes).includes(mode)) {
    return res.status(400).json({ 
      error: 'Invalid mode', 
      validModes: Object.values(SIMULATION_CONFIG.modes) 
    });
  }
  
  currentMode = mode;
  leakZone = zone || null;
  
  if (mode === SIMULATION_CONFIG.modes.EMERGENCY) {
    emergencyActive = true;
  } else {
    emergencyActive = false;
  }
  
  res.json({
    success: true,
    mode: currentMode,
    leakZone: leakZone,
    message: `Simulation mode changed to ${mode}${zone ? ` in ${zone}` : ''}`
  });
});

app.post('/simulation/leak', (req, res) => {
  const { zone = 'Zone_A', intensity = 'medium' } = req.body;
  
  currentMode = SIMULATION_CONFIG.modes.LEAK_SIMULATION;
  leakZone = zone;
  
  res.json({
    success: true,
    message: `Gas leak simulation started in ${zone}`,
    intensity: intensity,
    timestamp: new Date().toISOString()
  });
});

app.post('/simulation/emergency', (req, res) => {
  currentMode = SIMULATION_CONFIG.modes.EMERGENCY;
  emergencyActive = true;
  leakZone = 'all';
  
  res.json({
    success: true,
    message: 'Emergency scenario activated',
    affectedZones: 'all',
    timestamp: new Date().toISOString()
  });
});

app.post('/simulation/reset', (req, res) => {
  currentMode = SIMULATION_CONFIG.modes.NORMAL;
  leakZone = null;
  emergencyActive = false;
  
  // Reset all sensors to online
  sensorNodes.forEach(node => node.setOnline(true));
  
  res.json({
    success: true,
    message: 'Simulation reset to normal operation',
    timestamp: new Date().toISOString()
  });
});

app.post('/sensors/:nodeId/toggle', (req, res) => {
  const { nodeId } = req.params;
  const node = sensorNodes.find(n => n.nodeId === nodeId);
  
  if (!node) {
    return res.status(404).json({ error: 'Sensor node not found' });
  }
  
  node.setOnline(!node.isOnline);
  
  res.json({
    success: true,
    nodeId: nodeId,
    status: node.isOnline ? 'online' : 'offline',
    message: `Sensor ${nodeId} ${node.isOnline ? 'activated' : 'deactivated'}`
  });
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('Client connected to sensor simulator:', socket.id);
  
  // Send current sensor status
  socket.emit('sensor_status', {
    sensors: sensorNodes.length,
    mode: currentMode,
    zones: zones
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected from sensor simulator:', socket.id);
  });
  
  socket.on('request_sensor_data', () => {
    // Send latest readings from all sensors
    sensorNodes.forEach(node => {
      if (node.lastReading) {
        socket.emit('sensor_data', node.lastReading);
      }
    });
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

const PORT = process.env.PORT || 3003;

server.listen(PORT, () => {
  console.log(`🔬 Sensor simulator running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🌐 WebSocket available for real-time data`);
  
  // Start the simulation
  startSimulation();
});

module.exports = { app, io, sensorNodes };