// backend/server.js
const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// In-memory storage (replace with actual database in production)
let sensorReadings = [];
let emergencyEvents = [];
let ventilationStatus = {};

// Routes
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'online', 
    timestamp: new Date().toISOString(),
    services: {
      api: 'running',
      websocket: 'connected'
    }
  });
});

app.get('/api/sensors', (req, res) => {
  const recentReadings = sensorReadings.slice(-50);
  res.json({
    readings: recentReadings,
    totalSensors: 24,
    activeSensors: 24,
    lastUpdate: recentReadings.length > 0 ? recentReadings[recentReadings.length - 1].timestamp : null
  });
});

app.get('/api/sensors/:nodeId', (req, res) => {
  const { nodeId } = req.params;
  const nodeReadings = sensorReadings.filter(reading => reading.nodeId === nodeId);
  
  if (nodeReadings.length === 0) {
    return res.status(404).json({ error: 'Sensor node not found' });
  }
  
  res.json({
    nodeId,
    readings: nodeReadings.slice(-20),
    currentReading: nodeReadings[nodeReadings.length - 1]
  });
});

app.post('/api/emergency', async (req, res) => {
  const { action, zone, gasType, value, timestamp } = req.body;
  
  const emergencyEvent = {
    id: generateId(),
    action,
    zone: zone || 'all',
    gasType,
    value,
    timestamp: timestamp || new Date().toISOString(),
    status: 'active'
  };
  
  emergencyEvents.push(emergencyEvent);
  
  // Broadcast emergency event
  io.emit('emergency_event', emergencyEvent);
  
  // Log to blockchain
  try {
    await logToBlockchain('emergency_protocol', emergencyEvent);
  } catch (error) {
    console.error('Failed to log emergency to blockchain:', error);
  }
  
  res.json({
    success: true,
    eventId: emergencyEvent.id,
    message: `Emergency protocol ${action} activated`
  });
});

app.post('/api/ventilation', async (req, res) => {
  const { zone, action, fanSpeed } = req.body;
  
  const ventilationEvent = {
    id: generateId(),
    zone: zone || 'all',
    action, // 'activate', 'deactivate', 'adjust'
    fanSpeed: fanSpeed || 100,
    timestamp: new Date().toISOString()
  };
  
  ventilationStatus[zone || 'all'] = {
    active: action === 'activate',
    fanSpeed: fanSpeed || 100,
    lastUpdate: new Date().toISOString()
  };
  
  // Broadcast ventilation status
  io.emit('ventilation_update', ventilationEvent);
  
  // Log to blockchain
  try {
    await logToBlockchain('ventilation_control', ventilationEvent);
  } catch (error) {
    console.error('Failed to log ventilation to blockchain:', error);
  }
  
  res.json({
    success: true,
    eventId: ventilationEvent.id,
    message: `Ventilation ${action} in ${zone}`,
    status: ventilationStatus
  });
});

app.get('/api/ventilation', (req, res) => {
  res.json({
    status: ventilationStatus,
    zones: Object.keys(ventilationStatus)
  });
});

app.post('/api/sensor-data', (req, res) => {
  const sensorData = req.body;
  
  // Validate sensor data
  if (!validateSensorData(sensorData)) {
    return res.status(400).json({ error: 'Invalid sensor data format' });
  }
  
  // Store sensor reading
  sensorReadings.push({
    ...sensorData,
    receivedAt: new Date().toISOString()
  });
  
  // Keep only last 1000 readings in memory
  if (sensorReadings.length > 1000) {
    sensorReadings = sensorReadings.slice(-1000);
  }
  
  // Broadcast to connected clients
  io.emit('sensor_update', sensorData);
  
  res.json({ success: true, message: 'Sensor data received' });
});

app.get('/api/alerts', (req, res) => {
  const { limit = 10 } = req.query;
  const recentAlerts = emergencyEvents.slice(-parseInt(limit));
  
  res.json({
    alerts: recentAlerts,
    totalAlerts: emergencyEvents.length,
    activeAlerts: emergencyEvents.filter(alert => alert.status === 'active').length
  });
});

app.post('/api/alerts/:id/acknowledge', (req, res) => {
  const { id } = req.params;
  const alertIndex = emergencyEvents.findIndex(alert => alert.id === id);
  
  if (alertIndex === -1) {
    return res.status(404).json({ error: 'Alert not found' });
  }
  
  emergencyEvents[alertIndex].status = 'acknowledged';
  emergencyEvents[alertIndex].acknowledgedAt = new Date().toISOString();
  
  res.json({
    success: true,
    message: 'Alert acknowledged',
    alert: emergencyEvents[alertIndex]
  });
});

// Utility functions
function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

function validateSensorData(data) {
  const required = ['nodeId', 'timestamp', 'gases', 'environmental', 'location'];
  return required.every(field => data.hasOwnProperty(field));
}

async function logToBlockchain(eventType, data) {
  try {
    const response = await fetch('http://localhost:3002/log-event', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        eventType,
        data,
        timestamp: new Date().toISOString()
      })
    });
    
    if (!response.ok) {
      throw new Error(`Blockchain logging failed: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Blockchain logging error:', error);
    throw error;
  }
}

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Send current system status
  socket.emit('system_status', {
    sensors: sensorReadings.length,
    alerts: emergencyEvents.filter(e => e.status === 'active').length,
    ventilation: ventilationStatus
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
  
  socket.on('request_sensor_data', () => {
    socket.emit('sensor_data', sensorReadings.slice(-50));
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler - FIXED: Changed from '*' to catch-all
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.originalUrl
  });
});

const PORT = process.env.PORT || 3001;

server.listen(PORT, () => {
  console.log(`🚀 Backend API server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
});

module.exports = { app, io };