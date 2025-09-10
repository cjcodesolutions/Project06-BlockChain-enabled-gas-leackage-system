// frontend/src/App.js - Complete Version with ML Predictions and Emergency Response
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import io from 'socket.io-client';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const App = () => {
  const [sensorData, setSensorData] = useState([]);
  const [currentReading, setCurrentReading] = useState(null);
  const [mlPrediction, setMLPrediction] = useState(null);
  const [blockchainTxs, setBlockchainTxs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    sensors: 'online',
    ml: 'online',
    blockchain: 'online'
  });
  const [emergencyActive, setEmergencyActive] = useState(false);
  const [ventilationStatus, setVentilationStatus] = useState({});

  const API_BASE = 'http://localhost:3001';
  const ML_API = 'http://localhost:5000';
  const BLOCKCHAIN_API = 'http://localhost:3002';

  useEffect(() => {
    // Connect to sensor data socket
    const sensorSocket = io('http://localhost:3003');
    
    sensorSocket.on('sensor_data', (data) => {
      setCurrentReading(data);
      setSensorData(prev => [...prev.slice(-50), data]); // Keep last 50 readings
      
      // Check for alerts
      checkThresholds(data);
    });

    sensorSocket.on('connect', () => {
      setSystemStatus(prev => ({ ...prev, sensors: 'online' }));
    });

    sensorSocket.on('disconnect', () => {
      setSystemStatus(prev => ({ ...prev, sensors: 'offline' }));
    });

    return () => {
      sensorSocket.disconnect();
    };
  }, []);

  useEffect(() => {
    // Fetch ML prediction every 30 seconds
    const interval = setInterval(async () => {
      if (currentReading && sensorData.length >= 5) {
        try {
          const response = await axios.post(`${ML_API}/predict`, {
            sensorData: sensorData.slice(-10) // Last 10 readings
          });
          setMLPrediction(response.data);
          setSystemStatus(prev => ({ ...prev, ml: 'online' }));
        } catch (error) {
          console.error('ML prediction failed:', error);
          setSystemStatus(prev => ({ ...prev, ml: 'offline' }));
          // Fallback to simulated ML prediction
          simulateMLPrediction();
        }
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [currentReading, sensorData, ML_API]);

  useEffect(() => {
    // Trigger ML prediction simulation every 10 seconds
    const mlInterval = setInterval(() => {
      if (sensorData.length > 0 && !mlPrediction) {
        simulateMLPrediction();
      }
    }, 10000);

    return () => clearInterval(mlInterval);
  }, [sensorData, mlPrediction]);

  useEffect(() => {
    // Fetch blockchain transactions
    const fetchBlockchainData = async () => {
      try {
        const response = await axios.get(`${BLOCKCHAIN_API}/transactions`);
        setBlockchainTxs(response.data.transactions || []);
        setSystemStatus(prev => ({ ...prev, blockchain: 'online' }));
      } catch (error) {
        console.error('Blockchain fetch failed:', error);
        setSystemStatus(prev => ({ ...prev, blockchain: 'offline' }));
      }
    };

    fetchBlockchainData();
    const interval = setInterval(fetchBlockchainData, 10000);
    return () => clearInterval(interval);
  }, [BLOCKCHAIN_API]);

  const simulateMLPrediction = () => {
    if (sensorData.length > 0) {
      const latestReading = sensorData[sensorData.length - 1];
      const avgGasLevel = (latestReading.gases.methane + latestReading.gases.lpg + 
                          latestReading.gases.carbonMonoxide + latestReading.gases.hydrogenSulfide) / 4;
      
      const probability = Math.min(95, Math.max(5, (avgGasLevel / 300) * 100));
      const risk = probability > 70 ? 'CRITICAL' : probability > 50 ? 'HIGH' : 
                   probability > 30 ? 'MEDIUM' : 'LOW';
      
      setMLPrediction({
        leakProbability: probability,
        confidence: 85 + Math.random() * 10,
        riskLevel: risk,
        timeToThreshold: probability > 70 ? '< 1 hour' : probability > 50 ? '1-2 hours' : '2-4 hours'
      });
    }
  };

  const checkThresholds = (reading) => {
    const thresholds = {
      methane: 400,
      lpg: 300,
      carbonMonoxide: 150,
      hydrogenSulfide: 50
    };

    Object.entries(thresholds).forEach(([gas, threshold]) => {
      const value = reading.gases[gas];
      if (value > threshold) {
        const alert = `HIGH ${gas.toUpperCase()} detected: ${value.toFixed(1)}ppm in ${reading.location.zone}`;
        setAlerts(prev => [alert, ...prev.slice(0, 4)]); // Keep last 5 alerts
        
        // Auto-trigger emergency for critical levels
        if (value > threshold * 1.5) {
          triggerEmergencyProtocol();
        }
        
        // Log emergency event to blockchain
        logEmergencyEvent(gas, value, reading.location.zone);
      }
    });
  };

  const logEmergencyEvent = async (gasType, value, zone) => {
    try {
      await axios.post(`${BLOCKCHAIN_API}/log-emergency`, {
        gasType,
        value,
        zone,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to log emergency event:', error);
    }
  };

  const triggerEmergencyProtocol = async () => {
    if (emergencyActive) return; // Prevent multiple triggers
    
    setEmergencyActive(true);
    
    try {
      // Show immediate response
      setAlerts(prev => [' EMERGENCY PROTOCOL ACTIVATED - All systems shutting down', ...prev]);
      
      // Call backend API
      await axios.post(`${API_BASE}/api/emergency`, {
        action: 'full_shutdown',
        timestamp: new Date().toISOString()
      });
      
      // Show system responses with delays
      setTimeout(() => {
        setAlerts(prev => [' Gas supply lines shut off', ...prev]);
      }, 1000);
      
      setTimeout(() => {
        setAlerts(prev => [' Emergency ventilation activated', ...prev]);
        setVentilationStatus({ all: { active: true, emergency: true } });
      }, 2000);
      
      setTimeout(() => {
        setAlerts(prev => [' Emergency services notified', ...prev]);
      }, 3000);
      
      setTimeout(() => {
        setAlerts(prev => [' Facility lockdown initiated', ...prev]);
      }, 4000);
      
      // Reset emergency state after 30 seconds
      setTimeout(() => {
        setEmergencyActive(false);
      }, 30000);
      
    } catch (error) {
      console.error('Emergency protocol failed:', error);
      setAlerts(prev => [' Emergency protocol error - Manual intervention required', ...prev]);
      setEmergencyActive(false);
    }
  };

  const activateVentilation = async (zone = 'all') => {
    try {
      setAlerts(prev => [` Activating ventilation in ${zone}...`, ...prev]);
      
      await axios.post(`${API_BASE}/api/ventilation`, { zone, action: 'activate' });
      
      setVentilationStatus(prev => ({
        ...prev,
        [zone]: { active: true, manual: true }
      }));
      
      setTimeout(() => {
        setAlerts(prev => [` Ventilation system active in ${zone}`, ...prev]);
      }, 1500);
      
    } catch (error) {
      console.error('Ventilation activation failed:', error);
      setAlerts(prev => [` Ventilation activation failed in ${zone}`, ...prev]);
    }
  };

  const simulateGasLeak = async () => {
    try {
      await axios.post('http://localhost:3003/simulation/leak', {
        zone: 'Zone_A',
        intensity: 'high'
      });
      setAlerts(prev => [' Gas leak simulation activated in Zone A', ...prev]);
    } catch (error) {
      console.error('Simulation failed:', error);
    }
  };

  const resetSystem = async () => {
    try {
      await axios.post('http://localhost:3003/simulation/reset');
      setAlerts([' System reset to normal operation']);
      setEmergencyActive(false);
      setVentilationStatus({});
      setMLPrediction(null);
    } catch (error) {
      console.error('Reset failed:', error);
    }
  };

  const chartData = {
    labels: sensorData.map(d => new Date(d.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Methane (CH4)',
        data: sensorData.map(d => d.gases.methane),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
      },
      {
        label: 'LPG',
        data: sensorData.map(d => d.gases.lpg),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
      },
      {
        label: 'CO',
        data: sensorData.map(d => d.gases.carbonMonoxide),
        borderColor: 'rgb(255, 205, 86)',
        backgroundColor: 'rgba(255, 205, 86, 0.2)',
      },
      {
        label: 'H2S',
        data: sensorData.map(d => d.gases.hydrogenSulfide),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Real-time Gas Concentration',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Concentration (ppm)'
        }
      }
    }
  };

  const getStatusColor = (status) => {
    return status === 'online' ? '#4CAF50' : '#F44336';
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'LOW': return '#4CAF50';
      case 'MEDIUM': return '#FF9800';
      case 'HIGH': return '#FF5722';
      case 'CRITICAL': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>GasGuard - Smart IoT Gas Detection System</h1>
        <div className="status-bar">
          <div className="status-item">
            <span style={{ color: getStatusColor(systemStatus.sensors) }}>●</span>
            Sensors: {systemStatus.sensors}
          </div>
          <div className="status-item">
            <span style={{ color: getStatusColor(systemStatus.ml) }}>●</span>
            ML Engine: {systemStatus.ml}
          </div>
          <div className="status-item">
            <span style={{ color: getStatusColor(systemStatus.blockchain) }}>●</span>
            Blockchain: {systemStatus.blockchain}
          </div>
          {emergencyActive && (
            <div className="status-item" style={{ color: '#F44336', fontWeight: 'bold' }}>
               EMERGENCY ACTIVE
            </div>
          )}
        </div>
      </header>

      <main className="main-content">
        {/* Emergency Banner */}
        {emergencyActive && (
          <div style={{
            background: 'linear-gradient(45deg, #F44336, #d32f2f)',
            color: 'white',
            padding: '15px',
            textAlign: 'center',
            fontSize: '1.2em',
            fontWeight: 'bold',
            marginBottom: '20px',
            borderRadius: '8px',
            animation: 'pulse 2s infinite'
          }}>
             EMERGENCY PROTOCOL ACTIVE - FACILITY LOCKED DOWN 🚨
          </div>
        )}

        {/* Alerts Panel */}
        {alerts.length > 0 && (
          <div className="alerts-panel">
            <h3> System Alerts</h3>
            {alerts.map((alert, index) => (
              <div key={index} className="alert-item">
                {alert}
              </div>
            ))}
          </div>
        )}

        {/* Current Sensor Readings */}
        {currentReading && (
          <div className="sensor-grid">
            <div className="sensor-card">
              <h3>Methane (CH4)</h3>
              <div className="sensor-value" style={{
                color: currentReading.gases.methane > 400 ? '#F44336' : '#4CAF50'
              }}>
                {currentReading.gases.methane.toFixed(1)} ppm
              </div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
            <div className="sensor-card">
              <h3>LPG</h3>
              <div className="sensor-value" style={{
                color: currentReading.gases.lpg > 300 ? '#F44336' : '#4CAF50'
              }}>
                {currentReading.gases.lpg.toFixed(1)} ppm
              </div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
            <div className="sensor-card">
              <h3>Carbon Monoxide</h3>
              <div className="sensor-value" style={{
                color: currentReading.gases.carbonMonoxide > 150 ? '#F44336' : '#4CAF50'
              }}>
                {currentReading.gases.carbonMonoxide.toFixed(1)} ppm
              </div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
            <div className="sensor-card">
              <h3>Hydrogen Sulfide</h3>
              <div className="sensor-value" style={{
                color: currentReading.gases.hydrogenSulfide > 50 ? '#F44336' : '#4CAF50'
              }}>
                {currentReading.gases.hydrogenSulfide.toFixed(1)} ppm
              </div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
          </div>
        )}

        {/* Chart */}
        <div className="chart-container">
          <Line data={chartData} options={chartOptions} />
        </div>

        {/* Enhanced ML Prediction */}
        {mlPrediction && (
          <div className="ml-prediction">
            <h3> AI Prediction Engine</h3>
            <div className="prediction-grid">
              <div className="prediction-item">
                <span>Leak Probability:</span>
                <span 
                  className="prediction-value" 
                  style={{ 
                    color: mlPrediction.leakProbability > 70 ? '#F44336' : 
                           mlPrediction.leakProbability > 50 ? '#FF9800' : '#4CAF50',
                    fontSize: '1.3em',
                    fontWeight: 'bold'
                  }}
                >
                  {mlPrediction.leakProbability.toFixed(1)}%
                </span>
              </div>
              <div className="prediction-item">
                <span>Time to Critical:</span>
                <span className="prediction-value">{mlPrediction.timeToThreshold}</span>
              </div>
              <div className="prediction-item">
                <span>AI Confidence:</span>
                <span className="prediction-value">{mlPrediction.confidence.toFixed(1)}%</span>
              </div>
              <div className="prediction-item">
                <span>Risk Assessment:</span>
                <span 
                  className="prediction-value" 
                  style={{ 
                    color: getRiskColor(mlPrediction.riskLevel),
                    fontSize: '1.2em',
                    fontWeight: 'bold'
                  }}
                >
                  {mlPrediction.riskLevel}
                </span>
              </div>
            </div>
            
            {/* Risk Level Actions */}
            {mlPrediction.leakProbability > 60 && (
              <div style={{ 
                marginTop: '15px', 
                padding: '10px', 
                backgroundColor: 'rgba(244, 67, 54, 0.2)',
                borderRadius: '5px',
                border: '1px solid #F44336'
              }}>
                <strong> HIGH RISK DETECTED</strong>
                <p>Recommended Actions: Increase monitoring, prepare emergency response</p>
              </div>
            )}
          </div>
        )}

        {/* Enhanced Control Panel */}
        <div className="control-panel">
          <h3>Emergency Controls</h3>
          <div className="button-group">
            <button 
              className="btn emergency" 
              onClick={triggerEmergencyProtocol}
              disabled={emergencyActive}
            >
               Emergency Stop
            </button>
            <button 
              className="btn warning" 
              onClick={() => activateVentilation('all')}
            >
               Activate Ventilation
            </button>
            <button 
              className="btn normal" 
              onClick={() => {
                simulateMLPrediction();
                setAlerts(prev => [' ML prediction refreshed', ...prev]);
              }}
            >
               Refresh AI Prediction
            </button>
            <button 
              className="btn warning" 
              onClick={simulateGasLeak}
            >
               Simulate Gas Leak
            </button>
            <button 
              className="btn normal" 
              onClick={resetSystem}
            >
               Reset System
            </button>
            <button 
              className="btn normal" 
              onClick={() => setAlerts([])}
            >
               Clear Alerts
            </button>
          </div>
        </div>

        {/* Ventilation Status */}
        {Object.keys(ventilationStatus).length > 0 && (
          <div className="ml-prediction">
            <h3> Ventilation Status</h3>
            <div className="prediction-grid">
              {Object.entries(ventilationStatus).map(([zone, status]) => (
                <div key={zone} className="prediction-item">
                  <span>{zone}:</span>
                  <span 
                    className="prediction-value" 
                    style={{ color: status.active ? '#4CAF50' : '#F44336' }}
                  >
                    {status.active ? 'ACTIVE' : 'INACTIVE'}
                    {status.emergency && ' (EMERGENCY)'}
                    {status.manual && ' (MANUAL)'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Blockchain Transactions */}
        <div className="blockchain-panel">
          <h3> Blockchain Transactions</h3>
          <div className="transaction-log">
            {blockchainTxs.slice(0, 5).map((tx, index) => (
              <div key={index} className="transaction-item">
                <div className="tx-header">
                  <span>Block #{tx.blockNumber}</span>
                  <span>{new Date(tx.timestamp).toLocaleString()}</span>
                </div>
                <div className="tx-details">
                  <span>Type: {tx.eventType}</span>
                  <span>Hash: {tx.transactionHash ? tx.transactionHash.substring(0, 16) + '...' : 'N/A'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;