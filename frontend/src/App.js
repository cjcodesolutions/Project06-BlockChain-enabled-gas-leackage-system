// frontend/src/App.js
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
      if (currentReading) {
        try {
          const response = await axios.post(`${ML_API}/predict`, {
            sensorData: sensorData.slice(-10) // Last 10 readings
          });
          setMLPrediction(response.data);
          setSystemStatus(prev => ({ ...prev, ml: 'online' }));
        } catch (error) {
          console.error('ML prediction failed:', error);
          setSystemStatus(prev => ({ ...prev, ml: 'offline' }));
        }
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [currentReading, sensorData, ML_API]);

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
    try {
      await axios.post(`${API_BASE}/api/emergency`, {
        action: 'full_shutdown',
        timestamp: new Date().toISOString()
      });
      setAlerts(prev => ['Emergency protocol activated', ...prev]);
    } catch (error) {
      console.error('Emergency protocol failed:', error);
    }
  };

  const activateVentilation = async (zone = 'all') => {
    try {
      await axios.post(`${API_BASE}/api/ventilation`, { zone, action: 'activate' });
      setAlerts(prev => [`Ventilation activated in ${zone}`, ...prev]);
    } catch (error) {
      console.error('Ventilation activation failed:', error);
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
        </div>
      </header>

      <main className="main-content">
        {/* Alerts Panel */}
        {alerts.length > 0 && (
          <div className="alerts-panel">
            <h3>🚨 Active Alerts</h3>
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
              <div className="sensor-value">{currentReading.gases.methane.toFixed(1)} ppm</div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
            <div className="sensor-card">
              <h3>LPG</h3>
              <div className="sensor-value">{currentReading.gases.lpg.toFixed(1)} ppm</div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
            <div className="sensor-card">
              <h3>Carbon Monoxide</h3>
              <div className="sensor-value">{currentReading.gases.carbonMonoxide.toFixed(1)} ppm</div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
            <div className="sensor-card">
              <h3>Hydrogen Sulfide</h3>
              <div className="sensor-value">{currentReading.gases.hydrogenSulfide.toFixed(1)} ppm</div>
              <div className="sensor-location">Zone: {currentReading.location.zone}</div>
            </div>
          </div>
        )}

        {/* Chart */}
        <div className="chart-container">
          <Line data={chartData} options={chartOptions} />
        </div>

        {/* ML Prediction */}
        {mlPrediction && (
          <div className="ml-prediction">
            <h3>🤖 ML Prediction Engine</h3>
            <div className="prediction-grid">
              <div className="prediction-item">
                <span>Leak Probability:</span>
                <span className="prediction-value">{mlPrediction.leakProbability.toFixed(1)}%</span>
              </div>
              <div className="prediction-item">
                <span>Time to Threshold:</span>
                <span className="prediction-value">{mlPrediction.timeToThreshold}</span>
              </div>
              <div className="prediction-item">
                <span>Confidence:</span>
                <span className="prediction-value">{mlPrediction.confidence.toFixed(1)}%</span>
              </div>
              <div className="prediction-item">
                <span>Risk Level:</span>
                <span 
                  className="prediction-value" 
                  style={{ color: getRiskColor(mlPrediction.riskLevel) }}
                >
                  {mlPrediction.riskLevel}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Control Panel */}
        <div className="control-panel">
          <h3>Emergency Controls</h3>
          <div className="button-group">
            <button 
              className="btn emergency" 
              onClick={triggerEmergencyProtocol}
            >
              🚨 Emergency Stop
            </button>
            <button 
              className="btn warning" 
              onClick={() => activateVentilation('all')}
            >
              💨 Activate Ventilation
            </button>
            <button 
              className="btn normal" 
              onClick={() => setAlerts([])}
            >
              ✅ Clear Alerts
            </button>
          </div>
        </div>

        {/* Blockchain Transactions */}
        <div className="blockchain-panel">
          <h3>⛓️ Blockchain Transactions</h3>
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