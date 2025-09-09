// blockchain-service/server.js
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// In-memory blockchain simulation (fallback when Ganache not available)
let blockchainSimulation = {
  blocks: [],
  transactions: [],
  currentBlock: 0
};

class BlockchainService {
  constructor() {
    this.isConnected = false;
    this.contractAddress = null;
    this.initializeBlockchain();
  }

  async initializeBlockchain() {
    try {
      // Try to connect to Ganache (optional)
      console.log('Attempting to connect to Ganache...');
      // For now, we'll use simulation mode
      this.isConnected = false;
      this.initializeSimulation();
    } catch (error) {
      console.log('Ganache not available, using simulation mode');
      this.isConnected = false;
      this.initializeSimulation();
    }
  }

  initializeSimulation() {
    // Initialize with some sample transactions
    this.addSimulatedTransaction('system_start', { action: 'initialization' });
    this.addSimulatedTransaction('sensor_calibration', { sensors: 24 });
    console.log('Blockchain simulation initialized');
  }

  async logEvent(eventType, data) {
    const timestamp = Math.floor(Date.now() / 1000);
    
    if (this.isConnected) {
      // Real blockchain transaction (when Ganache is available)
      try {
        // This would be the Web3 transaction code
        return {
          success: true,
          transactionHash: this.generateHash(),
          blockNumber: ++blockchainSimulation.currentBlock,
          gasUsed: 21000
        };
      } catch (error) {
        console.error('Blockchain transaction failed:', error);
        return this.logEventSimulated(eventType, data);
      }
    } else {
      return this.logEventSimulated(eventType, data);
    }
  }

  logEventSimulated(eventType, data) {
    const transaction = this.addSimulatedTransaction(eventType, data);
    return {
      success: true,
      transactionHash: transaction.hash,
      blockNumber: transaction.blockNumber,
      gasUsed: 21000,
      simulated: true
    };
  }

  addSimulatedTransaction(eventType, data) {
    const transaction = {
      hash: this.generateHash(),
      eventType,
      data,
      timestamp: new Date().toISOString(),
      blockNumber: ++blockchainSimulation.currentBlock,
      gasUsed: 21000,
      from: '0x1234567890123456789012345678901234567890'
    };

    blockchainSimulation.transactions.push(transaction);
    
    // Create block every 5 transactions
    if (blockchainSimulation.transactions.length % 5 === 0) {
      this.createSimulatedBlock();
    }

    return transaction;
  }

  createSimulatedBlock() {
    const blockTransactions = blockchainSimulation.transactions.slice(-5);
    const block = {
      number: blockchainSimulation.currentBlock,
      hash: this.generateHash(),
      parentHash: blockchainSimulation.blocks.length > 0 
        ? blockchainSimulation.blocks[blockchainSimulation.blocks.length - 1].hash 
        : '0x0000000000000000000000000000000000000000000000000000000000000000',
      timestamp: new Date().toISOString(),
      transactions: blockTransactions.map(tx => tx.hash),
      gasUsed: blockTransactions.reduce((sum, tx) => sum + tx.gasUsed, 0),
      gasLimit: 8000000
    };

    blockchainSimulation.blocks.push(block);
  }

  generateHash() {
    return '0x' + Math.random().toString(16).substr(2, 64);
  }

  async getTransactions(limit = 10) {
    return this.getSimulatedTransactions(limit);
  }

  getSimulatedTransactions(limit = 10) {
    return blockchainSimulation.transactions
      .slice(-limit)
      .map(tx => ({
        eventType: tx.eventType,
        data: tx.data,
        timestamp: tx.timestamp,
        transactionHash: tx.hash,
        blockNumber: tx.blockNumber,
        gasUsed: tx.gasUsed,
        simulated: true
      }));
  }

  getNetworkStatus() {
    return {
      connected: this.isConnected,
      networkId: this.isConnected ? 5777 : 'simulation',
      blockNumber: blockchainSimulation.currentBlock,
      accounts: this.isConnected ? 10 : 1,
      contractAddress: this.contractAddress || '0x1234567890123456789012345678901234567890',
      mode: this.isConnected ? 'ganache' : 'simulation'
    };
  }
}

// Initialize blockchain service
const blockchainService = new BlockchainService();

// Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'online',
    service: 'Blockchain Logger',
    network: blockchainService.getNetworkStatus(),
    timestamp: new Date().toISOString()
  });
});

app.post('/log-event', async (req, res) => {
  try {
    const { eventType, data, timestamp } = req.body;
    
    if (!eventType || !data) {
      return res.status(400).json({ error: 'eventType and data are required' });
    }
    
    const result = await blockchainService.logEvent(eventType, data);
    
    res.json({
      success: true,
      transaction: result,
      timestamp: timestamp || new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Log event error:', error);
    res.status(500).json({
      error: 'Failed to log event',
      message: error.message
    });
  }
});

app.post('/log-emergency', async (req, res) => {
  try {
    const { gasType, value, zone, timestamp } = req.body;
    
    const emergencyData = {
      type: 'emergency_alert',
      gasType,
      concentration: value,
      zone,
      severity: value > 500 ? 'critical' : 'high',
      timestamp: timestamp || new Date().toISOString()
    };
    
    const result = await blockchainService.logEvent('emergency_event', emergencyData);
    
    res.json({
      success: true,
      emergency: emergencyData,
      transaction: result
    });
    
  } catch (error) {
    console.error('Log emergency error:', error);
    res.status(500).json({
      error: 'Failed to log emergency event',
      message: error.message
    });
  }
});

app.get('/transactions', async (req, res) => {
  try {
    const { limit = 10 } = req.query;
    const transactions = await blockchainService.getTransactions(parseInt(limit));
    
    res.json({
      transactions,
      count: transactions.length,
      network: blockchainService.getNetworkStatus()
    });
    
  } catch (error) {
    console.error('Get transactions error:', error);
    res.status(500).json({
      error: 'Failed to fetch transactions',
      message: error.message
    });
  }
});

app.get('/network-status', (req, res) => {
  res.json(blockchainService.getNetworkStatus());
});

app.get('/contract-info', (req, res) => {
  res.json({
    address: blockchainService.contractAddress || '0x1234567890123456789012345678901234567890',
    functions: ['logEvent', 'getEventCount', 'getEvent'],
    mode: 'simulation'
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

const PORT = process.env.PORT || 3002;

app.listen(PORT, () => {
  console.log(`⛓️  Blockchain service running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🔗 Network status: http://localhost:${PORT}/network-status`);
});

module.exports = { app, blockchainService };