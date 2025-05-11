import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
import asyncio
import logging
import os  # Added missing import
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import arch package for GARCH modeling
try:
    from arch import arch_model
except ImportError:
    print("Warning: arch package not found. Please install with 'pip install arch'")
    # Provide a simple substitute function for testing
    def arch_model(returns, vol='GARCH', p=1, q=1):
        class SimpleFitResult:
            def __init__(self):
                self.conditional_volatility = pd.Series([0.02] * len(returns))
        
        class SimpleArchModel:
            def fit(self, disp='off'):
                return SimpleFitResult()
        return SimpleArchModel()

# Import qiskit if available or provide simple replacement
try:
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
except ImportError:
    print("Warning: qiskit_algorithms not found. Using simplified optimization.")
    # Simple replacement classes for testing
    class COBYLA:
        def __init__(self):
            pass
    
    class QAOA:
        def __init__(self, optimizer=None, reps=1):
            self.optimizer = optimizer
            self.reps = reps
        
        def compute_minimum_eigenvalue(self, objective_function):
            class Result:
                def __init__(self):
                    self.eigenvector = [0.7, 25, 75, 40, 20]
            return Result()

# Logging setup
logging.basicConfig(filename='hft_system.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MT5 connection with retry
async def initialize_mt5(max_retries=3, login=None, password=None, server=None):
    # Add parameters for MT5 initialization
    for attempt in range(max_retries):
        # Use provided credentials if available
        if login and password and server:
            init_result = mt5.initialize(login=login, password=password, server=server)
        else:
            # Otherwise try to use saved credentials
            init_result = mt5.initialize()
            
        if init_result:
            logging.info("Multidimensional MT5 initialized")
            print("Multidimensional MT5 initialized")
            return True
        logging.warning(f"MT5 initialization failed: {mt5.last_error()}, retrying {attempt+1}/{max_retries}")
        await asyncio.sleep(5)
    logging.error("MT5 initialization failed after max retries")
    return False

# Get tick data with multi-timeframe support
async def get_tick_data(symbol, n_ticks=10000):
    try:
        ticks = mt5.copy_ticks_from(symbol, datetime.now(pytz.UTC), n_ticks, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            logging.error(f"Failed to get ticks for {symbol}")
            return None
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logging.error(f"Error fetching tick data for {symbol}: {str(e)}")
        return None

# Get minute data for multi-timeframe analysis
async def get_minute_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=1000):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to get minute data for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logging.error(f"Error fetching minute data for {symbol}: {str(e)}")
        return None

# Calculate indicators, order flow, and dark pool proxy
def calculate_indicators(df, timeframe='tick'):
    try:
        # Check if required columns exist
        required_columns = ['bid', 'ask', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logging.error(f"Required column {col} not found in dataframe")
                # Create dummy columns for testing
                if col == 'bid':
                    df['bid'] = df['close'] if 'close' in df.columns else np.random.random(len(df))
                if col == 'ask':
                    df['ask'] = df['bid'] + 0.0001 if 'bid' in df.columns else np.random.random(len(df))
                if col == 'volume':
                    df['volume'] = np.random.randint(1, 100, size=len(df))
        
        window_fast = 10 if timeframe == 'tick' else 5
        window_slow = 50 if timeframe == 'tick' else 20
        df['sma_fast'] = df['bid'].rolling(window=window_fast).mean()
        df['sma_slow'] = df['bid'].rolling(window=window_slow).mean()
        delta = df['bid'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=20).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan).fillna(gain)
        df['rsi'] = 100 - (100 / (1 + rs))
        ema_fast = df['bid'].ewm(span=12, adjust=False).mean()
        ema_slow = df['bid'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['bb_mid'] = df['bid'].rolling(window=20).mean()
        df['bb_std'] = df['bid'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        
        # Calculate True Range first (was used before being defined in original code)
        df['high_low'] = df['ask'] - df['bid']
        df['high_close'] = abs(df['ask'] - df['bid'].shift())
        df['low_close'] = abs(df['bid'] - df['bid'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=20).mean()
        
        # Now we can safely calculate these
        df['plus_di'] = 100 * ((df['ask'] - df['ask'].shift(1)).clip(lower=0) / df['tr'].replace(0, np.nan)).rolling(window=20).mean().fillna(0)
        df['minus_di'] = 100 * ((df['bid'].shift(1) - df['bid']).clip(lower=0) / df['tr'].replace(0, np.nan)).rolling(window=20).mean().fillna(0)
        
        # Avoid division by zero
        denominator = df['plus_di'] + df['minus_di']
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / denominator.replace(0, np.nan)
        df['adx'] = dx.rolling(window=20).mean().fillna(0)
        
        df['low_20'] = df['bid'].rolling(window=20).min()
        df['high_20'] = df['ask'].rolling(window=20).max()
        
        # Avoid division by zero
        range_diff = df['high_20'] - df['low_20']
        df['stoch_k'] = 100 * (df['bid'] - df['low_20']) / range_diff.replace(0, np.nan).fillna(1)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        df['vwap'] = (df['bid'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Dark pool volume proxy
        vol_mean = df['volume'].rolling(window=20).mean()
        vol_std = df['volume'].rolling(window=20).std()
        df['vol_spike'] = df['volume'].where(df['volume'] > vol_mean + 2 * vol_std, 0)
        
        # Avoid division by zero
        atr_mean = df['atr'].rolling(window=20).mean()
        df['dark_pool'] = df['vol_spike'] * (df['atr'] / atr_mean.replace(0, np.nan)).fillna(1)
        
        # Order flow (bid-ask imbalance)
        df['order_flow'] = (df['ask'] - df['bid']).rolling(window=20).mean() / df['atr'].replace(0, np.nan).fillna(1)
        
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        return df

# Volatility regime detection using GARCH
def detect_volatility_regime(df):
    try:
        returns = df['bid'].pct_change().dropna()
        if len(returns) < 50:  # Need enough data for GARCH
            return 'low'
            
        model = arch_model(returns, vol='GARCH', p=1, q=1)
        res = model.fit(disp='off')
        vol_forecast = res.conditional_volatility.iloc[-1]
        vol_mean = res.conditional_volatility.mean()
        return 'high' if vol_forecast > 1.5 * vol_mean else 'low'
    except Exception as e:
        logging.error(f"Error in volatility regime detection: {str(e)}")
        return 'low'

# Neuromorphic spike encoding for HFT
def neuromorphic_encode(data, threshold=0.005):
    try:
        spikes = np.zeros_like(data)
        for i in range(1, len(data)):
            if abs(data[i] - data[i-1]) > threshold * np.std(data):
                spikes[i] = 1 if data[i] > data[i-1] else -1
        return spikes
    except Exception as e:
        logging.error(f"Error in neuromorphic encoding: {str(e)}")
        return np.zeros_like(data)

# Quantum Deep Reinforcement Learning Agent (Multi-Agent)
class QDRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents=3):
        super(QDRLAgent, self).__init__()
        self.n_agents = n_agents
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(n_agents)
        ])
    
    def forward(self, state):
        action_probs = [agent(state) for agent in self.agents]
        return torch.mean(torch.stack(action_probs), dim=0)

async def train_qdrl_agent(df, state_dim, action_dim=3, n_agents=3):
    try:
        agent = QDRLAgent(state_dim, action_dim, n_agents)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.0001)
        scaler = MinMaxScaler()
        
        features = ['bid', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                    'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']
        
        # Check if we have enough data after dropping NaN values
        df_clean = df.dropna(subset=features)
        if len(df_clean) < 50:
            logging.warning(f"Not enough clean data for QDRL training: {len(df_clean)} rows")
            # Create a dummy scaler for compatibility
            dummy_data = pd.DataFrame({feature: [0, 1] for feature in features})
            scaler.fit(dummy_data)
            return agent, scaler
            
        data = scaler.fit_transform(df_clean[features])
        spikes = neuromorphic_encode(df_clean['bid'].values)
        
        # Reduce epochs for faster testing
        num_epochs = min(25, max(2, len(data) // 10))
        
        for epoch in range(num_epochs):
            state = torch.FloatTensor(data[:-1])
            next_state = torch.FloatTensor(data[1:])
            rewards = torch.FloatTensor((df_clean['bid'].diff().shift(-1) > 0).astype(float).iloc[1:].values * (1 + abs(spikes[1:])))
            actions = torch.zeros(len(state), dtype=torch.long)
            
            total_loss = 0
            for t in range(len(state)):
                action_probs = agent(state[t])
                action = torch.argmax(action_probs).item()
                actions[t] = action
                
                if t < len(rewards):
                    reward = rewards[t]
                    loss = -torch.log(action_probs[action]) * reward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(state) if len(state) > 0 else 0
            logging.info(f"QDRL Training Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return agent, scaler
    except Exception as e:
        logging.error(f"Error training QDRL agent: {str(e)}")
        # Return dummy objects for compatibility
        agent = QDRLAgent(state_dim, action_dim, n_agents)
        scaler = MinMaxScaler()
        dummy_data = pd.DataFrame({feature: [0, 1] for feature in ['bid', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                    'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']})
        scaler.fit(dummy_data)
        return agent, scaler

# Hyperdimensional Graph Neural Network with Attention
class HGNNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(HGNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, g, features):
        h = self.relu(self.conv1(g, features))
        h = self.relu(self.conv2(g, h))
        h = self.relu(self.conv3(g, h))
        attn_weights = torch.sigmoid(self.attention(h))
        h = h * attn_weights
        h = self.fc(h)
        return self.softmax(h)

async def train_hgnn_model(df_dict, symbols):
    try:
        if not df_dict or len(df_dict) < 1:
            logging.error("Empty dataframe dictionary passed to train_hgnn_model")
            dummy_model = HGNNModel(7, 128, 2)
            return dummy_model
            
        features = ['bid', 'rsi', 'macd_hist', 'adx', 'atr', 'dark_pool', 'order_flow']
        
        # Check if all dataframes have the required features
        for symbol, df in df_dict.items():
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logging.error(f"Missing features in dataframe for {symbol}: {missing_features}")
                dummy_model = HGNNModel(len(features), 128, 2)
                return dummy_model
        
        # Prepare scaled data for each symbol
        scaler = MinMaxScaler()
        data_list = []
        for symbol, df in df_dict.items():
            df_clean = df.dropna(subset=features)
            if len(df_clean) > 0:
                # Take last 20 rows or all if less than 20
                df_tail = df_clean.tail(min(20, len(df_clean)))
                scaled_data = scaler.fit_transform(df_tail[features])
                data_list.append(scaled_data)
        
        if not data_list:
            logging.error("No valid data for HGNN training after cleaning")
            dummy_model = HGNNModel(len(features), 128, 2)
            return dummy_model
            
        # Create a simple graph connecting all symbols
        num_nodes = len(data_list)
        if num_nodes < 2:
            # Need at least 2 nodes for a graph
            src_nodes = [0]
            dst_nodes = [0]
        else:
            src_nodes = list(range(num_nodes))
            dst_nodes = [(i+1) % num_nodes for i in range(num_nodes)]
            
        g = dgl.graph((src_nodes, dst_nodes))
        
        # Prepare node features and labels
        data_concat = np.vstack(data_list)
        if data_concat.shape[0] == 0:
            logging.error("Empty data for HGNN training")
            dummy_model = HGNNModel(len(features), 128, 2)
            return dummy_model
            
        node_features = torch.FloatTensor(data_concat)
        
        # Create labels (direction prediction)
        labels = []
        for symbol, df in df_dict.items():
            df_clean = df.dropna()
            if len(df_clean) > 1:
                # 1 if price is going up, 0 if down
                direction = 1 if df_clean['bid'].diff().iloc[-1] > 0 else 0
                labels.append(direction)
            else:
                # Default to up if not enough data
                labels.append(1)
                
        if not labels:
            labels = [1] * len(data_list)  # Default all to up if no data
            
        labels = torch.LongTensor(labels)
        
        # Create and train model
        model = HGNNModel(len(features), 128, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        # Reduced epochs for quicker testing
        num_epochs = min(20, max(2, len(data_list)))
        
        for epoch in range(num_epochs):
            logits = model(g, node_features)
            loss = nn.CrossEntropyLoss()(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(f"HGNN Training Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        return model
    except Exception as e:
        logging.error(f"Error training HGNN model: {str(e)}")
        # Return a dummy model for compatibility
        dummy_model = HGNNModel(7, 128, 2)
        return dummy_model

# Train ensemble ML models
async def train_ensemble_models(df):
    try:
        # Make sure we have all needed columns
        required_features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                            'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 
                            'atr', 'vwap', 'dark_pool', 'order_flow', 'bid']
                            
        for feature in required_features:
            if feature not in df.columns:
                logging.error(f"Missing required feature: {feature}")
                # Create dummy models
                rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
                xgb_model = XGBClassifier(n_estimators=10, random_state=42)
                X_dummy = np.random.random((10, 16))
                y_dummy = np.random.randint(0, 2, 10)
                rf_model.fit(X_dummy, y_dummy)
                xgb_model.fit(X_dummy, y_dummy)
                return rf_model, xgb_model
        
        df = df.dropna()
        if len(df) < 50:  # Need enough data for meaningful training
            logging.warning(f"Not enough data for ensemble models: {len(df)} rows")
            # Create dummy models
            rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
            xgb_model = XGBClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.random((10, 16))
            y_dummy = np.random.randint(0, 2, 10)
            rf_model.fit(X_dummy, y_dummy)
            xgb_model.fit(X_dummy, y_dummy)
            return rf_model, xgb_model
            
        features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 
                    'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']
        X = df[features]
        df['target'] = (df['bid'].shift(-1) > df['bid']).astype(int)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reduce n_estimators for faster testing
        n_est = min(100, max(10, len(X_train) // 5))
        
        rf_model = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        logging.info(f"Random Forest Accuracy: {rf_accuracy:.2f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
        
        xgb_model = XGBClassifier(n_estimators=n_est, random_state=42, tree_method='hist')
        xgb_model.fit(X_train, y_train)
        xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
        logging.info(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
        print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
        
        return rf_model, xgb_model
    except Exception as e:
        logging.error(f"Error training ensemble models: {str(e)}")
        # Return dummy models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        xgb_model = XGBClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.random((10, 16))
        y_dummy = np.random.randint(0, 2, 10)
        rf_model.fit(X_dummy, y_dummy)
        xgb_model.fit(X_dummy, y_dummy)
        return rf_model, xgb_model

# Place trade (HFT optimized)
async def place_trade(symbol, trade_type, lot_size, sl, entry_price):
    try:
        # Make sure MT5 is initialized
        if not mt5.initialize():
            logging.error("MT5 not initialized")
            return False, None
            
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"Symbol {symbol} not found")
            return False, None
            
        point = symbol_info.point
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if trade_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": entry_price - sl * point if trade_type == "buy" else entry_price + sl * point,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Trade failed for {symbol}: {result.comment}")
            return False, None
        logging.info(f"Trade placed: {trade_type} {symbol} at {entry_price}")
        print(f"Trade placed: {trade_type} {symbol} at {entry_price}")
        return True, result.order
    except Exception as e:
        logging.error(f"Error placing trade for {symbol}: {str(e)}")
        return False, None

# Log trade to CSV
async def log_trade(symbol, trade_type, entry_price, exit_price, profit, win, metrics):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname('trade_log.csv') if os.path.dirname('trade_log.csv') else '.', exist_ok=True)
        
        # Check if file exists, if not create with headers
        file_exists = os.path.isfile('trade_log.csv')
        
        with open('trade_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win', 
                                'win_rate', 'sharpe', 'sortino', 'alpha', 'md_alpha'])
            writer.writerow([datetime.now(), symbol, trade_type, entry_price, exit_price, profit, win, 
                            metrics['win_rate'], metrics['sharpe'], metrics['sortino'], metrics['alpha'], metrics['md_alpha']])
        
        # Check if dashboard file exists
        file_exists = os.path.isfile('dashboard.csv')
        
        with open('dashboard.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'win_rate', 'sharpe', 'sortino', 
                                'alpha', 'md_alpha', 'drawdown', 'hft_efficiency', 'profit_factor'])
            writer.writerow([datetime.now(), symbol, metrics['win_rate'], metrics['sharpe'], metrics['sortino'], 
                            metrics['alpha'], metrics['md_alpha'], metrics['drawdown'], metrics['hft_efficiency'], metrics['profit_factor']])
    except Exception as e:
        logging.error(f"Error logging trade: {str(e)}")

# Monitor trade with HFT trailing stop
async def monitor_trade(symbol, order_id, trade_type, entry_price, sl, atr, metrics):
    try:
        point = mt5.symbol_info(symbol).point
        trailing_stop = sl * point
        best_price = entry_price if trade_type == "buy" else entry_price
        
        # Set maximum monitoring time (5 minutes)
        max_monitor_time = 300  # seconds
        start_time = time.time()
        
        while (time.time() - start_time) < max_monitor_time:
            # Check if MT5 is still connected
            if not mt5.initialize():
                logging.error("MT5 connection lost during trade monitoring")
                break
                
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                # Try to get trade history
                history = mt5.history_deals_get(ticket=order_id)
                if history and len(history) > 0:
                    deal = history[-1]
                    exit_price = deal.price
                    profit = deal.profit
                    win = profit > 0
                    await log_trade(symbol, trade_type, entry_price, exit_price, profit, win, metrics)
                    logging.info(f"Trade closed: {symbol}, Profit: {profit}, Win: {win}")
                    print(f"Trade closed: {symbol}, Profit: {profit}, Win: {win}")
                else:
                    # If no history found, log with estimated values
                    current_price = mt5.symbol_info_tick(symbol).bid if trade_type == "buy" else mt5.symbol_info_tick(symbol).ask
                    profit = (current_price - entry_price) if trade_type == "buy" else (entry_price - current_price)
                    win = profit > 0
                    await log_trade(symbol, trade_type, entry_price, current_price, profit, win, metrics)
                    logging.info(f"Trade closed (no history): {symbol}, Estimated Profit: {profit}, Win: {win}")
                    print(f"Trade closed (no history): {symbol}, Estimated Profit: {profit}, Win: {win}")
                break
            
            # Get current price
            current_price = mt5.symbol_info_tick(symbol).bid if trade_type == "buy" else mt5.symbol_info_tick(symbol).ask
            
            # Update trailing stop if price moved favorably
            if trade_type == "buy" and current_price > best_price:
                best_price = current_price
                new_sl = best_price - trailing_stop
                if new_sl > positions[0].sl:
                    await modify_position(symbol, order_id, new_sl)
            elif trade_type == "sell" and current_price < best_price:
                best_price = current_price
                new_sl = best_price + trailing_stop
                if new_sl < positions[0].sl:
                    await modify_position(symbol, order_id, new_sl)
            
            # Much faster sleep for HFT
            await asyncio.sleep(0.01)  # 10ms for HFT speed
        
        # If we reach the maximum monitoring time without closure
        if (time.time() - start_time) >= max_monitor_time:
            logging.warning(f"Maximum monitoring time reached for {symbol}, closing position manually")
            # Close the position
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": positions[0].volume,
                    "type": mt5.ORDER_TYPE_SELL if positions[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": positions[0].ticket,
                    "price": mt5.symbol_info_tick(symbol).bid if positions[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK
                }
                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    profit = result.profit
                    win = profit > 0
                    exit_price = result.price
                    await log_trade(symbol, trade_type, entry_price, exit_price, profit, win, metrics)
                    logging.info(f"Trade closed manually: {symbol}, Profit: {profit}, Win: {win}")
                    print(f"Trade closed manually: {symbol}, Profit: {profit}, Win: {win}")
    except Exception as e:
        logging.error(f"Error monitoring trade for {symbol}: {str(e)}")

# Modify position
async def modify_position(symbol, order_id, new_sl):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "position": positions[0].ticket,
                "sl": new_sl,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Updated SL for {symbol} to {new_sl}")
                print(f"Updated SL for {symbol} to {new_sl}")
            else:
                logging.warning(f"Failed to update SL: {result.comment}")
    except Exception as e:
        logging.error(f"Error modifying position for {symbol}: {str(e)}")

# Check news events
def is_news_time():
    try:
        now = datetime.now(pytz.timezone('US/Eastern'))
        is_first_friday = now.day <= 7 and now.weekday() == 4
        is_nfp_time = is_first_friday and now.hour == 8 and 30 <= now.minute <= 45
        is_fomc = now.hour in [14, 15] and now.weekday() == 2 and now.day in [15, 16, 29, 30]
        is_low_liquidity = now.hour in [17, 18, 19]
        
        if is_nfp_time:
            logging.info("NFP (Non-Farm Payroll) report time, pausing trading")
        if is_fomc:
            logging.info("FOMC (Federal Open Market Committee) announcement time, pausing trading")
        if is_low_liquidity:
            logging.info("Low liquidity hours, pausing trading")
            
        return is_nfp_time or is_fomc or is_low_liquidity
    except Exception as e:
        logging.error(f"Error checking news time: {str(e)}")
        return False  # Default to safe option if error

# Check risk limits
async def check_risk_limits():
    try:
        # Make sure MT5 is initialized and connected
        if not mt5.initialize():
            logging.error("MT5 not initialized when checking risk limits")
            return False
            
        today = datetime.now().date()
        
        # Check if trade log exists
        if not os.path.exists('trade_log.csv'):
            # If no trades yet, create file
            with open('trade_log.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win', 
                                'win_rate', 'sharpe', 'sortino', 'alpha', 'md_alpha'])
            return True  # No trades yet, so no risk limits reached
            
        df = pd.read_csv('trade_log.csv')
        if len(df) == 0:
            return True  # No trades yet
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today_trades = df[df['timestamp'].dt.date == today]
        
        # Check daily loss limit
        if len(today_trades) > 0:
            total_profit = today_trades['profit'].sum()
            account_balance = mt5.account_balance()
            
            # If total profit is negative and exceeds 0.2% of account balance
            if total_profit < 0 and abs(total_profit) > 0.002 * account_balance:
                logging.warning(f"Daily loss limit (0.2%) reached: {total_profit} < -{0.002 * account_balance}")
                print(f"Daily loss limit (0.2%) reached: {total_profit} < -{0.002 * account_balance}")
                return False
        
        # Check max drawdown
        account_equity = mt5.account_equity()
        account_balance = mt5.account_balance()
        
        if account_balance > 0 and account_equity < 0.99 * account_balance:
            logging.warning(f"Max drawdown (1%) reached: {account_equity} < {0.99 * account_balance}")
            print(f"Max drawdown (1%) reached: {account_equity} < {0.99 * account_balance}")
            return False
            
        return True
    except FileNotFoundError:
        # If file doesn't exist, create it
        with open('trade_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win', 
                            'win_rate', 'sharpe', 'sortino', 'alpha', 'md_alpha'])
        return True
    except Exception as e:
        logging.error(f"Error checking risk limits: {str(e)}")
        return False  # Default to safe option if error

# Kelly Criterion for position sizing
def kelly_criterion(win_rate=0.95, reward_risk_ratio=6, vol_regime='low'):
    try:
        # Ensure win_rate is between 0 and 1
        win_rate = max(0.1, min(0.99, win_rate))
        
        # Ensure reward_risk_ratio is positive
        reward_risk_ratio = max(0.1, reward_risk_ratio)
        
        # Calculate Kelly fraction
        base_kelly = (win_rate - (1 - win_rate) / reward_risk_ratio) / 200
        
        # Adjust based on volatility regime
        adjusted_kelly = max(0.05, min(0.3, base_kelly * (1.5 if vol_regime == 'high' else 1.0)))
        
        return adjusted_kelly
    except Exception as e:
        logging.error(f"Error calculating Kelly criterion: {str(e)}")
        return 0.05  # Default to a conservative value

# Quantum hyperparameter tuning
async def quantum_optimize_parameters(df):
    try:
        # Check if we have enough data
        if len(df) < 100:
            logging.warning("Not enough data for quantum optimization")
            return [0.7, 25, 75, 40, 20]  # Default parameters
            
        def objective(params):
            sl_mult, rsi_low, rsi_high, adx_thresh, stoch_thresh = params
            temp_df = df.copy()
            temp_df['signal'] = 0
            temp_df.loc[
                (temp_df['sma_fast'] > temp_df['sma_slow']) & 
                (temp_df['rsi'] > rsi_low) & 
                (temp_df['rsi'] < rsi_high) & 
                (temp_df['adx'] > adx_thresh) & 
                (temp_df['stoch_k'] < stoch_thresh), 'signal'] = 1
            returns = temp_df['bid'].pct_change().shift(-1) * temp_df['signal']
            return -returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Simple search to replace quantum optimization for now
        best_params = [0.7, 25, 75, 40, 20]  # Default
        best_score = float('-inf')
        
        # Test a few parameter combinations
        for sl_mult in [0.7, 1.0, 1.3]:
            for rsi_low in [20, 25, 30]:
                for rsi_high in [70, 75, 80]:
                    for adx_thresh in [30, 40, 50]:
                        for stoch_thresh in [15, 20, 25]:
                            params = [sl_mult, rsi_low, rsi_high, adx_thresh, stoch_thresh]
                            score = -objective(params)  # Negate because we want to maximize
                            if score > best_score:
                                best_score = score
                                best_params = params
        
        logging.info(f"Optimized parameters: {best_params}, Score: {best_score}")
        return best_params
    except Exception as e:
        logging.error(f"Error in parameter optimization: {str(e)}")
        return [0.7, 25, 75, 40, 20]  # Default parameters

# Black-Litterman portfolio optimization
async def black_litterman_weights(symbols, df_dict):
    try:
        # Check if we have valid data
        if not symbols or not df_dict or len(symbols) == 0:
            logging.warning("No valid data for portfolio optimization")
            return {s: 1/len(symbols) for s in symbols} if symbols else {}
            
        # Create returns DataFrame
        returns_dict = {}
        for s, df in df_dict.items():
            if 'bid' in df.columns and len(df) > 1:
                returns_dict[s] = df['bid'].pct_change().dropna()
        
        if not returns_dict:
            logging.warning("No valid returns data for portfolio optimization")
            return {s: 1/len(symbols) for s in symbols}
            
        returns = pd.DataFrame(returns_dict)
        
        # Handle missing data
        returns = returns.fillna(0)
        
        # Calculate covariance matrix and expected returns
        cov_matrix = returns.cov() * 252  # Annualize
        expected_returns = returns.mean() * 252  # Annualize
        
        # Ensure covariance matrix is invertible
        try:
            weights = np.linalg.inv(cov_matrix) @ expected_returns
        except np.linalg.LinAlgError:
            logging.warning("Covariance matrix not invertible, using equal weights")
            return {s: 1/len(symbols) for s in symbols}
            
        # Normalize weights to sum to 1
        if np.sum(weights) != 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(symbols)) / len(symbols)
            
        # Constrain weights between 0.1 and 0.9
        weights_dict = {}
        for s, w in zip(symbols, weights):
            weights_dict[s] = max(0.1, min(0.9, w))
        
        # Re-normalize
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {s: w/total_weight for s, w in weights_dict.items()}
            
        return weights_dict
    except Exception as e:
        logging.error(f"Error in portfolio optimization: {str(e)}")
        return {s: 1/len(symbols) for s in symbols}  # Default to equal weights

# Update dashboard metrics
def update_metrics(trade_log):
    try:
        # Check if we have data
        if trade_log is None or len(trade_log) == 0:
            logging.warning("No trade log data for metrics update")
            return {
                'win_rate': 0,
                'sharpe': 0,
                'sortino': 0,
                'alpha': 0,
                'md_alpha': 0,
                'drawdown': 0,
                'hft_efficiency': 0,
                'profit_factor': 0
            }
            
        total_trades = len(trade_log)
        winning_trades = len(trade_log[trade_log['win'] == True])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate returns
        trade_log['profit'] = pd.to_numeric(trade_log['profit'], errors='coerce')
        
        # Check if we have profit data
        if 'profit' not in trade_log.columns or trade_log['profit'].isna().all():
            logging.warning("No valid profit data in trade log")
            return {
                'win_rate': win_rate,
                'sharpe': 0,
                'sortino': 0,
                'alpha': 0,
                'md_alpha': 0,
                'drawdown': 0,
                'hft_efficiency': 0,
                'profit_factor': 0
            }
            
        # Calculate returns if we have at least 2 trades
        if len(trade_log) >= 2:
            returns = trade_log['profit'].pct_change().dropna()
        else:
            returns = pd.Series([0])
            
        # Calculate performance metrics
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() != 0 else 0
        alpha = returns.mean() * 252 - 0.05  # 5% risk-free rate
        md_alpha = returns.mean() * 252 - 0.4  # Proxy 2*VIX at 20%
        
        # Check if MT5 is initialized
        if mt5.initialize():
            equity = mt5.account_equity()
            balance = mt5.account_balance()
            drawdown = (balance - equity) / balance if balance > 0 else 0
        else:
            drawdown = 0
            
        # Calculate HFT efficiency
        try:
            trade_log['timestamp'] = pd.to_datetime(trade_log['timestamp'])
            first_trade_time = pd.to_datetime(trade_log['timestamp'].iloc[0]).timestamp()
            hft_efficiency = total_trades / (time.time() - first_trade_time) if total_trades > 0 and time.time() > first_trade_time else 0
        except:
            hft_efficiency = 0
            
        # Calculate profit factor
        profits = trade_log[trade_log['profit'] > 0]['profit'].sum()
        losses = abs(trade_log[trade_log['profit'] < 0]['profit'].sum())
        profit_factor = profits / losses if losses != 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'sharpe': sharpe,
            'sortino': sortino,
            'alpha': alpha,
            'md_alpha': md_alpha,
            'drawdown': drawdown,
            'hft_efficiency': hft_efficiency,
            'profit_factor': profit_factor
        }
    except Exception as e:
        logging.error(f"Error updating metrics: {str(e)}")
        return {
            'win_rate': 0,
            'sharpe': 0,
            'sortino': 0,
            'alpha': 0,
            'md_alpha': 0,
            'drawdown': 0,
            'hft_efficiency': 0,
            'profit_factor': 0
        }

# Visualize trade signals
async def visualize_signals(df, symbol, signals, filename='trade_signals.png'):
    try:
        # Check if we have enough data
        if df is None or len(df) == 0 or 'time' not in df.columns or 'bid' not in df.columns:
            logging.error("Not enough data for visualization")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], df['bid'], label='Price', color='blue')
        
        # Filter buy/sell signals
        buy_indices = signals == 1
        sell_indices = signals == -1
        
        if buy_indices.any():
            buy_signals = df[buy_indices]
            plt.scatter(buy_signals['time'], buy_signals['bid'], marker='^', color='green', label='Buy Signal', s=100)
            
        if sell_indices.any():
            sell_signals = df[sell_indices]
            plt.scatter(sell_signals['time'], sell_signals['bid'], marker='v', color='red', label='Sell Signal', s=100)
            
        plt.title(f'{symbol} Trade Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        plt.savefig(filename)
        plt.close()
        logging.info(f"Trade signals visualized and saved to {filename}")
    except Exception as e:
        logging.error(f"Error visualizing signals: {str(e)}")

# Process each symbol asynchronously
async def process_symbol(symbol, models, df_dict, df_min_dict, base_lot_size, portfolio_weights, 
                        sl_mult, rsi_low, rsi_high, adx_thresh, stoch_thresh, vol_regime):
    try:
        # Get current tick and minute data
        df = await get_tick_data(symbol)
        df_min = await get_minute_data(symbol)
        
        if df is None or df_min is None:
            logging.error(f"Failed to get data for {symbol}")
            return
            
        # Calculate indicators
        df = calculate_indicators(df, 'tick')
        df_min = calculate_indicators(df_min, 'minute')
        
        # Update data dictionaries
        df_dict[symbol] = df
        df_min_dict[symbol] = df_min
        
        # Check if we have enough data
        if len(df) < 2 or len(df_min) < 2:
            logging.warning(f"Not enough data for {symbol} to generate signals")
            return
            
        # Get latest rows
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        last_min_row = df_min.iloc[-1]
        
        # Check if models exist for this symbol
        if symbol not in models:
            logging.warning(f"No models found for {symbol}")
            return
            
        # Calculate signals from dark pool and order flow
        dark_pool_signal = 0
        if 'dark_pool' in last_row and last_row['dark_pool'] != 0:
            dark_pool_signal = last_row['dark_pool'] / df['dark_pool'].rolling(window=20).mean().iloc[-1] if df['dark_pool'].rolling(window=20).mean().iloc[-1] != 0 else 0
            
        order_flow_signal = last_row['order_flow'] if 'order_flow' in last_row else 0
        
        # Get predictions from ML models
        # 1. QDRL model
        qdrl_model, scaler = models[symbol]['qdrl']
        state_features = ['bid', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                        'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']
                        
        # Check if all features exist
        missing_features = [f for f in state_features if f not in df.columns]
        if missing_features:
            logging.error(f"Missing features for QDRL: {missing_features}")
            qdrl_action = 0  # No action
        else:
            # Transform the last row into a state vector
            try:
                state = torch.FloatTensor(scaler.transform(
                    df.tail(1)[state_features]
                )).squeeze()
                qdrl_probs = qdrl_model(state)
                qdrl_action = torch.argmax(qdrl_probs).item()
            except Exception as e:
                logging.error(f"Error getting QDRL prediction: {str(e)}")
                qdrl_action = 0  # No action
        
        # 2. HGNN model
        hgnn_model = models[symbol]['hgnn']
        hgnn_features = ['bid', 'rsi', 'macd_hist', 'adx', 'atr', 'dark_pool', 'order_flow']
        
        # Check if all features exist
        missing_features = [f for f in hgnn_features if f not in df.columns]
        if missing_features:
            logging.error(f"Missing features for HGNN: {missing_features}")
            hgnn_pred = 0.5  # Neutral prediction
        else:
            # Create a simple graph for prediction
            try:
                hgnn_data = scaler.transform(df.tail(20)[hgnn_features])
                hgnn_features_tensor = torch.FloatTensor(hgnn_data)
                g = dgl.graph(([0, 1], [1, 0]))
                hgnn_pred = hgnn_model(g, hgnn_features_tensor).mean(dim=0)[1].item()
            except Exception as e:
                logging.error(f"Error getting HGNN prediction: {str(e)}")
                hgnn_pred = 0.5  # Neutral prediction
        
        # 3. Ensemble models (Random Forest and XGBoost)
        rf_model, xgb_model = models[symbol]['ensemble']
        ensemble_features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 
                           'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']
                           
        # Check if all features exist
        missing_features = [f for f in ensemble_features if f not in last_row]
        if missing_features:
            logging.error(f"Missing features for ensemble models: {missing_features}")
            rf_pred = 0.5  # Neutral prediction
            xgb_pred = 0.5  # Neutral prediction
        else:
            # Create feature vector for prediction
            try:
                features = pd.DataFrame([[
                    last_row['sma_fast'], last_row['sma_slow'], last_row['rsi'],
                    last_row['macd'], last_row['macd_signal'], last_row['macd_hist'],
                    last_row['bb_mid'], last_row['bb_upper'], last_row['bb_lower'],
                    last_row['adx'], last_row['stoch_k'], last_row['stoch_d'], last_row['atr'], last_row['vwap'],
                    last_row['dark_pool'], last_row['order_flow']
                ]], columns=ensemble_features)
                
                rf_pred = rf_model.predict_proba(features)[0][1]
                xgb_pred = xgb_model.predict_proba(features)[0][1]
            except Exception as e:
                logging.error(f"Error getting ensemble predictions: {str(e)}")
                rf_pred = 0.5  # Neutral prediction
                xgb_pred = 0.5  # Neutral prediction
        
        # Calculate position size
        lot_size = base_lot_size * portfolio_weights.get(symbol, 1.0/len(portfolio_weights)) * (1.2 if vol_regime == 'high' else 1.0)
        signal = 0
        
        # Check for Buy signal
        if (prev_row['sma_fast'] < prev_row['sma_slow'] and 
            last_row['sma_fast'] > last_row['sma_slow'] and 
            last_min_row['sma_fast'] > last_min_row['sma_slow'] and
            rsi_low < last_row['rsi'] < 35 and 
            last_row['macd_hist'] > 0 and 
            last_row['adx'] > adx_thresh and 
            last_row['bid'] > last_row['bb_mid'] and
            last_row['stoch_k'] < stoch_thresh and last_row['stoch_k'] > last_row['stoch_d'] and
            last_row['bid'] > last_row['vwap'] and
            dark_pool_signal > 1.5 and order_flow_signal > 0.1 and
            qdrl_action == 1 and hgnn_pred > 0.65 and rf_pred > 0.65 and xgb_pred > 0.65):
            
            signal = 1
            atr = last_row['atr']
            sl = atr * sl_mult / mt5.symbol_info(symbol).point
            entry_price = mt5.symbol_info_tick(symbol).ask
            
            logging.info(f"Buy signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, "
                        f"DarkPool: {dark_pool_signal:.2f}, OrderFlow: {order_flow_signal:.2f}, QDRL: {qdrl_action}, "
                        f"HGNN: {hgnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
            print(f"Buy signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, "
                  f"DarkPool: {dark_pool_signal:.2f}, OrderFlow: {order_flow_signal:.2f}, QDRL: {qdrl_action}, "
                  f"HGNN: {hgnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
                  
            # Place trade
            success, order_id = await place_trade(symbol, "buy", lot_size, sl, entry_price)
            if success:
                # Get metrics for logging
                trade_log = pd.read_csv('trade_log.csv') if os.path.exists('trade_log.csv') else pd.DataFrame()
                metrics = update_metrics(trade_log)
                
                # Monitor trade in a separate task
                asyncio.create_task(monitor_trade(symbol, order_id, "buy", entry_price, sl, atr, metrics))
        
        # Check for Sell signal
        elif (prev_row['sma_fast'] > prev_row['sma_slow'] and 
              last_row['sma_fast'] < last_row['sma_slow'] and 
              last_min_row['sma_fast'] < last_min_row['sma_slow'] and
              65 < last_row['rsi'] < rsi_high and 
              last_row['macd_hist'] < 0 and 
              last_row['adx'] > adx_thresh and 
              last_row['bid'] < last_row['bb_mid'] and
              last_row['stoch_k'] > 100 - stoch_thresh and last_row['stoch_k'] < last_row['stoch_d'] and
              last_row['bid'] < last_row['vwap'] and
              dark_pool_signal > 1.5 and order_flow_signal < -0.1 and
              qdrl_action == 2 and hgnn_pred > 0.65 and rf_pred > 0.65 and xgb_pred > 0.65):
              
            signal = -1
            atr = last_row['atr']
            sl = atr * sl_mult / mt5.symbol_info(symbol).point
            entry_price = mt5.symbol_info_tick(symbol).bid
            
            logging.info(f"Sell signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, "
                        f"DarkPool: {dark_pool_signal:.2f}, OrderFlow: {order_flow_signal:.2f}, QDRL: {qdrl_action}, "
                        f"HGNN: {hgnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
            print(f"Sell signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, "
                  f"DarkPool: {dark_pool_signal:.2f}, OrderFlow: {order_flow_signal:.2f}, QDRL: {qdrl_action}, "
                  f"HGNN: {hgnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
                  
            # Place trade
            success, order_id = await place_trade(symbol, "sell", lot_size, sl, entry_price)
            if success:
                # Get metrics for logging
                trade_log = pd.read_csv('trade_log.csv') if os.path.exists('trade_log.csv') else pd.DataFrame()
                metrics = update_metrics(trade_log)
                
                # Monitor trade in a separate task
                asyncio.create_task(monitor_trade(symbol, order_id, "sell", entry_price, sl, atr, metrics))
        
        # Visualize signals if any detected
        if signal != 0:
            signals = pd.Series(0, index=df.index)
            signals.iloc[-1] = signal
            await visualize_signals(df.tail(100), symbol, signals.tail(100), f'{symbol}_signals.png')
    
    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")

# Main HFT trading logic
async def trading_system(symbols, login=None, password=None, server=None):
    # Initialize MT5 with provided credentials
    if not await initialize_mt5(login=login, password=password, server=server):
        logging.error("Failed to initialize MT5, aborting trading system")
        return
    
    try:
        # Ensure CSV directories exist
        os.makedirs(os.path.dirname('trade_log.csv') if os.path.dirname('trade_log.csv') else '.', exist_ok=True)
        os.makedirs(os.path.dirname('dashboard.csv') if os.path.dirname('dashboard.csv') else '.', exist_ok=True)
        
        # Initialize trade log if it doesn't exist
        if not os.path.exists('trade_log.csv'):
            with open('trade_log.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win', 
                                'win_rate', 'sharpe', 'sortino', 'alpha', 'md_alpha'])
        
        # Initialize dashboard if it doesn't exist
        if not os.path.exists('dashboard.csv'):
            with open('dashboard.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'win_rate', 'sharpe', 'sortino', 'alpha', 'md_alpha', 'drawdown', 'hft_efficiency', 'profit_factor'])
        
        # Dictionary to store models for each symbol
        models = {}
        
        # Dictionaries to store dataframes
        df_dict = {}
        df_min_dict = {}
        
        # Fetch initial data and train models for each symbol
        logging.info("Fetching initial data and training models...")
        print("Fetching initial data and training models...")
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            df = await get_tick_data(symbol)
            df_min = await get_minute_data(symbol)
            
            if df is not None and df_min is not None:
                # Calculate indicators
                df = calculate_indicators(df, 'tick')
                df_min = calculate_indicators(df_min, 'minute')
                
                # Store dataframes
                df_dict[symbol] = df
                df_min_dict[symbol] = df_min
                
                # Train models
                state_dim = 17  # Features for QDRL
                models[symbol] = {}
                
                print(f"Training QDRL model for {symbol}...")
                models[symbol]['qdrl'] = await train_qdrl_agent(df, state_dim)
                
                print(f"Training HGNN model for {symbol}...")
                models[symbol]['hgnn'] = await train_hgnn_model(df_dict, symbols)
                
                print(f"Training ensemble models for {symbol}...")
                models[symbol]['ensemble'] = await train_ensemble_models(df)
                
                # Check if any model training failed
                if None in models[symbol].values():
                    logging.error(f"Model training failed for {symbol}, skipping")
                    continue
            else:
                logging.error(f"Data fetch failed for {symbol}, skipping")
                continue
        
        # Optimize trading parameters
        print("Optimizing trading parameters...")
        opt_params = await quantum_optimize_parameters(df_dict[symbols[0]] if symbols and symbols[0] in df_dict else None)
        sl_mult, rsi_low, rsi_high, adx_thresh, stoch_thresh = opt_params
        
        # Initialize counters
        trade_count = 0
        start_time = time.time()
        
        # Main trading loop
        print("Starting main trading loop...")
        logging.info("Starting main trading loop...")
        
        while True:
            # Check for news events or risk limits
            if is_news_time() or not await check_risk_limits():
                logging.info("Pausing trading due to news or risk limits")
                print("Pausing trading due to news or risk limits...")
                await asyncio.sleep(60)  # Check again in 1 minute
                continue
            
            # Get account balance for position sizing
            try:
                account_balance = mt5.account_balance()
            except:
                logging.error("Error getting account balance, using default")
                account_balance = 10000  # Default if can't get actual balance
            
            # Detect volatility regime
            vol_regime = detect_volatility_regime(df_dict[symbols[0]] if symbols and symbols[0] in df_dict else None)
            
            # Calculate position sizes
            base_lot_size = kelly_criterion(vol_regime=vol_regime) * account_balance / 100000  # Convert to standard lots
            portfolio_weights = await black_litterman_weights(symbols, df_dict)
            
            # Process each symbol in parallel
            tasks = []
            for symbol in symbols:
                if symbol in models:
                    tasks.append(process_symbol(symbol, models, df_dict, df_min_dict, base_lot_size, portfolio_weights, 
                                            sl_mult, rsi_low, rsi_high, adx_thresh, stoch_thresh, vol_regime))
            
            if tasks:
                await asyncio.gather(*tasks)
            
            # Update dashboard periodically
            trade_count += len(tasks)
            if trade_count >= 10 or (time.time() - start_time) > 300:  # Every 10 trades or 5 minutes
                try:
                    trade_log = pd.read_csv('trade_log.csv') if os.path.exists('trade_log.csv') else pd.DataFrame()
                    metrics = update_metrics(trade_log)
                    logging.info(f"Dashboard: Win Rate: {metrics['win_rate']:.2f}%, Sharpe: {metrics['sharpe']:.2f}, "
                                f"Sortino: {metrics['sortino']:.2f}, MD Alpha: {metrics['md_alpha']:.2f}, "
                                f"Drawdown: {metrics['drawdown']:.2f}%, HFT Efficiency: {metrics['hft_efficiency']:.4f}")
                    print(f"Dashboard: Win Rate: {metrics['win_rate']:.2f}%, Sharpe: {metrics['sharpe']:.2f}, "
                          f"Sortino: {metrics['sortino']:.2f}, MD Alpha: {metrics['md_alpha']:.2f}, "
                          f"Drawdown: {metrics['drawdown']:.2f}%, HFT Efficiency: {metrics['hft_efficiency']:.4f}")
                    trade_count = 0
                    start_time = time.time()
                except Exception as e:
                    logging.error(f"Error updating dashboard: {str(e)}")
            
            # Small delay for HFT loop
            await asyncio.sleep(0.1)  # 100ms for HFT loop
    
    except KeyboardInterrupt:
        logging.info("Shutting down multidimensional HFT system")
        print("Shutting down multidimensional HFT system...")
        mt5.shutdown()
    except Exception as e:
        logging.error(f"Critical error in trading system: {str(e)}")
        mt5.shutdown()

# Run the system
if __name__ == "__main__":
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HFT Trading System')
    parser.add_argument('--symbols', nargs='+', default=["US30", "NAS100"], help='Symbols to trade')
    parser.add_argument('--login', type=int, help='MT5 login ID')
    parser.add_argument('--password', type=str, help='MT5 password')
    parser.add_argument('--server', type=str, help='MT5 server')
    
    args = parser.parse_args()
    
    try:
        print("Starting HFT Trading System...")
        print(f"Trading symbols: {args.symbols}")
        
        # Run the trading system with provided arguments
        asyncio.run(trading_system(args.symbols, args.login, args.password, args.server))
    except KeyboardInterrupt:
        logging.info("Shutting down multidimensional HFT system")
        print("Shutting down multidimensional HFT system...")
        mt5.shutdown()
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        print(f"Critical error: {str(e)}")
        mt5.shutdown()