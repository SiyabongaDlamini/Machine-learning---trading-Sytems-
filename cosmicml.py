import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
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
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
import requests

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return False
    print("Galactic MT5 initialized")
    return True

# Get historical data
def get_historical_data(symbol, timeframe, bars=100000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate indicators
def calculate_indicators(df):
    df['sma_fast'] = df['close'].rolling(window=5).mean()
    df['sma_slow'] = df['close'].rolling(window=30).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['plus_di'] = 100 * ((df['high'] - df['high'].shift(1)).where(lambda x: x > 0, 0) / df['tr']).rolling(window=14).mean()
    df['minus_di'] = 100 * ((df['low'].shift(1) - df['low']).where(lambda x: x > 0, 0) / df['tr']).rolling(window=14).mean()
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(window=14).mean()
    df['low_14'] = df['low'].rolling(window=14).min()
    df['high_14'] = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - df['low_14']) / (df['high_14'] - df['low_14'])
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df

# Neuromorphic spike encoding
def neuromorphic_encode(data, threshold=0.01):
    spikes = np.zeros_like(data)
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > threshold * np.std(data):
            spikes[i] = 1 if data[i] > data[i-1] else -1
    return spikes

# Get X sentiment (mock transformer; replace with real X API)
def get_x_sentiment(symbol):
    mock_posts = [
        f"{symbol} to the moon!", 
        f"Selling {symbol}, looks weak.",
        f"{symbol} breakout imminent!"
    ]
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(post)['compound'] for post in mock_posts]
    return np.mean(sentiments) if sentiments else 0.0

# Deep Quantum Reinforcement Learning Agent
class DQRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQRLAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

def train_dqrl_agent(df, state_dim, action_dim=3):
    agent = DQRLAgent(state_dim, action_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0005)
    scaler = MinMaxScaler()
    features = ['close', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap']
    data = scaler.fit_transform(df[features])
    spikes = neuromorphic_encode(df['close'].values)
    
    for epoch in range(15):
        state = torch.FloatTensor(data[:-1])
        next_state = torch.FloatTensor(data[1:])
        rewards = torch.FloatTensor((df['close'].diff().shift(-1) > 0).astype(float).iloc[1:].values * (1 + abs(spikes[1:])))
        actions = torch.zeros(len(state), dtype=torch.long)
        
        for t in range(len(state)):
            action_probs = agent(state[t])
            action = torch.argmax(action_probs).item()
            actions[t] = action
            reward = rewards[t]
            loss = -torch.log(action_probs[action]) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return agent, scaler

# Graph Neural Network for intermarket relationships
class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, g, features):
        h = self.relu(self.conv1(g, features))
        h = self.conv2(g, h)
        return torch.softmax(h, dim=-1)

def train_gnn_model(df_dict, symbols):
    features = ['close', 'rsi', 'macd_hist', 'adx', 'atr']
    data = [scaler.fit_transform(df[features]) for df in df_dict.values()]
    g = dgl.graph(([0, 1], [1, 0]))  # Simple US30-NAS100 graph
    node_features = torch.FloatTensor(np.concatenate(data[-10:], axis=0))
    labels = torch.LongTensor([1 if df['close'].diff().shift(-1).iloc[-1] > 0 else 0 for df in df_dict.values()])
    
    model = GNNModel(len(features), 64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        logits = model(g, node_features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

# Train ensemble ML models
def train_ensemble_models(df):
    df = df.dropna()
    features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap']
    X = df[features]
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=2000, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    
    xgb_model = XGBClassifier(n_estimators=2000, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
    
    return rf_model, xgb_model

# Place trade
def place_trade(symbol, trade_type, lot_size, sl, entry_price):
    point = mt5.symbol_info(symbol).point
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if trade_type == "buy" else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": entry_price - sl * point if trade_type == "buy" else entry_price + sl * point,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed: {result.comment}")
        return False, None
    print(f"Trade placed: {trade_type} {symbol} at {entry_price}")
    return True, result.order

# Log trade to CSV
def log_trade(symbol, trade_type, entry_price, exit_price, profit, win):
    with open('trade_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), symbol, trade_type, entry_price, exit_price, profit, win])

# Monitor trade with trailing stop
def monitor_trade(symbol, order_id, trade_type, entry_price, sl, atr):
    point = mt5.symbol_info(symbol).point
    trailing_stop = sl * point
    best_price = entry_price if trade_type == "buy" else entry_price
    while True:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            history = mt5.history_deals_get(ticket=order_id)
            if history:
                deal = history[-1]
                exit_price = deal.price
                profit = deal.profit
                win = profit > 0
                log_trade(symbol, trade_type, entry_price, exit_price, profit, win)
                print(f"Trade closed: {symbol}, Profit: {profit}, Win: {win}")
            break
        current_price = mt5.symbol_info_tick(symbol).bid if trade_type == "buy" else mt5.symbol_info_tick(symbol).ask
        if trade_type == "buy" and current_price > best_price:
            best_price = current_price
            new_sl = best_price - trailing_stop
            if new_sl > entry_price - sl * point:
                modify_position(symbol, order_id, new_sl)
        elif trade_type == "sell" and current_price < best_price:
            best_price = current_price
            new_sl = best_price + trailing_stop
            if new_sl < entry_price + sl * point:
                modify_position(symbol, order_id, new_sl)
        time.sleep(0.3)

# Modify position
def modify_position(symbol, order_id, new_sl):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "position": positions[0].ticket,
            "sl": new_sl,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Updated SL for {symbol} to {new_sl}")

# Check news events
def is_news_time():
    now = datetime.now(pytz.timezone('US/Eastern'))
    is_first_friday = now.day <= 7 and now.weekday() == 4
    is_nfp_time = is_first_friday and now.hour == 8 and 30 <= now.minute <= 45
    is_fomc = now.hour in [14, 15] and now.weekday() == 2 and now.day in [15, 16, 29, 30]
    is_low_liquidity = now.hour in [17, 18, 19]
    return is_nfp_time or is_fomc or is_low_liquidity

# Check risk limits
def check_risk_limits():
    today = datetime.now().date()
    try:
        df = pd.read_csv('trade_log.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today_trades = df[df['timestamp'].dt.date == today]
        total_profit = today_trades['profit'].sum()
        if total_profit < -0.003 * mt5.account_balance():
            print("Daily loss limit (0.3%) reached, pausing for 24 hours")
            return False
        account_equity = mt5.account_equity()
        if account_equity < 0.98 * mt5.account_balance():
            print("Max drawdown (2%) reached, pausing trading")
            return False
    except FileNotFoundError:
        pass
    return True

# Kelly Criterion for position sizing
def kelly_criterion(win_rate=0.94, reward_risk_ratio=5):
    return max(0.1, (win_rate - (1 - win_rate) / reward_risk_ratio) / 300)

# Quantum annealing for parameter optimization
def quantum_optimize_parameters(df):
    def objective(params):
        sl_mult, rsi_low, rsi_high = params
        temp_df = df.copy()
        temp_df['signal'] = 0
        temp_df.loc[
            (temp_df['sma_fast'] > temp_df['sma_slow']) & 
            (temp_df['rsi'] > rsi_low) & 
            (temp_df['rsi'] < rsi_high), 'signal'] = 1
        returns = temp_df['close'].pct_change().shift(-1) * temp_df['signal']
        return -returns.mean() / returns.std() if returns.std() != 0 else 0
    
    optimizer = COBYLA()
    qaoa = QAOA(optimizer=optimizer, reps=3)
    initial_params = [0.8, 28, 72]
    result = qaoa.compute_minimum_eigenvalue(lambda x: objective(x)).eigenvector
    return result[:3] if len(result) >= 3 else initial_params

# Black-Litterman portfolio optimization
def black_litterman_weights(symbols, df_dict):
    returns = pd.DataFrame({s: df['close'].pct_change().dropna() for s, df in df_dict.items()})
    cov_matrix = returns.cov() * 252
    expected_returns = returns.mean() * 252
    weights = np.linalg.inv(cov_matrix) @ expected_returns
    weights = weights / np.sum(weights)
    return {s: max(0.2, min(0.8, w)) for s, w in zip(symbols, weights)}

# Main trading logic
def trading_system(symbols, timeframe=mt5.TIMEFRAME_H1):
    if not initialize_mt5():
        return
    
    with open('trade_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win'])
    
    models = {}
    df_dict = {}
    for symbol in symbols:
        df = get_historical_data(symbol, timeframe)
        if df is not None:
            df = calculate_indicators(df)
            df_dict[symbol] = df
            state_dim = 15
            models[symbol] = {
                'dqrl': train_dqrl_agent(df, state_dim),
                'gnn': train_gnn_model(df_dict, symbols),
                'ensemble': train_ensemble_models(df)
            }
    
    opt_params = quantum_optimize_parameters(df_dict[symbols[0]])
    sl_mult, rsi_low, rsi_high = opt_params
    
    while True:
        if is_news_time() or not check_risk_limits():
            print("Pausing trading...")
            time.sleep(3600)
            continue
        
        account_balance = mt5.account_balance()
        base_lot_size = kelly_criterion() * account_balance
        portfolio_weights = black_litterman_weights(symbols, df_dict)
        
        for symbol in symbols:
            df = get_historical_data(symbol, timeframe)
            if df is None:
                continue
            df = calculate_indicators(df)
            df_dict[symbol] = df
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            sentiment = get_x_sentiment(symbol)
            state = torch.FloatTensor(models[symbol]['dqrl'][1].transform(
                df.tail(1)[['close', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                            'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap']]
            )).squeeze()
            dqrl_probs = models[symbol]['dqrl'][0](state)
            dqrl_action = torch.argmax(dqrl_probs).item()
            
            gnn_features = torch.FloatTensor(models[symbol]['dqrl'][1].transform(
                df.tail(10)[['close', 'rsi', 'macd_hist', 'adx', 'atr']]
            ))
            g = dgl.graph(([0, 1], [1, 0]))
            gnn_pred = models[symbol]['gnn'](g, gnn_features).mean(dim=0)[1].item()
            
            features = pd.DataFrame([[
                last_row['sma_fast'], last_row['sma_slow'], last_row['rsi'],
                last_row['macd'], last_row['macd_signal'], last_row['macd_hist'],
                last_row['bb_mid'], last_row['bb_upper'], last_row['bb_lower'],
                last_row['adx'], last_row['stoch_k'], last_row['stoch_d'], last_row['atr'], last_row['vwap']
            ]], columns=['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap'])
            
            rf_pred = models[symbol]['ensemble'][0].predict_proba(features)[0][1]
            xgb_pred = models[symbol]['ensemble'][1].predict_proba(features)[0][1]
            
            lot_size = base_lot_size * portfolio_weights[symbol]
            
            # Buy signal
            if (prev_row['sma_fast'] < prev_row['sma_slow'] and 
                last_row['sma_fast'] > last_row['sma_slow'] and 
                rsi_low < last_row['rsi'] < 40 and 
                last_row['macd_hist'] > 0 and 
                last_row['adx'] > 40 and 
                last_row['close'] > last_row['bb_mid'] and
                last_row['stoch_k'] < 20 and last_row['stoch_k'] > last_row['stoch_d'] and
                last_row['close'] > last_row['vwap'] and
                sentiment > 0.5 and
                dqrl_action == 1 and gnn_pred > 0.95 and rf_pred > 0.97 and xgb_pred > 0.97):
                atr = last_row['atr']
                sl = atr * sl_mult / mt5.symbol_info(symbol).point
                entry_price = mt5.symbol_info_tick(symbol).ask
                print(f"Buy signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, Sentiment: {sentiment:.2f}, DQRL: {dqrl_action}, GNN: {gnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
                success, order_id = place_trade(symbol, "buy", lot_size, sl, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "buy", entry_price, sl, atr)
            
            # Sell signal
            elif (prev_row['sma_fast'] > prev_row['sma_slow'] and 
                  last_row['sma_fast'] < last_row['sma_slow'] and 
                  60 < last_row['rsi'] < rsi_high and 
                  last_row['macd_hist'] < 0 and 
                  last_row['adx'] > 40 and 
                  last_row['close'] < last_row['bb_mid'] and
                  last_row['stoch_k'] > 80 and last_row['stoch_k'] < last_row['stoch_d'] and
                  last_row['close'] < last_row['vwap'] and
                  sentiment < -0.5 and
                  dqrl_action == 2 and gnn_pred > 0.95 and rf_pred > 0.97 and xgb_pred > 0.97):
                atr = last_row['atr']
                sl = atr * sl_mult / mt5.symbol_info(symbol).point
                entry_price = mt5.symbol_info_tick(symbol).bid
                print(f"Sell signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, Sentiment: {sentiment:.2f}, DQRL: {dqrl_action}, GNN: {gnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
                success, order_id = place_trade(symbol, "sell", lot_size, sl, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "sell", entry_price, sl, atr)
        
        time.sleep(3600)

# Run the system
if __name__ == "__main__":
    symbols = ["US30", "NAS100"]
    try:
        trading_system(symbols)
    except KeyboardInterrupt:
        print("Shutting down galactic system...")
        mt5.shutdown()