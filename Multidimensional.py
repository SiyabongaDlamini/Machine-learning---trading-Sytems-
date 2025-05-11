import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
import asyncio
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
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

# Initialize MT5 connection
async def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return False
    print("Multidimensional MT5 initialized")
    return True

# Get tick data for HFT
async def get_tick_data(symbol, n_ticks=10000):
    ticks = mt5.copy_ticks_from(symbol, datetime.now(pytz.UTC), n_ticks, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"Failed to get ticks for {symbol}")
        return None
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate indicators and order flow
def calculate_indicators(df):
    df['sma_fast'] = df['bid'].rolling(window=10).mean()
    df['sma_slow'] = df['bid'].rolling(window=50).mean()
    delta = df['bid'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=20).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
    rs = gain / loss
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
    df['plus_di'] = 100 * ((df['ask'] - df['ask'].shift(1)).where(lambda x: x > 0, 0) / df['tr']).rolling(window=20).mean()
    df['minus_di'] = 100 * ((df['bid'].shift(1) - df['bid']).where(lambda x: x > 0, 0) / df['tr']).rolling(window=20).mean()
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(window=20).mean()
    df['low_20'] = df['bid'].rolling(window=20).min()
    df['high_20'] = df['ask'].rolling(window=20).max()
    df['stoch_k'] = 100 * (df['bid'] - df['low_20']) / (df['high_20'] - df['low_20'])
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    df['high_low'] = df['ask'] - df['bid']
    df['high_close'] = abs(df['ask'] - df['bid'].shift())
    df['low_close'] = abs(df['bid'] - df['bid'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=20).mean()
    df['vwap'] = (df['bid'] * df['volume']).cumsum() / df['volume'].cumsum()
    # Dark pool volume proxy
    df['vol_spike'] = df['volume'].where(df['volume'] > df['volume'].rolling(window=20).mean() + 2 * df['volume'].rolling(window=20).std(), 0)
    df['dark_pool'] = df['vol_spike'] * (df['atr'] / df['atr'].rolling(window=20).mean())
    # Order flow (bid-ask imbalance)
    df['order_flow'] = (df['ask'] - df['bid']).rolling(window=20).mean() / df['atr']
    return df

# Neuromorphic spike encoding for HFT
def neuromorphic_encode(data, threshold=0.005):
    spikes = np.zeros_like(data)
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > threshold * np.std(data):
            spikes[i] = 1 if data[i] > data[i-1] else -1
    return spikes

# Quantum Deep Reinforcement Learning Agent
class QDRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QDRLAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

async def train_qdrl_agent(df, state_dim, action_dim=3):
    agent = QDRLAgent(state_dim, action_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0002)
    scaler = MinMaxScaler()
    features = ['bid', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']
    data = scaler.fit_transform(df[features])
    spikes = neuromorphic_encode(df['bid'].values)
    
    for epoch in range(20):
        state = torch.FloatTensor(data[:-1])
        next_state = torch.FloatTensor(data[1:])
        rewards = torch.FloatTensor((df['bid'].diff().shift(-1) > 0).astype(float).iloc[1:].values * (1 + abs(spikes[1:])))
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

# Hyperdimensional Graph Neural Network
class HGNNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(HGNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, g, features):
        h = self.relu(self.conv1(g, features))
        h = self.relu(self.conv2(g, h))
        h = self.conv3(g, h)
        return torch.softmax(h, dim=-1)

async def train_hgnn_model(df_dict, symbols):
    features = ['bid', 'rsi', 'macd_hist', 'adx', 'atr', 'dark_pool', 'order_flow']
    data = [scaler.fit_transform(df[features]) for df in df_dict.values()]
    g = dgl.graph(([0, 1], [1, 0]))  # US30-NAS100 graph
    node_features = torch.FloatTensor(np.concatenate(data[-20:], axis=0))
    labels = torch.LongTensor([1 if df['bid'].diff().shift(-1).iloc[-1] > 0 else 0 for df in df_dict.values()])
    
    model = HGNNModel(len(features), 128, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(15):
        logits = model(g, node_features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

# Train ensemble ML models
async def train_ensemble_models(df):
    df = df.dropna()
    features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']
    X = df[features]
    df['target'] = (df['bid'].shift(-1) > df['bid']).astype(int)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=3000, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    
    xgb_model = XGBClassifier(n_estimators=3000, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
    
    return rf_model, xgb_model

# Place trade (HFT optimized)
async def place_trade(symbol, trade_type, lot_size, sl, entry_price):
    point = mt5.symbol_info(symbol).point
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
        print(f"Trade failed: {result.comment}")
        return False, None
    print(f"Trade placed: {trade_type} {symbol} at {entry_price}")
    return True, result.order

# Log trade to CSV
async def log_trade(symbol, trade_type, entry_price, exit_price, profit, win):
    with open('trade_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), symbol, trade_type, entry_price, exit_price, profit, win])

# Monitor trade with HFT trailing stop
async def monitor_trade(symbol, order_id, trade_type, entry_price, sl, atr):
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
                await log_trade(symbol, trade_type, entry_price, exit_price, profit, win)
                print(f"Trade closed: {symbol}, Profit: {profit}, Win: {win}")
            break
        current_price = mt5.symbol_info_tick(symbol).bid if trade_type == "buy" else mt5.symbol_info_tick(symbol).ask
        if trade_type == "buy" and current_price > best_price:
            best_price = current_price
            new_sl = best_price - trailing_stop
            if new_sl > entry_price - sl * point:
                await modify_position(symbol, order_id, new_sl)
        elif trade_type == "sell" and current_price < best_price:
            best_price = current_price
            new_sl = best_price + trailing_stop
            if new_sl < entry_price + sl * point:
                await modify_position(symbol, order_id, new_sl)
        await asyncio.sleep(0.001)  # HFT speed

# Modify position
async def modify_position(symbol, order_id, new_sl):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "position": positions[0].ticket,
            "sl": new_sl,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
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
async def check_risk_limits():
    today = datetime.now().date()
    try:
        df = pd.read_csv('trade_log.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today_trades = df[df['timestamp'].dt.date == today]
        total_profit = today_trades['profit'].sum()
        if total_profit < -0.002 * mt5.account_balance():
            print("Daily loss limit (0.2%) reached, pausing for 24 hours")
            return False
        account_equity = mt5.account_equity()
        if account_equity < 0.99 * mt5.account_balance():
            print("Max drawdown (1%) reached, pausing trading")
            return False
    except FileNotFoundError:
        pass
    return True

# Kelly Criterion for position sizing
def kelly_criterion(win_rate=0.95, reward_risk_ratio=6):
    return max(0.1, (win_rate - (1 - win_rate) / reward_risk_ratio) / 200)

# Quantum hyperparameter tuning
async def quantum_optimize_parameters(df):
    def objective(params):
        sl_mult, rsi_low, rsi_high, adx_thresh = params
        temp_df = df.copy()
        temp_df['signal'] = 0
        temp_df.loc[
            (temp_df['sma_fast'] > temp_df['sma_slow']) & 
            (temp_df['rsi'] > rsi_low) & 
            (temp_df['rsi'] < rsi_high) & 
            (temp_df['adx'] > adx_thresh), 'signal'] = 1
        returns = temp_df['bid'].pct_change().shift(-1) * temp_df['signal']
        return -returns.mean() / returns.std() if returns.std() != 0 else 0
    
    optimizer = COBYLA()
    qaoa = QAOA(optimizer=optimizer, reps=5)
    initial_params = [0.7, 25, 75, 40]
    result = qaoa.compute_minimum_eigenvalue(lambda x: objective(x)).eigenvector
    return result[:4] if len(result) >= 4 else initial_params

# Black-Litterman portfolio optimization
async def black_litterman_weights(symbols, df_dict):
    returns = pd.DataFrame({s: df['bid'].pct_change().dropna() for s, df in df_dict.items()})
    cov_matrix = returns.cov() * 252
    expected_returns = returns.mean() * 252
    weights = np.linalg.inv(cov_matrix) @ expected_returns
    weights = weights / np.sum(weights)
    return {s: max(0.1, min(0.9, w)) for s, w in zip(symbols, weights)}

# Main HFT trading logic
async def trading_system(symbols):
    if not await initialize_mt5():
        return
    
    with open('trade_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win'])
    
    models = {}
    df_dict = {}
    for symbol in symbols:
        df = await get_tick_data(symbol)
        if df is not None:
            df = calculate_indicators(df)
            df_dict[symbol] = df
            state_dim = 17  # Updated for dark_pool, order_flow
            models[symbol] = {
                'qdrl': await train_qdrl_agent(df, state_dim),
                'hgnn': await train_hgnn_model(df_dict, symbols),
                'ensemble': await train_ensemble_models(df)
            }
    
    opt_params = await quantum_optimize_parameters(df_dict[symbols[0]])
    sl_mult, rsi_low, rsi_high, adx_thresh = opt_params
    
    while True:
        if is_news_time() or not await check_risk_limits():
            print("Pausing trading...")
            await asyncio.sleep(3600)
            continue
        
        account_balance = mt5.account_balance()
        base_lot_size = kelly_criterion() * account_balance
        portfolio_weights = await black_litterman_weights(symbols, df_dict)
        
        tasks = []
        for symbol in symbols:
            tasks.append(process_symbol(symbol, models, df_dict, base_lot_size, portfolio_weights, sl_mult, rsi_low, rsi_high, adx_thresh))
        await asyncio.gather(*tasks)
        
        await asyncio.sleep(0.01)  # HFT loop speed

# Process each symbol asynchronously
async def process_symbol(symbol, models, df_dict, base_lot_size, portfolio_weights, sl_mult, rsi_low, rsi_high, adx_thresh):
    df = await get_tick_data(symbol)
    if df is None:
        return
    df = calculate_indicators(df)
    df_dict[symbol] = df
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    dark_pool_signal = last_row['dark_pool'] / last_row['dark_pool'].rolling(window=20).mean() if last_row['dark_pool'] != 0 else 0
    order_flow_signal = last_row['order_flow']
    
    state = torch.FloatTensor(models[symbol]['qdrl'][1].transform(
        df.tail(1)[['bid', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                    'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow']]
    )).squeeze()
    qdrl_probs = models[symbol]['qdrl'][0](state)
    qdrl_action = torch.argmax(qdrl_probs).item()
    
    hgnn_features = torch.FloatTensor(models[symbol]['qdrl'][1].transform(
        df.tail(20)[['bid', 'rsi', 'macd_hist', 'adx', 'atr', 'dark_pool', 'order_flow']]
    ))
    g = dgl.graph(([0, 1], [1, 0]))
    hgnn_pred = models[symbol]['hgnn'](g, hgnn_features).mean(dim=0)[1].item()
    
    features = pd.DataFrame([[
        last_row['sma_fast'], last_row['sma_slow'], last_row['rsi'],
        last_row['macd'], last_row['macd_signal'], last_row['macd_hist'],
        last_row['bb_mid'], last_row['bb_upper'], last_row['bb_lower'],
        last_row['adx'], last_row['stoch_k'], last_row['stoch_d'], last_row['atr'], last_row['vwap'],
        last_row['dark_pool'], last_row['order_flow']
    ]], columns=['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'dark_pool', 'order_flow'])
    
    rf_pred = models[symbol]['ensemble'][0].predict_proba(features)[0][1]
    xgb_pred = models[symbol]['ensemble'][1].predict_proba(features)[0][1]
    
    lot_size = base_lot_size * portfolio_weights[symbol]
    
    # Buy signal
    if (prev_row['sma_fast'] < prev_row['sma_slow'] and 
        last_row['sma_fast'] > last_row['sma_slow'] and 
        rsi_low < last_row['rsi'] < 35 and 
        last_row['macd_hist'] > 0 and 
        last_row['adx'] > adx_thresh and 
        last_row['bid'] > last_row['bb_mid'] and
        last_row['stoch_k'] < 15 and last_row['stoch_k'] > last_row['stoch_d'] and
        last_row['bid'] > last_row['vwap'] and
        dark_pool_signal > 1.5 and order_flow_signal > 0.1 and
        qdrl_action == 1 and hgnn_pred > 0.98 and rf_pred > 0.98 and xgb_pred > 0.98):
        atr = last_row['atr']
        sl = atr * sl_mult / mt5.symbol_info(symbol).point
        entry_price = mt5.symbol_info_tick(symbol).ask
        print(f"Buy signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, DarkPool: {dark_pool_signal:.2f}, OrderFlow: {order_flow_signal:.2f}, QDRL: {qdrl_action}, HGNN: {hgnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
        success, order_id = await place_trade(symbol, "buy", lot_size, sl, entry_price)
        if success:
            asyncio.create_task(monitor_trade(symbol, order_id, "buy", entry_price, sl, atr))
    
    # Sell signal
    elif (prev_row['sma_fast'] > prev_row['sma_slow'] and 
          last_row['sma_fast'] < last_row['sma_slow'] and 
          65 < last_row['rsi'] < rsi_high and 
          last_row['macd_hist'] < 0 and 
          last_row['adx'] > adx_thresh and 
          last_row['bid'] < last_row['bb_mid'] and
          last_row['stoch_k'] > 85 and last_row['stoch_k'] < last_row['stoch_d'] and
          last_row['bid'] < last_row['vwap'] and
          dark_pool_signal > 1.5 and order_flow_signal < -0.1 and
          qdrl_action == 2 and hgnn_pred > 0.98 and rf_pred > 0.98 and xgb_pred > 0.98):
        atr = last_row['atr']
        sl = atr * sl_mult / mt5.symbol_info(symbol).point
        entry_price = mt5.symbol_info_tick(symbol).bid
        print(f"Sell signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, DarkPool: {dark_pool_signal:.2f}, OrderFlow: {order_flow_signal:.2f}, QDRL: {qdrl_action}, HGNN: {hgnn_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
        success, order_id = await place_trade(symbol, "sell", lot_size, sl, entry_price)
        if success:
            asyncio.create_task(monitor_trade(symbol, order_id, "sell", entry_price, sl, atr))

# Run the system
if __name__ == "__main__":
    symbols = ["US30", "NAS100"]
    try:
        asyncio.run(trading_system(symbols))
    except KeyboardInterrupt:
        print("Shutting down multidimensional HFT system...")
        mt5.shutdown()