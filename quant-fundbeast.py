import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Model, Input

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return False
    print("MT5 initialized successfully")
    return True

# Get historical data
def get_historical_data(symbol, timeframe, bars=30000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate indicators
def calculate_indicators(df):
    df['sma_fast'] = df['close'].rolling(window=10).mean()
    df['sma_slow'] = df['close'].rolling(window=50).mean()
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
    return df

# Prepare LSTM+Transformer data
def prepare_deep_data(df, lookback=60):
    scaler = MinMaxScaler()
    features = ['close', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr']
    data = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)
    return np.array(X), np.array(y), scaler

# Train LSTM+Transformer model
def train_deep_model(df):
    X, y, scaler = prepare_deep_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_layer = Input(shape=(X.shape[1], X.shape[2]))
    lstm = LSTM(100, return_sequences=True)(input_layer)
    attn = MultiHeadAttention(num_heads=4, key_dim=25)(lstm, lstm)
    norm = LayerNormalization()(attn + lstm)
    lstm2 = LSTM(50)(norm)
    drop = Dropout(0.3)(lstm2)
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Deep Model Accuracy: {accuracy:.2f}")
    return model, scaler

# Train ensemble ML models
def train_ensemble_models(df):
    df = df.dropna()
    features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr']
    X = df[features]
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    
    xgb_model = XGBClassifier(n_estimators=500, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
    
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
    print(f"SVM Accuracy: {svm_accuracy:.2f}")
    
    return rf_model, xgb_model, svm_model

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
        time.sleep(1)

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
        if total_profit < -0.01 * mt5.account_balance():
            print("Daily loss limit (1%) reached, pausing for 24 hours")
            return False
        account_equity = mt5.account_equity()
        if account_equity < 0.95 * mt5.account_balance():
            print("Max drawdown (5%) reached, pausing trading")
            return False
    except FileNotFoundError:
        pass
    return True

# Kelly Criterion for position sizing
def kelly_criterion(win_rate=0.9, reward_risk_ratio=3):
    return max(0.1, (win_rate - (1 - win_rate) / reward_risk_ratio) / 1000)

# Portfolio optimization (simplified Markowitz)
def optimize_portfolio(symbols):
    weights = {s: 0.5 for s in symbols}  # Equal weights for simplicity
    return weights

# Main trading logic
def trading_system(symbols, timeframe=mt5.TIMEFRAME_H1):
    if not initialize_mt5():
        return
    
    with open('trade_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win'])
    
    models = {}
    for symbol in symbols:
        df = get_historical_data(symbol, timeframe)
        if df is not None:
            df = calculate_indicators(df)
            models[symbol] = {
                'deep': train_deep_model(df),
                'ensemble': train_ensemble_models(df)
            }
    
    portfolio_weights = optimize_portfolio(symbols)
    
    while True:
        if is_news_time() or not check_risk_limits():
            print("Pausing trading...")
            time.sleep(3600)
            continue
        
        account_balance = mt5.account_balance()
        base_lot_size = kelly_criterion() * account_balance
        
        for symbol in symbols:
            df = get_historical_data(symbol, timeframe)
            if df is None:
                continue
            df = calculate_indicators(df)
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            lstm_data = df.tail(60)[['close', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr']]
            lstm_scaled = models[symbol]['deep'][1].transform(lstm_data)
            lstm_X = np.array([lstm_scaled])
            deep_pred = models[symbol]['deep'][0].predict(lstm_X, verbose=0)[0][0]
            
            features = pd.DataFrame([[
                last_row['sma_fast'], last_row['sma_slow'], last_row['rsi'],
                last_row['macd'], last_row['macd_signal'], last_row['macd_hist'],
                last_row['bb_mid'], last_row['bb_upper'], last_row['bb_lower'],
                last_row['adx'], last_row['stoch_k'], last_row['stoch_d'], last_row['atr']
            ]], columns=['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'stoch_k', 'stoch_d', 'atr'])
            
            rf_pred = models[symbol]['ensemble'][0].predict_proba(features)[0][1]
            xgb_pred = models[symbol]['ensemble'][1].predict_proba(features)[0][1]
            svm_pred = models[symbol]['ensemble'][2].predict_proba(features)[0][1]
            
            lot_size = base_lot_size * portfolio_weights[symbol]
            
            # Buy signal
            if (prev_row['sma_fast'] < prev_row['sma_slow'] and 
                last_row['sma_fast'] > last_row['sma_slow'] and 
                30 < last_row['rsi'] < 45 and 
                last_row['macd_hist'] > 0 and 
                last_row['adx'] > 30 and 
                last_row['close'] > last_row['bb_mid'] and
                last_row['stoch_k'] < 30 and last_row['stoch_k'] > last_row['stoch_d'] and
                deep_pred > 0.9 and rf_pred > 0.9 and xgb_pred > 0.9 and svm_pred > 0.9):
                atr = last_row['atr']
                sl = atr * 1.0 / mt5.symbol_info(symbol).point
                entry_price = mt5.symbol_info_tick(symbol).ask
                print(f"Buy signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, Deep: {deep_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}, SVM: {svm_pred:.2f}")
                success, order_id = place_trade(symbol, "buy", lot_size, sl, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "buy", entry_price, sl, atr)
            
            # Sell signal
            elif (prev_row['sma_fast'] > prev_row['sma_slow'] and 
                  last_row['sma_fast'] < last_row['sma_slow'] and 
                  55 < last_row['rsi'] < 70 and 
                  last_row['macd_hist'] < 0 and 
                  last_row['adx'] > 30 and 
                  last_row['close'] < last_row['bb_mid'] and
                  last_row['stoch_k'] > 70 and last_row['stoch_k'] < last_row['stoch_d'] and
                  deep_pred > 0.9 and rf_pred > 0.9 and xgb_pred > 0.9 and svm_pred > 0.9):
                atr = last_row['atr']
                sl = atr * 1.0 / mt5.symbol_info(symbol).point
                entry_price = mt5.symbol_info_tick(symbol).bid
                print(f"Sell signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, Deep: {deep_pred:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}, SVM: {svm_pred:.2f}")
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
        print("Shutting down...")
        mt5.shutdown()