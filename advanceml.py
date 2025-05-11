import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return False
    print("MT5 initialized successfully")
    return True

# Get historical data
def get_historical_data(symbol, timeframe, bars=10000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate indicators
def calculate_indicators(df):
    # SMA
    df['sma_fast'] = df['close'].rolling(window=10).mean()
    df['sma_slow'] = df['close'].rolling(window=50).mean()
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    # ADX
    df['plus_di'] = 100 * ((df['high'] - df['high'].shift(1)).where(lambda x: x > 0, 0) / df['tr']).rolling(window=14).mean()
    df['minus_di'] = 100 * ((df['low'].shift(1) - df['low']).where(lambda x: x > 0, 0) / df['tr']).rolling(window=14).mean()
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(window=14).mean()
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    return df

# Train ML models
def train_ml_models(df):
    df = df.dropna()
    features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'atr']
    X = df[features]
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    
    # XGBoost
    xgb_model = XGBClassifier(n_estimators=200, random_state=42)
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
        time.sleep(1)

# Modify position (update SL)
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

# Check for news events
def is_news_time():
    now = datetime.now(pytz.timezone('US/Eastern'))
    is_first_friday = now.day <= 7 and now.weekday() == 4
    is_nfp_time = is_first_friday and now.hour == 8 and 30 <= now.minute <= 45
    is_fomc = now.hour in [14, 15] and now.weekday() == 2 and now.day in [15, 16, 29, 30]
    return is_nfp_time or is_fomc

# Check daily loss limit
def check_daily_loss():
    today = datetime.now().date()
    df = pd.read_csv('trade_log.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today_trades = df[df['timestamp'].dt.date == today]
    total_profit = today_trades['profit'].sum()
    if total_profit < -0.02 * mt5.account_balance():
        print("Daily loss limit reached, pausing for 24 hours")
        return False
    return True

# Calculate dynamic lot size
def calculate_lot_size(account_balance, risk_percent=0.5):
    return max(0.1, (account_balance * risk_percent / 100) / 1000)

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
            models[symbol] = train_ml_models(df)
    
    while True:
        if is_news_time() or not check_daily_loss():
            print("Pausing trading...")
            time.sleep(3600)
            continue
        
        account_balance = mt5.account_balance()
        lot_size = calculate_lot_size(account_balance)
        
        for symbol in symbols:
            df = get_historical_data(symbol, timeframe)
            if df is None:
                continue
            df = calculate_indicators(df)
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            features = pd.DataFrame([[
                last_row['sma_fast'], last_row['sma_slow'], last_row['rsi'],
                last_row['macd'], last_row['macd_signal'], last_row['macd_hist'],
                last_row['bb_mid'], last_row['bb_upper'], last_row['bb_lower'],
                last_row['adx'], last_row['atr']
            ]], columns=['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'atr'])
            
            rf_pred = models[symbol][0].predict_proba(features)[0][1]
            xgb_pred = models[symbol][1].predict_proba(features)[0][1]
            
            # Buy signal
            if (prev_row['sma_fast'] < prev_row['sma_slow'] and 
                last_row['sma_fast'] > last_row['sma_slow'] and 
                30 < last_row['rsi'] < 50 and 
                last_row['macd_hist'] > 0 and 
                last_row['adx'] > 25 and 
                last_row['close'] > last_row['bb_mid'] and
                rf_pred > 0.8 and xgb_pred > 0.8):
                atr = last_row['atr']
                sl = atr * 1.5 / mt5.symbol_info(symbol).point
                entry_price = mt5.symbol_info_tick(symbol).ask
                print(f"Buy signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
                success, order_id = place_trade(symbol, "buy", lot_size, sl, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "buy", entry_price, sl, atr)
            
            # Sell signal
            elif (prev_row['sma_fast'] > prev_row['sma_slow'] and 
                  last_row['sma_fast'] < last_row['sma_slow'] and 
                  50 < last_row['rsi'] < 70 and 
                  last_row['macd_hist'] < 0 and 
                  last_row['adx'] > 25 and 
                  last_row['close'] < last_row['bb_mid'] and
                  rf_pred > 0.8 and xgb_pred > 0.8):
                atr = last_row['atr']
                sl = atr * 1.5 / mt5.symbol_info(symbol).point
                entry_price = mt5.symbol_info_tick(symbol).bid
                print(f"Sell signal for {symbol}, RSI: {last_row['rsi']:.2f}, ADX: {last_row['adx']:.2f}, RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}")
                success, order_id = place_trade(symbol, "sell", lot_size, sl, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "sell", entry_price, sl, atr)
        
        time.sleep(3600)

# Run the system
if __name__ == "__main__":
    symbols = ["US30m", "NAS100"]
    try:
        trading_system(symbols)
    except KeyboardInterrupt:
        print("Shutting down...")
        mt5.shutdown()