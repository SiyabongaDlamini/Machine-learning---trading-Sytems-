import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return False
    print("MT5 initialized successfully")
    return True

# Get historical data
def get_historical_data(symbol, timeframe, bars=5000):
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
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    return df

# Train ML model
def train_ml_model(df):
    df = df.dropna()
    features = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr']
    X = df[features]
    
    # Create target: 1 if next candle closes higher (for buys), else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ML Model Accuracy: {accuracy:.2f}")
    
    return model

# Place trade
def place_trade(symbol, trade_type, lot_size, sl, tp, entry_price):
    point = mt5.symbol_info(symbol).point
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if trade_type == "buy" else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": entry_price - sl * point if trade_type == "buy" else entry_price + sl * point,
        "tp": entry_price + tp * point if trade_type == "buy" else entry_price - tp * point,
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

# Monitor and close trade
def monitor_trade(symbol, order_id, trade_type, entry_price, sl, tp):
    point = mt5.symbol_info(symbol).point
    sl_price = entry_price - sl * point if trade_type == "buy" else entry_price + sl * point
    tp_price = entry_price + tp * point if trade_type == "buy" else entry_price - tp * point
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
        time.sleep(1)

# Check for news events (simplified, assumes high-impact news times are known)
def is_news_time():
    # Example: Avoid trading during US NFP (first Friday of month, 8:30 AM EST)
    now = datetime.now(pytz.timezone('US/Eastern'))
    is_first_friday = now.day <= 7 and now.weekday() == 4
    is_nfp_time = is_first_friday and now.hour == 8 and 30 <= now.minute <= 45
    return is_nfp_time

# Calculate dynamic lot size
def calculate_lot_size(account_balance, risk_percent=1.0):
    return max(0.1, (account_balance * risk_percent / 100) / 1000)

# Main trading logic
def trading_system(symbols, timeframe=mt5.TIMEFRAME_H1):
    if not initialize_mt5():
        return
    
    # Initialize CSV log
    with open('trade_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit', 'win'])
    
    # Train ML models for each symbol
    models = {}
    for symbol in symbols:
        df = get_historical_data(symbol, timeframe)
        if df is not None:
            df = calculate_indicators(df)
            models[symbol] = train_ml_model(df)
    
    while True:
        if is_news_time():
            print("High-impact news detected, pausing trading...")
            time.sleep(3600)
            continue
        
        account_balance = mt5.account_balance()
        lot_size = calculate_lot_size(account_balance)
        
        for symbol in symbols:
            # Get and process data
            df = get_historical_data(symbol, timeframe)
            if df is None:
                continue
            df = calculate_indicators(df)
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Prepare ML features
            features = pd.DataFrame([[
                last_row['sma_fast'], last_row['sma_slow'], last_row['rsi'],
                last_row['macd'], last_row['macd_signal'], last_row['macd_hist'], last_row['atr']
            ]], columns=['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr'])
            
            # Buy signal
            if (prev_row['sma_fast'] < prev_row['sma_slow'] and 
                last_row['sma_fast'] > last_row['sma_slow'] and 
                last_row['rsi'] < 35 and 
                last_row['macd_hist'] > 0 and
                models[symbol].predict(features)[0] == 1):
                atr = last_row['atr']
                sl = atr * 1.5 / mt5.symbol_info(symbol).point
                tp = sl * 2
                entry_price = mt5.symbol_info_tick(symbol).ask
                print(f"Buy signal detected for {symbol}, RSI: {last_row['rsi']:.2f}, MACD Hist: {last_row['macd_hist']:.2f}")
                success, order_id = place_trade(symbol, "buy", lot_size, sl, tp, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "buy", entry_price, sl, tp)
            
            # Sell signal
            elif (prev_row['sma_fast'] > prev_row['sma_slow'] and 
                  last_row['sma_fast'] < last_row['sma_slow'] and 
                  last_row['rsi'] > 65 and 
                  last_row['macd_hist'] < 0 and
                  models[symbol].predict(features)[0] == 1):
                atr = last_row['atr']
                sl = atr * 1.5 / mt5.symbol_info(symbol).point
                tp = sl * 2
                entry_price = mt5.symbol_info_tick(symbol).bid
                print(f"Sell signal detected for {symbol}, RSI: {last_row['rsi']:.2f}, MACD Hist: {last_row['macd_hist']:.2f}")
                success, order_id = place_trade(symbol, "sell", lot_size, sl, tp, entry_price)
                if success:
                    monitor_trade(symbol, order_id, "sell", entry_price, sl, tp)
        
        time.sleep(3600)  # Check every hour

# Run the system
if __name__ == "__main__":
    symbols = ["US30m", "USTECm"]
    try:
        trading_system(symbols)
    except KeyboardInterrupt:
        print("Shutting down...")
        mt5.shutdown()