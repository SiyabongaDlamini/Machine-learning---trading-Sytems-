import asyncio
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler

# Configuration Module
# Define all constants and settings for the HFT system
MT5_LOGIN = 123456  # Replace with your MT5 account login
MT5_PASSWORD = "your_password"  # Replace with your MT5 password
MT5_SERVER = "your_server"  # Replace with your MT5 server
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCHF"]
BASE_LOT_SIZE = 0.01
TIMEFRAME = mt5.TIMEFRAME_M1
MAX_DAILY_LOSS = 0.002  # 0.2% of account balance
MAX_DRAWDOWN = 0.01  # 1% maximum drawdown
MAX_POSITION_SIZE = 0.1
VOLATILITY_THRESHOLD = 0.005
BACKTEST_DATA_PATH = "historical_data/"
HEARTBEAT_INTERVAL = 300  # 5 minutes in seconds
RECONNECT_DELAY = 5  # Seconds to wait before reconnect attempt

# Utilities Module
def setup_logging():
    """Configure logging with file rotation and console output."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'hft_system.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.info("Logging configured successfully")

def load_models(symbols):
    """Load pre-trained machine learning models from disk."""
    try:
        if os.path.exists('models.pkl'):
            with open('models.pkl', 'rb') as f:
                models = pickle.load(f)
                return {symbol: models.get(symbol, {'rf': None, 'xgb': None}) for symbol in symbols}
        else:
            logging.warning("Models file not found. Initializing with None.")
            return {symbol: {'rf': None, 'xgb': None} for symbol in symbols}
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return {symbol: {'rf': None, 'xgb': None} for symbol in symbols}

def save_models(models):
    """Save machine learning models to disk."""
    try:
        with open('models.pkl', 'wb') as f:
            pickle.dump(models, f)
        logging.info("Models saved successfully")
    except Exception as e:
        logging.error(f"Error saving models: {str(e)}")

def compute_volatility(series):
    """Calculate volatility as standard deviation over mean."""
    try:
        pct_change = series.pct_change().dropna()
        if len(pct_change) < 2:
            return 0.001
        return pct_change.std() / pct_change.mean() if pct_change.mean() != 0 else 0.001
    except Exception as e:
        logging.error(f"Error computing volatility: {str(e)}")
        return 0.001

def ensure_mt5_connection():
    """Ensure MT5 connection is active, attempt reconnect if needed."""
    if not mt5.terminal_info():
        logging.warning("MT5 connection lost. Attempting to reconnect...")
        for attempt in range(3):
            if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                logging.info("MT5 connection re-established")
                return True
            time.sleep(RECONNECT_DELAY)
        logging.error("Failed to reconnect to MT5 after multiple attempts")
        return False
    return True

# Data Fetcher Module
class DataFetcher:
    def __init__(self, symbols, timeframe):
        self.symbols = symbols
        self.timeframe = timeframe
        self.cache = {}
        self.last_fetch_time = {}

    async def fetch_data(self):
        """Fetch latest tick and bar data for all symbols asynchronously."""
        if not ensure_mt5_connection():
            return {}
        
        data = {}
        tasks = [self._fetch_symbol_data(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                logging.error(f"Failed to fetch data for {symbol}: {str(result)}")
                data[symbol] = self.cache.get(symbol, pd.DataFrame())
            else:
                data[symbol] = result
        return data

    async def _fetch_symbol_data(self, symbol):
        """Fetch data for a single symbol with caching."""
        try:
            current_time = time.time()
            if symbol in self.last_fetch_time and (current_time - self.last_fetch_time[symbol]) < 0.1:
                return self.cache.get(symbol, pd.DataFrame())
                
            # Fetch tick data
            ticks = mt5.copy_ticks_from(symbol, datetime.utcnow(), 1000, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                raise ValueError(f"No tick data available for {symbol}")
                
            tick_df = pd.DataFrame(ticks)
            tick_df['time'] = pd.to_datetime(tick_df['time'], unit='s')
            tick_df.set_index('time', inplace=True)
            
            # Fetch bar data
            rates = mt5.copy_rates_from(symbol, self.timeframe, datetime.utcnow(), 100)
            if rates is None or len(rates) == 0:
                raise ValueError(f"No bar data available for {symbol}")
                
            bar_df = pd.DataFrame(rates)
            bar_df['time'] = pd.to_datetime(bar_df['time'], unit='s')
            bar_df.set_index('time', inplace=True)
            
            # Combine and cache
            combined_df = bar_df.join(tick_df[['bid', 'ask']], how='outer').ffill()
            self.cache[symbol] = combined_df.tail(1000)  # Limit cache size
            self.last_fetch_time[symbol] = current_time
            return combined_df
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return self.cache.get(symbol, pd.DataFrame())

    def get_historical_data(self, symbol, start_date, end_date):
        """Fetch historical data for backtesting."""
        if not ensure_mt5_connection():
            return pd.DataFrame()
            
        try:
            rates = mt5.copy_rates_range(symbol, self.timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                logging.error(f"No historical data for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

# Signal Generator Module
class SignalGenerator:
    def __init__(self, models):
        self.models = models
        self.scaler = StandardScaler()
        
    def generate_signals(self, data):
        """Generate trading signals for all symbols."""
        signals = {}
        for symbol, df in data.items():
            signals[symbol] = self._generate_signal_for_symbol(symbol, df)
        return signals

    def _generate_signal_for_symbol(self, symbol, df):
        """Generate signal for a single symbol."""
        if symbol not in self.models or df.empty or len(df) < 50:
            return 0
            
        try:
            df = self._compute_indicators(df)
            features = df[['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'bollinger_width', 'atr']].dropna()
            if features.empty or len(features) < 10:
                return 0
                
            scaled_features = self.scaler.fit_transform(features)
            latest_features = scaled_features[-1].reshape(1, -1)
            
            rf_pred = self._predict(self.models[symbol]['rf'], latest_features)
            xgb_pred = self._predict(self.models[symbol]['xgb'], latest_features)
            return self._combine_predictions(rf_pred, xgb_pred)
            
        except Exception as e:
            logging.error(f"Error generating signal for {symbol}: {str(e)}")
            return 0

    def _compute_indicators(self, df):
        """Calculate technical indicators."""
        df = df.copy()
        
        # Moving Averages
        df['sma_fast'] = df['bid'].rolling(window=10, min_periods=1).mean()
        df['sma_slow'] = df['bid'].rolling(window=50, min_periods=1).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['bid'])
        
        # MACD
        df['ema_fast'] = df['bid'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['bid'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['bid'].rolling(window=20, min_periods=1).mean()
        df['bb_std'] = df['bid'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bollinger_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        return df

    def _calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index."""
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(50, index=series.index)

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range."""
        try:
            high_low = df['ask'] - df['bid']
            high_close = np.abs(df['ask'] - df['bid'].shift())
            low_close = np.abs(df['bid'] - df['bid'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(window=period, min_periods=1).mean()
        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(0.0001, index=df.index)

    def _predict(self, model, features):
        """Safe model prediction with fallback."""
        try:
            return model.predict(features)[0] if model else 0
        except Exception as e:
            logging.error(f"Model prediction failed: {str(e)}")
            return 0

    def _combine_predictions(self, rf_pred, xgb_pred):
        """Combine predictions into a trading signal."""
        if rf_pred == 1 and xgb_pred == 1:
            return 1  # Buy
        elif rf_pred == -1 and xgb_pred == -1:
            return -1  # Sell
        return 0  # Neutral

# Trade Executor Module
class TradeExecutor:
    def __init__(self, base_lot_size):
        self.base_lot_size = base_lot_size
        self.max_retries = 3
        self.order_timeout = 2  # Seconds

    async def execute_trade(self, symbol, signal, lot_size):
        """Execute a trade with retry logic."""
        if not ensure_mt5_connection():
            return False
            
        for attempt in range(self.max_retries):
            try:
                request = self._build_request(symbol, signal, lot_size)
                result = await self._send_order_with_timeout(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Trade executed: {symbol}, {'Buy' if signal > 0 else 'Sell'}, Lot: {lot_size}, Ticket: {result.deal}")
                    return True
                else:
                    logging.error(f"Trade failed for {symbol}: {result.comment}, Retcode: {result.retcode}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(0.5)
            except Exception as e:
                logging.error(f"Error executing trade for {symbol}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5)
        return False

    async def _send_order_with_timeout(self, request):
        """Send order with a timeout to prevent hanging."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, mt5.order_send, request)
        return await asyncio.wait_for(future, timeout=self.order_timeout)

    def _build_request(self, symbol, signal, lot_size):
        """Build a trade request."""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            raise ValueError(f"No tick data for {symbol}")
            
        price = tick.ask if signal > 0 else tick.bid
        sl_offset = 0.0005
        tp_offset = 0.0010
        
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - sl_offset if signal > 0 else price + sl_offset,
            "tp": price + tp_offset if signal > 0 else price - tp_offset,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "comment": f"HFT_{'Buy' if signal > 0 else 'Sell'}"
        }

# Risk Manager Module
class RiskManager:
    def __init__(self):
        self.daily_loss = 0
        self.max_drawdown = MAX_DRAWDOWN
        self.max_position_size = MAX_POSITION_SIZE
        self.last_reset = datetime.utcnow().date()

    def check_global_risk(self):
        """Check global risk limits."""
        if not ensure_mt5_connection():
            return False
            
        try:
            # Reset daily loss if new day
            current_date = datetime.utcnow().date()
            if current_date > self.last_reset:
                self.daily_loss = 0
                self.last_reset = current_date
                
            account_balance = mt5.account_balance()
            account_equity = mt5.account_equity()
            if account_balance <= 0 or account_equity <= 0:
                logging.error("Invalid account balance or equity")
                return False
                
            if self.daily_loss > MAX_DAILY_LOSS * account_balance:
                logging.warning(f"Daily loss exceeded: {self.daily_loss} > {MAX_DAILY_LOSS * account_balance}")
                return False
                
            drawdown = (account_balance - account_equity) / account_balance
            if drawdown > self.max_drawdown:
                logging.warning(f"Max drawdown exceeded: {drawdown} > {self.max_drawdown}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error checking global risk: {str(e)}")
            return False

    def adjust_lot_size(self, base_lot_size, volatility):
        """Adjust lot size based on volatility."""
        try:
            if volatility > VOLATILITY_THRESHOLD:
                adjusted = base_lot_size * (VOLATILITY_THRESHOLD / volatility)
            else:
                adjusted = base_lot_size
            return min(max(0.01, adjusted), self.max_position_size)
        except Exception as e:
            logging.error(f"Error adjusting lot size: {str(e)}")
            return base_lot_size

    def update_daily_loss(self, profit):
        """Update daily loss tracking."""
        try:
            self.daily_loss = max(0, self.daily_loss - profit)
        except Exception as e:
            logging.error(f"Error updating daily loss: {str(e)}")

# Order Manager Module
class OrderManager:
    def __init__(self):
        self.open_orders = {}
        self.trailing_stop_pips = 30  # 30 pips trailing stop
        
    async def manage_open_orders(self):
        """Monitor and manage open orders."""
        if not ensure_mt5_connection():
            return
            
        try:
            positions = mt5.positions_get()
            if not positions:
                self.open_orders.clear()
                return
                
            current_orders = {pos.ticket: pos for pos in positions}
            for ticket, pos in list(self.open_orders.items()):
                if ticket not in current_orders:
                    del self.open_orders[ticket]
                    logging.info(f"Position {ticket} closed")
                    
            for pos in positions:
                ticket = pos.ticket
                symbol = pos.symbol
                order_type = pos.type
                open_price = pos.price_open
                current_price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                
                if ticket not in self.open_orders:
                    self.open_orders[ticket] = pos
                    logging.info(f"New position detected: {ticket}, {symbol}")
                    
                # Implement trailing stop
                pip_value = mt5.symbol_info(symbol).point * 10  # Assuming 5-digit pricing
                trailing_stop = self.trailing_stop_pips * pip_value
                
                if order_type == mt5.ORDER_TYPE_BUY and current_price > open_price + trailing_stop:
                    new_sl = current_price - trailing_stop
                    if new_sl > pos.sl + pip_value:  # Only move SL if significantly better
                        self._modify_order(ticket, new_sl, pos.tp)
                elif order_type == mt5.ORDER_TYPE_SELL and current_price < open_price - trailing_stop:
                    new_sl = current_price + trailing_stop
                    if new_sl < pos.sl - pip_value or pos.sl == 0:
                        self._modify_order(ticket, new_sl, pos.tp)
                        
        except Exception as e:
            logging.error(f"Error managing orders: {str(e)}")

    def _modify_order(self, ticket, sl, tp):
        """Modify an existing order."""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": tp,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to modify order {ticket}: {result.comment}")
            else:
                logging.info(f"Order {ticket} modified: SL={sl}, TP={tp}")
        except Exception as e:
            logging.error(f"Error modifying order {ticket}: {str(e)}")

# Performance Tracker Module
class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.total_profit = 0
        self.win_count = 0
        self.loss_count = 0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0

    def record_trade(self, symbol, signal, lot_size, entry_price):
        """Record a new trade."""
        try:
            trade = {
                'symbol': symbol,
                'signal': signal,
                'lot_size': lot_size,
                'entry_price': entry_price,
                'entry_time': datetime.utcnow(),
                'exit_price': None,
                'exit_time': None,
                'profit': 0,
                'ticket': None
            }
            self.trades.append(trade)
            logging.info(f"Trade recorded: {symbol}, {'Buy' if signal > 0 else 'Sell'}, Entry: {entry_price}")
        except Exception as e:
            logging.error(f"Error recording trade: {str(e)}")

    def update_metrics(self):
        """Update performance metrics based on closed trades."""
        if not ensure_mt5_connection():
            return
            
        try:
            positions = {pos.ticket: pos for pos in mt5.positions_get()}
            closed_trades = []
            
            for trade in self.trades:
                if trade['exit_price'] is not None:
                    continue
                    
                ticket = trade.get('ticket')
                if ticket and ticket not in positions:
                    # Assume closed if no longer in open positions
                    symbol = trade['symbol']
                    tick = mt5.symbol_info_tick(symbol)
                    trade['exit_price'] = tick.bid if trade['signal'] > 0 else tick.ask
                    trade['exit_time'] = datetime.utcnow()
                    self._calculate_trade_profit(trade)
                    closed_trades.append(trade)
                elif not ticket and positions:
                    # Assign ticket to new trade
                    for pos in positions.values():
                        if (pos.symbol == trade['symbol'] and 
                            abs(pos.price_open - trade['entry_price']) < 0.0001 and 
                            (pos.type == mt5.ORDER_TYPE_BUY if trade['signal'] > 0 else mt5.ORDER_TYPE_SELL)):
                            trade['ticket'] = pos.ticket
                            break
            
            if closed_trades:
                self._update_statistics(closed_trades)
                self._log_performance()
        except Exception as e:
            logging.error(f"Error updating performance metrics: {str(e)}")

    def _calculate_trade_profit(self, trade):
        """Calculate profit for a trade."""
        try:
            if trade['signal'] > 0:
                trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['lot_size'] * 100000
            else:
                trade['profit'] = (trade['entry_price'] - trade['exit_price']) * trade['lot_size'] * 100000
        except Exception as e:
            logging.error(f"Error calculating profit: {str(e)}")
            trade['profit'] = 0

    def _update_statistics(self, closed_trades):
        """Update running statistics."""
        for trade in closed_trades:
            self.total_profit += trade['profit']
            if trade['profit'] > 0:
                self.win_count += 1
                self.current_consecutive_losses = 0
            else:
                self.loss_count += 1
                self.current_consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_consecutive_losses)

    def _log_performance(self):
        """Log current performance metrics."""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        logging.info(f"Performance: Total Profit={self.total_profit:.2f}, "
                    f"Wins={self.win_count}, Losses={self.loss_count}, "
                    f"Win Rate={win_rate:.2%}, Max Consecutive Losses={self.max_consecutive_losses}")

# Backtester Module
class Backtester:
    def __init__(self, data_fetcher, signal_generator, trade_executor, risk_manager):
        self.data_fetcher = data_fetcher
        self.signal_generator = signal_generator
        self.trade_executor = trade_executor
        self.risk_manager = risk_manager

    async def run_backtest(self, start_date, end_date):
        """Run backtest over historical data."""
        total_profit = 0
        trade_count = 0
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        for symbol in self.data_fetcher.symbols:
            try:
                data = self.data_fetcher.get_historical_data(symbol, start_date, end_date)
                if data.empty or len(data) < 100:
                    logging.warning(f"Insufficient data for {symbol}")
                    continue
                    
                for i in range(100, len(data)):
                    window = data.iloc[:i+1]
                    signals = self.signal_generator.generate_signals({symbol: window})
                    signal = signals.get(symbol, 0)
                    
                    if signal != 0:
                        entry_price = window['bid'].iloc[-1]
                        volatility = compute_volatility(window['bid'].tail(50))
                        lot_size = self.risk_manager.adjust_lot_size(self.trade_executor.base_lot_size, volatility)
                        exit_price = self._simulate_exit(window, signal, i)
                        profit = self._calculate_simulated_profit(entry_price, exit_price, lot_size, signal)
                        total_profit += profit
                        trade_count += 1
                        self.risk_manager.update_daily_loss(profit)
                        logging.debug(f"Backtest trade: {symbol}, {'Buy' if signal > 0 else 'Sell'}, Profit={profit:.2f}")
                        
            except Exception as e:
                logging.error(f"Backtest error for {symbol}: {str(e)}")
                
        results = {"profit": total_profit, "trades": trade_count}
        logging.info(f"Backtest Results: Total Profit={results['profit']:.2f}, Trades={results['trades']}")
        return results

    def _simulate_exit(self, data, signal, current_idx):
        """Simulate trade exit based on simple rules."""
        try:
            future_data = data.iloc[current_idx:]
            if len(future_data) < 5:
                return data['bid'].iloc[-1]
                
            sl = data['bid'].iloc[current_idx] - 0.0005 if signal > 0 else data['bid'].iloc[current_idx] + 0.0005
            tp = data['bid'].iloc[current_idx] + 0.0010 if signal > 0 else data['bid'].iloc[current_idx] - 0.0010
            
            for price in future_data['bid']:
                if signal > 0 and (price <= sl or price >= tp):
                    return price
                elif signal < 0 and (price >= sl or price <= tp):
                    return price
            return future_data['bid'].iloc[-1]
        except Exception as e:
            logging.error(f"Error simulating exit: {str(e)}")
            return data['bid'].iloc[current_idx]

    def _calculate_simulated_profit(self, entry, exit, lot_size, signal):
        """Calculate profit for simulated trade."""
        try:
            if signal > 0:
                return (exit - entry) * lot_size * 100000
            return (entry - exit) * lot_size * 100000
        except Exception as e:
            logging.error(f"Error calculating simulated profit: {str(e)}")
            return 0

# System Health Monitor
class HealthMonitor:
    def __init__(self):
        self.last_heartbeat = time.time()
        
    async def check_health(self):
        """Perform periodic health checks."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat > HEARTBEAT_INTERVAL:
                    if ensure_mt5_connection():
                        account_info = mt5.account_info()
                        if account_info:
                            logging.info(f"Heartbeat: Account {account_info.login}, Balance={account_info.balance}")
                        else:
                            logging.warning("Heartbeat failed: Unable to retrieve account info")
                    self.last_heartbeat = current_time
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"Health check error: {str(e)}")
                await asyncio.sleep(60)

# Main Trading System
class HFTSystem:
    def __init__(self):
        self.data_fetcher = DataFetcher(SYMBOLS, TIMEFRAME)
        self.models = load_models(SYMBOLS)
        self.signal_generator = SignalGenerator(self.models)
        self.trade_executor = TradeExecutor(BASE_LOT_SIZE)
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager()
        self.performance_tracker = PerformanceTracker()
        self.backtester = Backtester(self.data_fetcher, self.signal_generator, self.trade_executor, self.risk_manager)
        self.health_monitor = HealthMonitor()
        self.running = True

    async def initialize(self):
        """Initialize the trading system."""
        setup_logging()
        logging.info("Initializing HFT system...")
        
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logging.error("Failed to initialize MT5 connection")
            return False
            
        account_info = mt5.account_info()
        if not account_info:
            logging.error("Failed to retrieve account info")
            return False
            
        logging.info(f"Connected to MT5: Account {account_info.login}, Balance={account_info.balance}")
        
        # Run initial backtest
        start_date = datetime.utcnow() - timedelta(days=365)
        end_date = datetime.utcnow()
        backtest_results = await self.backtester.run_backtest(start_date, end_date)
        if backtest_results['trades'] == 0:
            logging.warning("Backtest completed with no trades. Check data or strategy.")
            
        return True

    async def run(self):
        """Run the main trading loop."""
        if not await self.initialize():
            return
            
        # Start health monitor
        asyncio.create_task(self.health_monitor.check_health())
        
        logging.info("Starting live trading loop...")
        while self.running:
            try:
                if not self.risk_manager.check_global_risk():
                    logging.warning("Risk limits exceeded. Pausing for 60 seconds.")
                    await asyncio.sleep(60)
                    continue
                    
                data = await self.data_fetcher.fetch_data()
                if not data or all(df.empty for df in data.values()):
                    logging.warning("No valid data fetched. Retrying in 1 second.")
                    await asyncio.sleep(1)
                    continue
                    
                signals = self.signal_generator.generate_signals(data)
                if not any(signals.values()):
                    await asyncio.sleep(0.05)
                    continue
                    
                tasks = []
                for symbol, signal in signals.items():
                    if signal != 0:
                        volatility = compute_volatility(data[symbol]['bid'].tail(50))
                        lot_size = self.risk_manager.adjust_lot_size(BASE_LOT_SIZE, volatility)
                        tasks.append(self._process_trade(symbol, signal, lot_size, data[symbol]))
                        
                if tasks:
                    await asyncio.gather(*tasks)
                    
                await self.order_manager.manage_open_orders()
                self.performance_tracker.update_metrics()
                await asyncio.sleep(0.05)  # High-frequency loop
                
            except Exception as e:
                logging.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(1)

    async def _process_trade(self, symbol, signal, lot_size, data):
        """Process a single trade."""
        try:
            success = await self.trade_executor.execute_trade(symbol, signal, lot_size)
            if success:
                entry_price = data['bid'].iloc[-1]
                self.performance_tracker.record_trade(symbol, signal, lot_size, entry_price)
        except Exception as e:
            logging.error(f"Error processing trade for {symbol}: {str(e)}")

    async def shutdown(self):
        """Gracefully shut down the system."""
        self.running = False
        logging.info("Initiating system shutdown...")
        
        try:
            positions = mt5.positions_get()
            if positions:
                logging.info(f"Closing {len(positions)} open positions")
                for pos in positions:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "position": pos.ticket,
                        "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                        "comment": "Shutdown Close"
                    }
                    mt5.order_send(request)
                    
            mt5.shutdown()
            logging.info("MT5 connection closed")
            self._log_final_state()
        except Exception as e:
            logging.error(f"Error during shutdown: {str(e)}")

    def _log_final_state(self):
        """Log final system state."""
        total_trades = self.performance_tracker.win_count + self.performance_tracker.loss_count
        win_rate = self.performance_tracker.win_count / total_trades if total_trades > 0 else 0
        logging.info(f"Final State: Total Profit={self.performance_tracker.total_profit:.2f}, "
                    f"Total Trades={total_trades}, Win Rate={win_rate:.2%}")

# Main Entry Point
async def main():
    system = HFTSystem()
    try:
        await system.run()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
        await system.shutdown()
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())