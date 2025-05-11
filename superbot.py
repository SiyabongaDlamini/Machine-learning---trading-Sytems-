import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import talib as ta
import telegram
import os
import pickle
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
import logging
import uuid
import dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

class EnhancedChozenTradingBot:
    def __init__(self, symbol, initial_balance=1000, risk_percent=2, reward_ratio=2, max_trades=5):
        """Initialize the trading bot with given parameters."""
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent / 100
        self.reward_ratio = reward_ratio
        self.max_trades = max_trades
        self.use_trailing_stop = True
        self.trailing_stop_activation = 0.005  # 0.5% profit
        self.trailing_stop_distance = 0.003    # 0.3% distance
        self.breakeven_activation = 0.007      # 0.7% profit
        self.use_ai_prediction = True
        self.telegram_notifications = bool(TELEGRAM_TOKEN != "YOUR_TELEGRAM_TOKEN" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID")
        
        # Model paths
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.rf_model_path = os.path.join(self.model_dir, "rf_model.pkl")
        self.gb_model_path = os.path.join(self.model_dir, "gb_model.pkl")
        self.lstm_model_path = os.path.join(self.model_dir, "lstm_model")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        
        # Trading stats
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_balance = initial_balance
        
        # Initialize MT5
        logger.info("Initializing MT5...")
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            raise Exception("MT5 initialization failed")
        
        # Verify symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol {self.symbol} not found or not visible")
            if not mt5.symbol_select(self.symbol, True):
                mt5.shutdown()
                raise Exception(f"Symbol {self.symbol} selection failed")
        
        self.point = symbol_info.point
        self.digits = symbol_info.digits
        
        # Initialize ML models
        self.rf_model = None
        self.gb_model = None
        self.lstm_model = None
        self.scaler = None
        self.load_or_train_models()
        
        # Initialize Telegram
        if self.telegram_notifications:
            try:
                self.telegram_bot = telegram.Bot(token=TELEGRAM_TOKEN)
                self.send_telegram_message("🚀 Enhanced Chozen 3.0 Trading Bot initialized!")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.telegram_notifications = False
        
        logger.info(f"Bot initialized for {self.symbol}")

    def load_or_train_models(self):
        """Load existing ML models or train new ones."""
        try:
            self.rf_model = joblib.load(self.rf_model_path)
            self.gb_model = joblib.load(self.gb_model_path)
            self.lstm_model = load_model(self.lstm_model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("AI models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}. Training new models.")
            self.train_models()

    def train_models(self):
        """Train ML models on historical data."""
        logger.info("Training AI models...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        end_timestamp = int(end_date.timestamp())
        start_timestamp = int(start_date.timestamp())
        
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_H1, start_timestamp, end_timestamp)
        if rates is None or len(rates) < 100:
            logger.error("Insufficient historical data for training")
            return
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = self.prepare_features(df)
        
        df['future_price'] = df['close'].shift(-24)
        df['target'] = (df['future_price'] > df['close'] * 1.01).astype(int)
        df.dropna(inplace=True)
        
        X = df.drop(['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'future_price', 'target'], axis=1)
        y = df['target']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_scaled, y)
        joblib.dump(self.rf_model, self.rf_model_path)
        
        # Train Gradient Boosting
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.gb_model.fit(X_scaled, y)
        joblib.dump(self.gb_model, self.gb_model_path)
        
        # Train LSTM
        sequence_length = 24
        X_lstm = []
        y_lstm = []
        for i in range(sequence_length, len(X_scaled)):
            X_lstm.append(X_scaled[i-sequence_length:i])
            y_lstm.append(y.iloc[i])
        
        if X_lstm:
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            self.lstm_model.fit(
                X_lstm, y_lstm,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=0
            )
            self.lstm_model.save(self.lstm_model_path)
        
        logger.info("AI models trained and saved")

    def prepare_features(self, df):
        """Prepare technical indicators and features for ML models."""
        df['sma_5'] = ta.SMA(df['close'], timeperiod=5)
        df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
        df['sma_200'] = ta.SMA(df['close'], timeperiod=200)
        df['ema_5'] = ta.EMA(df['close'], timeperiod=5)
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = ta.MACD(df['close'])
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['price_sma_ratio_20'] = df['close'] / df['sma_20']
        df['volatility_20'] = df['close'].rolling(window=20).std() / df['close'] * 100
        df['trend_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df.fillna(method='ffill', inplace=True)
        return df

    def send_telegram_message(self, message):
        """Send a message via Telegram."""
        if not self.telegram_notifications:
            return
        try:
            self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def get_current_price(self):
        """Get the current market price with fallbacks."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick and tick.last > 0:
            return tick.last
        if tick and (tick.bid > 0 or tick.ask > 0):
            return (tick.bid + tick.ask) / 2 if tick.bid > 0 and tick.ask > 0 else tick.bid or tick.ask
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates and len(rates) > 0:
            return rates[0]['close']
        logger.error("Cannot get current price. Using last known price.")
        return self.get_current_price() if hasattr(self, 'last_price') else 1.0

    def get_market_data(self, timeframe, lookback=500):
        """Fetch market data and compute indicators."""
        tf_map = {'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, lookback)
        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get {timeframe} data")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return self.prepare_features(df)

    def check_trade_status(self, ticket):
        """Check the status of a trade."""
        deals = mt5.history_deals_get(ticket=ticket)
        if not deals:
            return "OPEN", 0
        profit = sum(deal.profit for deal in deals)
        self.trades_count += 1
        if profit >= 0:
            self.winning_trades += 1
            self.current_balance += profit
            self.send_telegram_message(f"✅ WINNING TRADE #{ticket}: Profit ${profit:.2f}")
            return "WIN", profit
        else:
            self.losing_trades += 1
            self.current_balance += profit
            self.send_telegram_message(f"❌ LOSING TRADE #{ticket}: Loss ${profit:.2f}")
            return "LOSS", profit

    def check_news_events(self):
        """Simulate checking for high-impact news events."""
        now = datetime.now()
        events = [
            {"time": now + timedelta(hours=2), "currency": "USD", "event": "Non-Farm Payrolls", "impact": "high"},
            {"time": now + timedelta(minutes=45), "currency": "GBP", "event": "PMI", "impact": "medium"}
        ]
        return [e for e in events if (e["time"] - now).total_seconds() <= 10800 and e["impact"] == "high"]

    def detect_market_structure(self, timeframe="H1"):
        """Detect market structure (trend, range, breakout)."""
        df = self.get_market_data(timeframe, lookback=100)
        if df is None:
            return "unknown", 0
        sma_20, sma_50, sma_200 = df['sma_20'].iloc[-1], df['sma_50'].iloc[-1], df['sma_200'].iloc[-1]
        current_price = self.get_current_price()
        if current_price > sma_20 > sma_50 > sma_200:
            return "uptrend", 0.8
        elif current_price < sma_20 < sma_50 < sma_200:
            return "downtrend", 0.8
        range_size = (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / current_price
        if range_size < 0.03:
            return "range", range_size
        return "weak_trend", 0.5

    def get_key_levels(self):
        """Identify support and resistance levels."""
        h1_data = self.get_market_data("H1", lookback=200)
        if h1_data is None:
            return [], []
        supports, resistances = [], []
        current_price = self.get_current_price()
        sma_200 = h1_data['sma_200'].iloc[-1]
        if sma_200 < current_price:
            supports.append({'level': sma_200, 'strength': 8})
        else:
            resistances.append({'level': sma_200, 'strength': 8})
        prev_day = h1_data.iloc[-24:-1]
        supports.append({'level': prev_day['low'].min(), 'strength': 7})
        resistances.append({'level': prev_day['high'].max(), 'strength': 7})
        return supports, resistances

    def detect_pattern(self, timeframe="H1"):
        """Detect candlestick patterns."""
        df = self.get_market_data(timeframe, lookback=10)
        if df is None:
            return []
        patterns = []
        candles = df.iloc[-5:]
        for i in range(1, len(candles)):
            prev, curr = candles.iloc[i-1], candles.iloc[i]
            prev_body = abs(prev['close'] - prev['open'])
            curr_body = abs(curr['close'] - curr['open'])
            if curr_body > prev_body:
                if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['open'] <= prev['close'] and curr['close'] >= prev['open']:
                    patterns.append({"pattern": "bullish_engulfing", "strength": 7})
                elif prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] >= prev['close'] and curr['close'] <= prev['open']:
                    patterns.append({"pattern": "bearish_engulfing", "strength": 7})
        return patterns

    def check_price_action(self, timeframe="H1"):
        """Analyze price action patterns."""
        df = self.get_market_data(timeframe, lookback=50)
        if df is None:
            return []
        signals = []
        highs, lows = [], []
        for i in range(2, len(df)-2):
            if df['high'].iloc[i] > max(df['high'].iloc[i-2:i]) and df['high'].iloc[i] > max(df['high'].iloc[i+1:i+3]):
                highs.append(df['high'].iloc[i])
            if df['low'].iloc[i] < min(df['low'].iloc[i-2:i]) and df['low'].iloc[i] < min(df['low'].iloc[i+1:i+3]):
                lows.append(df['low'].iloc[i])
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                signals.append({"pattern": "higher_highs_higher_lows", "strength": 7})
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                signals.append({"pattern": "lower_highs_lower_lows", "strength": 7})
        return signals

    def generate_trade_signals(self):
        """Generate trade signals based on multiple analyses."""
        current_price = self.get_current_price()
        market_structure, _ = self.detect_market_structure()
        supports, resistances = self.get_key_levels()
        h1_patterns = self.detect_pattern("H1")
        price_action = self.check_price_action()
        news_events = self.check_news_events()
        
        buy_score, sell_score = 50, 50
        if market_structure == "uptrend":
            buy_score += 15
        elif market_structure == "downtrend":
            sell_score += 15
        
        for pattern in h1_patterns:
            if pattern["pattern"] == "bullish_engulfing":
                buy_score += pattern["strength"]
            elif pattern["pattern"] == "bearish_engulfing":
                sell_score += pattern["strength"]
        
        for signal in price_action:
            if signal["pattern"] == "higher_highs_higher_lows":
                buy_score += signal["strength"]
            elif signal["pattern"] == "lower_highs_lower_lows":
                sell_score += signal["strength"]
        
        if news_events:
            buy_score *= 0.7
            sell_score *= 0.7
        
        buy_signal = buy_score > 70 and buy_score > sell_score + 20
        sell_signal = sell_score > 70 and sell_score > buy_score + 20
        buy_confidence = (buy_score - 50) / 50 if buy_score > 50 else 0
        sell_confidence = (sell_score - 50) / 50 if sell_score > 50 else 0
        
        buy_sl = min(s['level'] for s in supports) * 0.995 if supports else current_price * 0.98
        buy_tp = min(r['level'] for r in resistances) if resistances else current_price + (self.reward_ratio * (current_price - buy_sl))
        sell_sl = max(r['level'] for r in resistances) * 1.005 if resistances else current_price * 1.02
        sell_tp = max(s['level'] for s in supports) if supports else current_price - (self.reward_ratio * (sell_sl - current_price))
        
        return {
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "buy_confidence": buy_confidence,
            "sell_confidence": sell_confidence,
            "buy_stop_loss": buy_sl,
            "buy_take_profit": buy_tp,
            "sell_stop_loss": sell_sl,
            "sell_take_profit": sell_tp,
            "current_price": current_price,
            "market_structure": market_structure,
            "timeframe": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def open_trade(self, order_type, price, sl, tp, confidence):
        """Open a new trade."""
        lot_size = self.calculate_lot_size(price, sl)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Trade_{uuid.uuid4()}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to open trade: {result.retcode}")
            return None
        logger.info(f"Opened trade: Ticket #{result.order}")
        return result.order

    def calculate_lot_size(self, price, sl):
        """Calculate lot size based on risk percentage."""
        risk_amount = self.current_balance * self.risk_percent
        pip_value = abs(price - sl) / self.point
        if pip_value == 0:
            return 0.01
        lot_size = risk_amount / (pip_value * 10)  # Assuming 10 USD per pip for standard lot
        return max(0.01, round(lot_size, 2))

    def modify_trade(self, ticket, new_sl, new_tp):
        """Modify trade's stop loss and take profit."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position #{ticket} not found")
            return False
        position = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": new_sl,
            "tp": new_tp
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify trade #{ticket}: {result.retcode}")
            return False
        logger.info(f"Modified trade #{ticket}: SL={new_sl}, TP={new_tp}")
        return True

    def smart_trailing_stop(self, ticket, initial_trail_points=50):
        """Apply adaptive trailing stop."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        position = position[0]
        current_price = self.get_current_price()
        h1_data = self.get_market_data("H1", 20)
        adaptive_trail = initial_trail_points
        if h1_data is not None:
            atr = h1_data['atr'].iloc[-1]
            adaptive_trail = max(int(atr / self.point * 1.5), initial_trail_points)
        
        profit_pips = ((current_price - position.price_open) / self.point if position.type == mt5.POSITION_TYPE_BUY
                       else (position.price_open - current_price) / self.point)
        if profit_pips <= adaptive_trail * 0.5:
            return False
        
        new_sl = (current_price - (adaptive_trail * self.point) if position.type == mt5.POSITION_TYPE_BUY
                  else current_price + (adaptive_trail * self.point))
        if position.sl == 0 or (new_sl > position.sl if position.type == mt5.POSITION_TYPE_BUY else new_sl < position.sl):
            return self.modify_trade(ticket, new_sl, position.tp)
        return False

    def run(self):
        """Main trading loop."""
        logger.info("Starting Enhanced Chozen 3.0 Trading Bot...")
        self.send_telegram_message("🚀 Bot started!")
        active_trades = {}
        signal_cooldown = {'buy': [], 'sell': []}
        
        while True:
            try:
                current_price = self.get_current_price()
                account_info = mt5.account_info()
                if account_info:
                    self.current_balance = account_info.balance
                
                # Periodic stats update
                current_time = datetime.now()
                if current_time.minute % 15 == 0 and current_time.second < 10:
                    stats = (f"📊 Stats: Balance=${self.current_balance:.2f}, "
                             f"Trades={self.trades_count}, Wins={self.winning_trades}, "
                             f"Losses={self.losing_trades}, Active={len(active_trades)}")
                    logger.info(stats)
                    self.send_telegram_message(stats)
                
                # Manage active trades
                for ticket in list(active_trades.keys()):
                    position = mt5.positions_get(ticket=ticket)
                    if not position:
                        status, profit = self.check_trade_status(ticket)
                        if status != "OPEN":
                            logger.info(f"Trade #{ticket} closed: {status}, Profit=${profit:.2f}")
                            del active_trades[ticket]
                        continue
                    position = position[0]
                    trade_info = active_trades[ticket]
                    
                    profit_pips = ((current_price - position.price_open) / self.point if position.type == mt5.POSITION_TYPE_BUY
                                   else (position.price_open - current_price) / self.point)
                    
                    if profit_pips >= 50 and not trade_info.get("breakeven_set", False):
                        new_sl = position.price_open + (2 * self.point if position.type == mt5.POSITION_TYPE_BUY else -2 * self.point)
                        if self.modify_trade(ticket, new_sl, position.tp):
                            trade_info["breakeven_set"] = True
                            active_trades[ticket] = trade_info
                    
                    if self.use_trailing_stop and profit_pips > 70:
                        self.smart_trailing_stop(ticket)
                
                # Generate and process signals
                signals = self.generate_trade_signals()
                current_key = f"{signals['market_structure']}_{signals['timeframe'][:13]}"
                
                if signals['buy_signal'] and current_key not in signal_cooldown['buy'] and len(active_trades) < self.max_trades:
                    logger.info(f"Processing BUY signal: Confidence={signals['buy_confidence']:.2f}")
                    ticket = self.open_trade(
                        mt5.ORDER_TYPE_BUY,
                        current_price,
                        signals['buy_stop_loss'],
                        signals['buy_take_profit'],
                        signals['buy_confidence']
                    )
                    if ticket:
                        active_trades[ticket] = {
                            "type": "BUY",
                            "entry": current_price,
                            "stop_loss": signals['buy_stop_loss'],
                            "take_profit": signals['buy_take_profit'],
                            "timestamp": datetime.now(),
                            "confidence": signals['buy_confidence']
                        }
                        signal_cooldown['buy'].append(current_key)
                        if len(signal_cooldown['buy']) > 10:
                            signal_cooldown['buy'].pop(0)
                
                if signals['sell_signal'] and current_key not in signal_cooldown['sell'] and len(active_trades) < self.max_trades:
                    logger.info(f"Processing SELL signal: Confidence={signals['sell_confidence']:.2f}")
                    ticket = self.open_trade(
                        mt5.ORDER_TYPE_SELL,
                        current_price,
                        signals['sell_stop_loss'],
                        signals['sell_take_profit'],
                        signals['sell_confidence']
                    )
                    if ticket:
                        active_trades[ticket] = {
                            "type": "SELL",
                            "entry": current_price,
                            "stop_loss": signals['sell_stop_loss'],
                            "take_profit": signals['sell_take_profit'],
                            "timestamp": datetime.now(),
                            "confidence": signals['sell_confidence']
                        }
                        signal_cooldown['sell'].append(current_key)
                        if len(signal_cooldown['sell']) > 10:
                            signal_cooldown['sell'].pop(0)
                
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.send_telegram_message(f"⚠️ Error: {str(e)}")
                time.sleep(30)

    def shutdown(self):
        """Shut down the bot and provide final statistics."""
        logger.info("Shutting down Enhanced Chozen 3.0 Trading Bot...")
        
        # Close open positions
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            logger.info(f"Closing {len(positions)} open positions...")
            for pos in positions:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": self.get_current_price(),
                    "deviation": 20,
                    "magic": 123456,
                    "comment": "Close on shutdown",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to close position #{pos.ticket}: {result.retcode}")
                else:
                    logger.info(f"Position #{pos.ticket} closed")
        
        # Generate statistics
        profit_loss = self.current_balance - self.initial_balance
        profit_percentage = (profit_loss / self.initial_balance) * 100
        stats = (
            f"========== TRADING STATISTICS ==========\n"
            f"Initial Balance: ${self.initial_balance:.2f}\n"
            f"Final Balance: ${self.current_balance:.2f}\n"
            f"Profit/Loss: ${profit_loss:.2f} ({profit_percentage:.2f}%)\n"
            f"Total Trades: {self.trades_count}\n"
            f"Winning Trades: {self.winning_trades}\n"
            f"Losing Trades: {self.losing_trades}\n"
        )
        if self.trades_count > 0:
            win_rate = (self.winning_trades / self.trades_count) * 100
            stats += f"Win Rate: {win_rate:.2f}%\n"
        
        logger.info(stats)
        self.send_telegram_message(f"🏁 BOT SHUTDOWN\n\n{stats}")
        
        mt5.shutdown()
        logger.info("MT5 connection closed. Bot shutdown complete.")

if __name__ == "__main__":
    bot = EnhancedChozenTradingBot(symbol="BTCUSDm", initial_balance=1000, risk_percent=2, reward_ratio=2, max_trades=5)
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        bot.shutdown()