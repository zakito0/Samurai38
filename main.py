import asyncio
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import ccxt
from dotenv import load_dotenv

# Import our modules
from data_handler import DataHandler
from ai_analyzer import MarketSentimentAnalyzer, AITrader, MarketDataset, LSTMModel
from strategies import (
    Action, MovingAverageCrossover, RSIStrategy, 
    MACDStrategy, CompositeStrategy, StrategyResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the trading agent for Binance Testnet Spot"""
        self.config = {}
        self.load_config(config_path)
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])  # Default symbols for spot trading
        self._last_symbol_refresh = 0.0
        
        # Initialize exchange connection
        if not self.setup_exchange():
            raise RuntimeError("Failed to initialize exchange connection. Check logs for details.")
        self.refresh_symbol_universe(force=True)
            
        # Initialize data handler with the exchange instance
        self.data_handler = DataHandler(exchange=self.exchange)
        self.setup_strategies()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.ai_trader = None
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.balance = {}
        
        # Load or initialize AI model
        self.load_ai_model()
        
        logger.info("Trading agent initialized successfully for Binance Testnet Spot")
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "1h",
                "max_position_size": 0.1,  # 10% of balance per position
                "stop_loss_pct": 0.05,     # 5% stop loss
                "take_profit_pct": 0.1,    # 10% take profit
                "trading_enabled": False,  # Paper trading by default
                "paper_balance": 10000.0,  # Starting paper balance in USDT
                "risk_free_rate": 0.02,    # 2% annual risk-free rate
                "max_drawdown_pct": 0.2,   # 20% max drawdown
                "data_refresh_interval": 300,  # 5 minutes
                "quote_asset": "USDT",
                "use_all_quote_pairs": False
            }

    def _get_quote_asset(self) -> str:
        return str(self.config.get('quote_asset', 'USDT')).upper()

    def refresh_symbol_universe(self, force: bool = False):
        """Refresh the tradable symbol list based on config and exchange markets."""
        configured_symbols = self.config.get('symbols') or []
        if configured_symbols:
            self.symbols = configured_symbols
            return

        quote_asset = self._get_quote_asset()
        use_all_pairs = self.config.get('use_all_quote_pairs', False)

        if not use_all_pairs or not hasattr(self, 'exchange'):
            defaults = [f"BTC/{quote_asset}", f"ETH/{quote_asset}"]
            self.symbols = defaults
            return

        refresh_interval = self.config.get('symbol_refresh_interval', 1800)
        now = time.time()
        if not force and self._last_symbol_refresh and (now - self._last_symbol_refresh) < refresh_interval:
            if self.symbols:
                return

        try:
            markets = self.exchange.load_markets()
            dynamic_symbols = sorted([
                symbol for symbol, market in markets.items()
                if isinstance(symbol, str)
                and symbol.upper().endswith(f"/{quote_asset}")
                and market.get('active', True)
            ])

            if dynamic_symbols:
                self.symbols = dynamic_symbols
                self._last_symbol_refresh = now
                logger.info(
                    f"Loaded {len(dynamic_symbols)} {quote_asset}-quoted symbols from Binance"
                )
                return
            logger.warning(f"No active {quote_asset}-quoted markets were returned by Binance")
        except Exception as err:
            logger.warning(f"Unable to refresh symbol universe: {err}")

        if not self.symbols:
            fallback = [f"BTC/{quote_asset}", f"ETH/{quote_asset}"]
            logger.warning(
                "Falling back to default symbols %s due to missing market data", fallback
            )
            self.symbols = fallback
    
    def setup_strategies(self):
        """Initialize trading strategies"""
        # Individual strategies
        ma_crossover = MovingAverageCrossover(short_window=20, long_window=50)
        rsi_strategy = RSIStrategy(rsi_period=14, overbought=70, oversold=30)
        macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        
        # Combine strategies with weights
        self.strategy = CompositeStrategy([
            (ma_crossover, 0.4),
            (rsi_strategy, 0.3),
            (macd_strategy, 0.3)
        ])
    
    def setup_exchange(self):
        """Initialize the exchange connection with Binance Spot (mainnet)"""
        try:
            # Load environment variables from .env file
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            logger.info(f"Loading .env file from: {env_path}")
            
            # Verify .env file exists and is readable
            if not os.path.exists(env_path):
                logger.error(f"Error: .env file not found at {env_path}")
                raise FileNotFoundError(f".env file not found at {env_path}")
            
            logger.info(".env file found, loading environment variables...")
            load_dotenv(env_path, override=True)
            
            # Get API keys from environment
            api_key = os.getenv('BINANCE_API_KEY', '').strip()
            api_secret = os.getenv('BINANCE_SECRET_KEY', '').strip()
            
            # Debug log the environment variables (show first 4 chars for security)
            logger.info(f"API Key present: {'Yes' if api_key else 'No'}")
            if api_key:
                logger.info(f"API Key starts with: {api_key[:4]}...")
            logger.info(f"API Secret present: {'Yes' if api_secret else 'No'}")
            
            if not api_key or not api_secret:
                error_msg = "API key or secret is missing or empty in .env file"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Initialize exchange with Binance mainnet config for Spot
            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                    'warnOnFetchCurrenciesWithoutPermission': False,
                    'fetchCurrencies': False,
                },
                'urls': {
                    'api': {
                        'public': 'https://api.binance.com/api/v3',
                        'private': 'https://api.binance.com/api/v3',
                        'sapi': 'https://api.binance.com/sapi/v1',
                    }
                },
                'timeout': 30000,  # 30 seconds timeout
                'verbose': True,   # Enable verbose output for debugging
            }
            
            logger.info("Initializing Binance Spot exchange (mainnet)...")
            self.exchange = ccxt.binance(exchange_config)
            
            # Test the connection with a simple public API call first
            logger.info("Testing public API connection...")
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"Successfully connected to Binance Spot. Current BTC/USDT price: {ticker['last']}")
            
            # Now test private API
            logger.info("Testing private API access...")
            balance = self.exchange.fetch_balance()
            if 'free' in balance and 'USDT' in balance['free']:
                usdt_balance = float(balance['free']['USDT'])
                logger.info(f"Account balance: {usdt_balance:.2f} USDT")
            else:
                logger.warning("Could not fetch USDT balance. Available balances:")
                for currency, amount in balance['total'].items():
                    if amount > 0:
                        logger.info(f"{currency}: {amount}")
            
            return True
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            logger.error("Please verify your API keys and ensure they're for Binance Testnet")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
        
        logger.error("Failed to connect to Binance Spot. Please check the following:")
        logger.error("1. Your .env file contains valid Binance mainnet API keys")
        logger.error("2. The API keys have the correct permissions (enable reading and trading)")
        logger.error("3. Your system clock is synchronized")
        logger.error("4. You're not behind a proxy that might block the connection")
        logger.error("5. The API keys are not restricted by IP")
        return False
    
    def load_ai_model(self):
        """Load or initialize the AI model"""
        model_path = 'models/ai_trader.pth'
        input_size = 10  # Number of features (adjust based on your feature set)
        
        self.ai_trader = AITrader(input_size)
        
        if os.path.exists(model_path):
            try:
                self.ai_trader.load_model(model_path)
                logger.info("Loaded pre-trained AI model")
            except Exception as e:
                logger.error(f"Error loading AI model: {e}")
    
    async def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch market data for a symbol"""
        try:
            df = await self.data_handler.fetch_ohlcv(symbol, timeframe, limit)
            if df is not None and not df.empty:
                df = self.data_handler.add_technical_indicators(df)
            return df
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions"""
        try:
            # Fetch market data
            df = await self.fetch_market_data(symbol, self.config['timeframe'])
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return {}
            
            # Get strategy signals
            strategy_result = self.strategy.analyze(df)
            
            # Get sentiment from news/social media (simplified example)
            news_sentiment = await self.analyze_news_sentiment(symbol)
            
            # Get AI recommendation if AI trader is available
            ai_recommendation = None
            if self.ai_trader is not None:
                try:
                    # Prepare features for AI model
                    features = self.prepare_features_for_ai(df)
                    if not features.empty:
                        ai_recommendation = self.ai_trader.predict(features.iloc[-1:])  # Get prediction for latest data point
                except Exception as e:
                    logger.error(f"Error getting AI recommendation: {e}", exc_info=True)
            
            # Prepare analysis result
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'price': float(df['close'].iloc[-1]),
                'action': strategy_result.action.name if strategy_result else 'HOLD',
                'confidence': float(strategy_result.confidence) if strategy_result else 0.5,
                'indicators': strategy_result.indicators if strategy_result else {},
                'sentiment': news_sentiment,
            }
            
            # Get AI recommendation if AI trader is available
            if self.ai_trader is not None:
                try:
                    # Prepare features for AI model
                    features = self.prepare_features_for_ai(df)
                    if not features.empty:
                        # Get the last row of features as a numpy array
                        last_features = features.iloc[-1:].values  # This is now a 2D array with shape (1, n_features)
                        
                        # Convert to tensor and add batch and sequence dimensions if needed
                        features_tensor = torch.FloatTensor(last_features).unsqueeze(0)  # Shape: (1, 1, n_features)
                        
                        # Get prediction
                        ai_prediction = self.ai_trader.predict(features_tensor)
                        
                        # Get the action with highest probability
                        action_idx = np.argmax([ai_prediction['buy_prob'], 
                                              ai_prediction['sell_prob'], 
                                              ai_prediction['hold_prob']])
                        
                        analysis['ai_recommendation'] = {
                            'action': ['BUY', 'SELL', 'HOLD'][action_idx],
                            'confidence': max(ai_prediction.values()),
                            'probabilities': ai_prediction
                        }
                except Exception as e:
                    logger.error(f"AI prediction error: {e}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return {}
    
    async def analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from news and social media"""
        try:
            # In a real implementation, you would fetch news/articles here
            # This is a simplified example
            news_text = f"{symbol} shows strong growth potential according to latest market analysis."
            sentiment = await self.sentiment_analyzer.analyze_sentiment(news_text)
            return {
                'score': sentiment['score'],
                'sentiment': sentiment['sentiment'],
                'sources': ['example_news']
            }
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'score': 0.5, 'sentiment': 'NEUTRAL', 'sources': []}
    
    def prepare_features_for_ai(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for AI model
        
        Returns:
            DataFrame with exactly 10 features as expected by the LSTM model
        """
        try:
            features = pd.DataFrame()
            
            # 1. Price and volume features
            if 'close' in df.columns:
                features['close'] = df['close']
            if 'volume' in df.columns:
                features['volume'] = df['volume']
            
            # 2. Technical indicators
            if 'RSI' in df.columns:
                features['rsi'] = df['RSI']
            
            # 3. MACD and Signal Line
            if 'MACD' in df.columns and 'Signal_Line' in df.columns:
                features['macd'] = df['MACD']
                features['signal'] = df['Signal_Line']
                features['macd_hist'] = df['MACD'] - df['Signal_Line']
            
            # 4. Moving Averages
            if 'SMA_20' in df.columns:
                features['sma_20'] = df['SMA_20']
            if 'SMA_50' in df.columns:
                features['sma_50'] = df['SMA_50']
            
            # 5. Bollinger Bands
            if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
                features['bb_upper'] = df['BB_upper']
                features['bb_middle'] = df['BB_middle']
                features['bb_lower'] = df['BB_lower']
                features['bb_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # 6. Price changes
            if 'close' in df.columns:
                features['price_change_1'] = df['close'].pct_change(1)
                features['price_change_5'] = df['close'].pct_change(5)
            
            # 7. Volume changes
            if 'volume' in df.columns:
                features['volume_change'] = df['volume'].pct_change()
            
            # Drop any rows with NaN values
            features = features.dropna()
            
            # Ensure we have exactly 10 features (add zeros if needed)
            required_features = 10
            current_features = len(features.columns)
            
            if current_features < required_features:
                # Add zero columns if we don't have enough features
                for i in range(required_features - current_features):
                    features[f'feature_{i}'] = 0.0
            elif current_features > required_features:
                # Select only the first 10 features if we have too many
                features = features.iloc[:, :required_features]
            
            if len(features) == 0:
                logger.warning("No valid features could be prepared for AI model")
                return pd.DataFrame()
                
            return features.iloc[:, :10]  # Ensure exactly 10 features
            
        except Exception as e:
            logger.error(f"Error preparing features for AI: {e}")
            return pd.DataFrame()
    
    async def get_balance(self, symbol: str) -> float:
        """Get available balance for a symbol"""
        try:
            # Use the synchronous fetch_balance() since we're using the sync CCXT client
            if not hasattr(self, 'exchange') or not self.exchange:
                logger.error("Exchange not initialized")
                return 0.0
                
            # Make sure we're not in a request loop
            if self.exchange.rateLimit > 0:
                import time
                time.sleep(self.exchange.rateLimit / 1000)  # Convert ms to seconds
                
            # Fetch balance synchronously
            balance = self.exchange.fetch_balance()
            
            if symbol.endswith('USDT'):
                return float(balance.get('USDT', {}).get('free', 0.0))
            else:
                base_currency = symbol.split('/')[0]
                return float(balance.get(base_currency, {}).get('free', 0.0))
                
        except Exception as e:
            logger.error(f"Error getting balance for {symbol}: {e}", exc_info=True)
            return 0.0

    async def create_order(self, symbol: str, side: str, order_type: str, amount: float, 
                         price: float = None, stop_price: float = None, params: dict = None) -> dict:
        """
        Create an order on the exchange with enhanced order types
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop_loss', 'take_profit', 'stop_loss_limit', 'take_profit_limit', 'trailing_stop'
            amount: Amount to buy/sell
            price: Price for limit orders (required for limit, stop_loss_limit, take_profit_limit)
            stop_price: Stop price for stop-loss/take-profit orders
            params: Additional order parameters
            
        Returns:
            dict: Order details
        """
        try:
            if not self.config.get('trading_enabled', False) and not self.config.get('paper_trading', False):
                logger.info(f"[PAPER TRADE] {side.upper()} {amount} {symbol} @ {price if price else 'MARKET'}")
                return {'id': 'PAPER_TRADE', 'status': 'closed', 'filled': amount, 'symbol': symbol, 'side': side}
            
            order_params = params or {}
            order_params['newOrderRespType'] = 'FULL'  # Get full order details in response
            
            # Handle different order types
            order_type = order_type.upper()
            
            # Market Order
            if order_type == 'MARKET':
                return await self.exchange.create_market_order(
                    symbol, 
                    side.lower(), 
                    amount, 
                    params=order_params
                )
            
            # Limit Order
            elif order_type == 'LIMIT':
                if price is None:
                    raise ValueError("Price is required for limit orders")
                
                # Add time in force (GTC by default)
                order_params['timeInForce'] = order_params.get('timeInForce', 'GTC')
                
                return await self.exchange.create_limit_order(
                    symbol, 
                    side.lower(), 
                    amount, 
                    price, 
                    params=order_params
                )
            
            # Stop-Loss Order (Market)
            elif order_type == 'STOP_LOSS':
                if stop_price is None:
                    raise ValueError("Stop price is required for stop-loss orders")
                
                order_params['stopPrice'] = stop_price
                return await self.exchange.create_order(
                    symbol, 
                    'STOP_LOSS', 
                    side.lower(), 
                    amount, 
                    None,  # price not needed for market stop-loss
                    params=order_params
                )
            
            # Take-Profit Order (Market)
            elif order_type == 'TAKE_PROFIT':
                if stop_price is None:
                    raise ValueError("Stop price is required for take-profit orders")
                
                order_params['stopPrice'] = stop_price
                return await self.exchange.create_order(
                    symbol, 
                    'TAKE_PROFIT', 
                    side.lower(), 
                    amount, 
                    None,  # price not needed for market take-profit
                    params=order_params
                )
            
            # Stop-Loss-Limit Order
            elif order_type == 'STOP_LOSS_LIMIT':
                if price is None or stop_price is None:
                    raise ValueError("Both price and stop price are required for stop-loss-limit orders")
                
                order_params['stopPrice'] = stop_price
                order_params['timeInForce'] = order_params.get('timeInForce', 'GTC')
                
                return await self.exchange.create_order(
                    symbol, 
                    'STOP_LOSS_LIMIT', 
                    side.lower(), 
                    amount, 
                    price,
                    params=order_params
                )
            
            # Take-Profit-Limit Order
            elif order_type == 'TAKE_PROFIT_LIMIT':
                if price is None or stop_price is None:
                    raise ValueError("Both price and stop price are required for take-profit-limit orders")
                
                order_params['stopPrice'] = stop_price
                order_params['timeInForce'] = order_params.get('timeInForce', 'GTC')
                
                return await self.exchange.create_order(
                    symbol, 
                    'TAKE_PROFIT_LIMIT', 
                    side.lower(), 
                    amount, 
                    price,
                    params=order_params
                )
            
            # Trailing Stop Order
            elif order_type == 'TRAILING_STOP':
                if 'trailingDelta' not in order_params and 'trailingPercent' not in order_params:
                    raise ValueError("Either 'trailingDelta' or 'trailingPercent' must be specified for trailing stop orders")
                
                return await self.exchange.create_order(
                    symbol,
                    'TRAILING_STOP_MARKET',
                    side.lower(),
                    amount,
                    None,
                    params=order_params
                )
            
            # OCO (One-Cancels-Other) Order
            elif order_type == 'OCO':
                if 'stopPrice' not in order_params or 'stopLimitPrice' not in order_params:
                    raise ValueError("'stopPrice' and 'stopLimitPrice' are required for OCO orders")
                
                return await self.exchange.private_post_order_oco({
                    'symbol': symbol.replace('/', ''),
                    'side': side.upper(),
                    'quantity': amount,
                    'price': str(price),
                    'stopPrice': str(order_params['stopPrice']),
                    'stopLimitPrice': str(order_params['stopLimitPrice']),
                    'stopLimitTimeInForce': order_params.get('timeInForce', 'GTC'),
                    'newOrderRespType': 'FULL'
                })
            
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
        except Exception as e:
            logger.error(f"Error creating {order_type} order for {symbol}: {e}")
            logger.exception("Order creation failed with traceback:")
            raise

    async def execute_trade(self, symbol: str, action: Action, amount: float, 
                          price: float = None, order_type: str = 'MARKET', 
                          stop_price: float = None, params: dict = None) -> dict:
        """
        Execute a trade with advanced order types and position management
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            action: Action enum (BUY/SELL/HOLD)
            amount: Amount to trade
            price: Price for limit orders
            order_type: Type of order ('market', 'limit', 'stop_loss', etc.)
            stop_price: Stop price for stop/trailing orders
            params: Additional order parameters
            
        Returns:
            dict: Order execution details or None if failed
        """
        try:
            if action == Action.HOLD:
                return {'status': 'HOLD', 'message': 'No action taken (HOLD)'}
            
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1] if '/' in symbol else 'USDT'
            side = 'buy' if action == Action.BUY else 'sell'
            
            # Get current balance
            balance_currency = quote_currency if action == Action.BUY else base_currency
            current_balance = await self.get_balance(balance_currency)
            
            # Get current price if not provided for risk calculation
            if price is None and order_type.upper() != 'MARKET':
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
            else:
                current_price = price
            
            # Calculate position size with risk management
            max_risk_amount = current_balance * self.config.get('risk_per_trade', 0.02)  # Default 2% risk per trade
            
            if action == Action.BUY:
                # For buying, calculate max amount based on available quote currency
                max_amount = (current_balance * 0.95) / (current_price or 1)  # Use 95% of available balance
                amount = min(amount, max_amount)
                
                # Additional risk check based on position size
                position_value = amount * (current_price or 1)
                if position_value > max_risk_amount * 2:  # Cap at 2x risk amount
                    amount = (max_risk_amount * 2) / (current_price or 1)
            else:  # SELL
                # For selling, use available base currency balance
                max_amount = current_balance
                amount = min(amount, max_amount)
                
                # Additional risk check for short positions
                if order_type.upper() in ['STOP_LOSS', 'STOP_LOSS_LIMIT'] and stop_price:
                    risk_per_share = (stop_price / (current_price or 1)) - 1
                    if risk_per_share > 0:  # Only if stop is above current price for short
                        max_risk_units = max_risk_amount / (risk_per_share * (current_price or 1))
                        amount = min(amount, max_risk_units)
            
            # Check minimum order size
            min_amount = 0.001  # This should be fetched from exchange info in production
            if amount < min_amount:
                logger.warning(f"Order amount {amount} is below minimum {min_amount} for {symbol}")
                return {'status': 'error', 'message': f'Order amount below minimum {min_amount}'}
            
            # Execute the order
            logger.info(f"Executing {order_type.upper()} {action.name} order for {amount} {symbol} @ {price if price else 'market'}")
            
            # Add order parameters
            order_params = params or {}
            
            # For paper trading, just log and return
            if not self.config.get('trading_enabled', False) and self.config.get('paper_trading', False):
                order_id = f"PAPER_{int(time.time() * 1000)}"
                logger.info(f"[PAPER TRADE] {side.upper()} {amount} {symbol} @ {price if price else 'MARKET'}")
                return {
                    'id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'type': order_type.lower(),
                    'amount': amount,
                    'price': price,
                    'status': 'closed',
                    'filled': amount,
                    'timestamp': int(time.time() * 1000)
                }
            
            # For real trading
            try:
                order = await self.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    stop_price=stop_price,
                    params=order_params
                )
                
                # Log the successful order
                logger.info(f"Successfully placed {order_type.upper()} {action.name} order for {amount} {symbol}")
                
                # Update positions and trade history
                await self.update_positions(order, action)
                
                return order
                
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                logger.exception("Order execution error:")
                return {'status': 'error', 'message': str(e)}
            
        except Exception as e:
            logger.error(f"Error in execute_trade for {symbol}: {e}")
            logger.exception("Trade execution failed with traceback:")
            return {'status': 'error', 'message': str(e)}
    
    async def check_positions(self, symbol: str) -> dict:
        """Check open positions for a symbol"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            return {p['symbol']: p for p in positions if float(p['contracts']) > 0}
        except Exception as e:
            logger.error(f"Error checking positions for {symbol}: {e}")
            return {}

    async def manage_risk(self, symbol: str):
        """
        Advanced risk management for open positions
        Implements dynamic stop-loss, take-profit, and position sizing
        """
        try:
            # Get current positions
            positions = await self.check_positions(symbol)
            if not positions:
                return
            
            # Get current market data
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            for pos_symbol, position in positions.items():
                try:
                    entry_price = float(position.get('entryPrice', 0))
                    size = float(position.get('contracts', 0))
                    side = position.get('side', '').lower()
                    
                    if size <= 0:
                        continue
                    
                    # Calculate current PnL
                    if side == 'long':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:  # short
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # Get volatility-based stop levels (using ATR)
                    df = await self.fetch_market_data(symbol, self.config.get('timeframe', '1h'), limit=14)
                    if not df.empty and 'ATR' in df.columns:
                        atr = df['ATR'].iloc[-1]
                        atr_multiplier = 2.0  # Can be adjusted based on strategy
                        volatility_stop = atr * atr_multiplier
                    else:
                        # Fallback to fixed percentage if ATR not available
                        volatility_stop = entry_price * 0.02  # 2% of entry price
                    
                    # Dynamic stop-loss and take-profit based on volatility and market conditions
                    if side == 'long':
                        # Trailing stop logic
                        trailing_stop_pct = 0.03  # 3% trailing stop
                        current_trailing_stop = current_price * (1 - trailing_stop_pct)
                        
                        # Update trailing stop if price moves up
                        if hasattr(self, f'trailing_stop_{symbol}'):
                            self.trailing_stop = max(getattr(self, f'trailing_stop_{symbol}'), current_trailing_stop)
                        else:
                            self.trailing_stop = current_trailing_stop
                        
                        # Check if we should exit the position
                        if current_price <= self.trailing_stop:
                            logger.info(f"Trailing stop triggered for {symbol} at {current_price}")
                            await self.execute_trade(
                                symbol, 
                                Action.SELL, 
                                size, 
                                order_type='MARKET',
                                params={'trailingStop': True}
                            )
                            continue
                        
                        # Take profit based on risk-reward ratio (e.g., 1:2)
                        risk_reward_ratio = 2.0
                        take_profit = entry_price + (volatility_stop * risk_reward_ratio)
                        
                        if current_price >= take_profit:
                            logger.info(f"Take profit target reached for {symbol} at {current_price}")
                            # Consider taking partial profits
                            await self.execute_trade(
                                symbol,
                                Action.SELL,
                                size * 0.5,  # Close half position
                                order_type='TAKE_PROFIT_LIMIT',
                                price=current_price,
                                stop_price=current_price * 0.995  # Slightly below current
                            )
                    
                    elif side == 'short':
                        # Similar logic for short positions
                        trailing_stop_pct = 0.03
                        current_trailing_stop = current_price * (1 + trailing_stop_pct)
                        
                        if hasattr(self, f'trailing_stop_{symbol}'):
                            self.trailing_stop = min(getattr(self, f'trailing_stop_{symbol}'), current_trailing_stop)
                        else:
                            self.trailing_stop = current_trailing_stop
                        
                        if current_price >= self.trailing_stop:
                            logger.info(f"Trailing stop triggered for {symbol} at {current_price}")
                            await self.execute_trade(
                                symbol, 
                                Action.BUY, 
                                size, 
                                order_type='MARKET',
                                params={'trailingStop': True}
                            )
                            continue
                        
                        risk_reward_ratio = 2.0
                        take_profit = entry_price - (volatility_stop * risk_reward_ratio)
                        
                        if current_price <= take_profit:
                            logger.info(f"Take profit target reached for {symbol} at {current_price}")
                            await self.execute_trade(
                                symbol,
                                Action.BUY,
                                size * 0.5,  # Close half position
                                order_type='TAKE_PROFIT_LIMIT',
                                price=current_price,
                                stop_price=current_price * 1.005  # Slightly above current
                            )
                    
                    # Monitor position health (e.g., max drawdown)
                    max_drawdown_pct = self.config.get('max_drawdown_pct', 0.1)  # 10% max drawdown
                    if pnl_pct < -max_drawdown_pct * 100:  # Convert to percentage
                        logger.warning(f"Max drawdown reached for {symbol} ({pnl_pct:.2f}%)")
                        await self.execute_trade(
                            symbol,
                            Action.SELL if side == 'long' else Action.BUY,
                            size,
                            order_type='MARKET'
                        )
                    
                except Exception as e:
                    logger.error(f"Error managing position for {pos_symbol}: {e}")
                    logger.exception("Position management error:")
        
        except Exception as e:
            logger.error(f"Error in risk management for {symbol}: {e}")
            logger.exception("Risk management error:")
    
    async def update_positions(self, order: dict, action: Action) -> None:
        """Update internal position tracking after order execution"""
        try:
            symbol = order.get('symbol')
            if not symbol:
                return
                
            base_currency = symbol.split('/')[0]
            amount = float(order.get('filled', order.get('amount', 0)))
            
            if amount <= 0:
                return
            
            # Initialize position if it doesn't exist
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'size': 0.0,
                    'entry_price': 0.0,
                    'current_value': 0.0,
                    'unrealized_pnl': 0.0,
                    'last_updated': int(time.time() * 1000)
                }
            
            position = self.positions[symbol]
            
            if action == Action.BUY:
                # Calculate new average entry price
                total_cost = (position['size'] * position['entry_price']) + (amount * float(order.get('price', 0)))
                position['size'] += amount
                position['entry_price'] = total_cost / position['size'] if position['size'] > 0 else 0
            else:  # SELL
                if position['size'] > 0:
                    # Reduce position size
                    position['size'] = max(0, position['size'] - amount)
                    # If position is closed, reset entry price
                    if position['size'] <= 0:
                        position['entry_price'] = 0.0
            
            # Update position metrics
            if symbol in self.positions:
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                self.positions[symbol].update({
                    'current_price': current_price,
                    'current_value': self.positions[symbol]['size'] * current_price,
                    'unrealized_pnl': (current_price - self.positions[symbol]['entry_price']) * self.positions[symbol]['size'],
                    'last_updated': int(time.time() * 1000)
                })
                
                # Log position update
                logger.info(f"Position updated - {symbol}: {self.positions[symbol]}")
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            logger.exception("Position update error:")
    
    async def check_positions(self, symbol: str = None) -> dict:
        """
        Get open positions with detailed information
        
        Args:
            symbol: Optional symbol to filter positions
            
        Returns:
            dict: Dictionary of open positions with detailed information
        """
        try:
            if symbol:
                # Check if we have a position for this symbol
                if symbol in self.positions and self.positions[symbol]['size'] > 0:
                    return {symbol: self.positions[symbol]}
                return {}
            
            # Return all non-zero positions
            return {s: p for s, p in self.positions.items() if p['size'] > 0}
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return {}
    
    async def get_position_metrics(self, symbol: str) -> dict:
        """
        Get detailed metrics for a position
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            dict: Position metrics including PnL, ROI, etc.
        """
        try:
            if symbol not in self.positions or self.positions[symbol]['size'] <= 0:
                return {}
                
            position = self.positions[symbol]
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate PnL
            entry_value = position['size'] * position['entry_price']
            current_value = position['size'] * current_price
            unrealized_pnl = current_value - entry_value
            unrealized_pnl_pct = (unrealized_pnl / entry_value) * 100 if entry_value > 0 else 0
            
            # Calculate daily PnL
            daily_pnl = 0.0
            daily_pnl_pct = 0.0
            if 'last_updated' in position:
                time_diff_hours = (time.time() * 1000 - position['last_updated']) / (1000 * 3600)
                if time_diff_hours > 0:
                    daily_pnl = (unrealized_pnl / time_diff_hours) * 24
                    daily_pnl_pct = (unrealized_pnl_pct / time_diff_hours) * 24
            
            return {
                'symbol': symbol,
                'size': position['size'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'current_value': current_value,
                'last_updated': int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting position metrics for {symbol}: {e}")
            return {}

    async def run(self):
        """Run the trading bot"""
        logger.info("Starting trading bot...")
        
        try:
            while True:
                try:
                    # Analyze each trading pair
                    for symbol in self.symbols:
                        try:
                            # Check and manage existing positions
                            await self.manage_risk(symbol)
                            
                            # Get market analysis
                            analysis = await self.analyze_market(symbol)
                            logger.info(f"{symbol} Analysis:")
                            logger.info(f"  Price: {analysis.get('price')}")
                            logger.info(f"  Strategy Action: {analysis.get('action')} (Confidence: {analysis.get('confidence', 0.0):.2f})")
                            
                            if 'ai_recommendation' in analysis:
                                rec = analysis['ai_recommendation']
                                logger.info(f"  AI Recommendation: {rec['action']} (Confidence: {rec['confidence']:.2f})")
                            
                            # Execute trades based on analysis if auto-trading is enabled
                            if self.config.get('auto_trade', False):
                                action = Action[analysis['action']]
                                
                                # Get available balance and calculate position size
                                if action == Action.BUY:
                                    balance = await self.get_balance('USDT')
                                    amount = (balance * 0.95) / analysis['price']  # Use 95% of available balance
                                else:  # SELL
                                    base_currency = symbol.split('/')[0]
                                    amount = await self.get_balance(base_currency)
                                
                                # Ensure minimum order size (e.g., 0.001 BTC)
                                min_amount = 0.001
                                if amount >= min_amount:
                                    await self.execute_trade(symbol, action, amount, analysis['price'])
                                    
                                    # Place stop loss and take profit orders
                                    if action != Action.HOLD:
                                        stop_loss_price = analysis['price'] * (0.95 if action == Action.BUY else 1.05)
                                        take_profit_price = analysis['price'] * (1.10 if action == Action.BUY else 0.90)
                                        
                                        await self.execute_trade(
                                            symbol,
                                            Action.SELL if action == Action.BUY else Action.BUY,
                                            amount,
                                            stop_loss_price,
                                            'STOP_LOSS'
                                        )
                                        
                                        await self.execute_trade(
                                            symbol,
                                            Action.SELL if action == Action.BUY else Action.BUY,
                                            amount,
                                            take_profit_price,
                                            'TAKE_PROFIT'
                                        )
                                else:
                                    logger.warning(f"Insufficient balance for {symbol} trade. Needed: {min_amount}, Available: {amount}")
                                    
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                    
                    # Wait for the next update
                    await asyncio.sleep(self.config.get('data_refresh_interval', 300))  # Default 5 minutes
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(60)  # Wait a minute before retrying
                    
        except asyncio.CancelledError:
            logger.info("Shutting down...")
            if hasattr(self.exchange, 'close'):
                await self.exchange.close()
            logger.info("Trading bot stopped")

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get available balance
            if self.config.get('trading_enabled', False):
                balance = self.exchange.fetch_balance()
                usdt_balance = balance['USDT']['free']
            else:
                usdt_balance = self.config.get('paper_balance', 10000.0)
            
            # Calculate position size (simplified)
            max_position_value = usdt_balance * self.config.get('max_position_size', 0.1)
            position_size = max_position_value / price
            
            # Round to appropriate precision
            market = self.exchange.market(symbol)
            precision = market['precision']['amount']
            position_size = round(position_size, precision)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
            
    async def get_ai_market_data(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 100) -> dict:
        """
        Gather comprehensive market data for AI analysis
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for OHLCV data
            limit: Number of candles to fetch
            
        Returns:
            Dictionary containing all relevant market data for AI analysis
        """
        try:
            # Fetch OHLCV data with technical indicators
            ohlcv = await self.fetch_market_data(symbol, timeframe, limit)
            if ohlcv.empty:
                return {"error": f"No data available for {symbol}"}
                
            # Get current price and order book
            ticker = await self.data_handler.fetch_ticker(symbol)
            orderbook = await self.data_handler.fetch_order_book(symbol, limit=10)
            
            # Prepare the data dictionary
            latest = ohlcv.iloc[-1]
            
            data = {
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "price_data": {
                    "current": float(ticker['last']) if ticker else float(latest['close']),
                    "open": float(latest['open']),
                    "high": float(latest['high']),
                    "low": float(latest['low']),
                    "close": float(latest['close']),
                    "volume": float(latest['volume']),
                    "bid": float(ticker['bid']) if ticker else None,
                    "ask": float(ticker['ask']) if ticker else None,
                    "spread": float(ticker['ask'] - ticker['bid']) if ticker and 'ask' in ticker and 'bid' in ticker else None
                },
                "indicators": {
                    "rsi": float(latest['RSI']) if 'RSI' in ohlcv.columns else None,
                    "macd": float(latest['MACD']) if 'MACD' in ohlcv.columns else None,
                    "macd_signal": float(latest['Signal_Line']) if 'Signal_Line' in ohlcv.columns else None,
                    "bb_upper": float(latest['BB_upper']) if 'BB_upper' in ohlcv.columns else None,
                    "bb_middle": float(latest['BB_middle']) if 'BB_middle' in ohlcv.columns else None,
                    "bb_lower": float(latest['BB_lower']) if 'BB_lower' in ohlcv.columns else None,
                    "sma_20": float(latest['SMA_20']) if 'SMA_20' in ohlcv.columns else None,
                    "sma_50": float(latest['SMA_50']) if 'SMA_50' in ohlcv.columns else None,
                },
                "orderbook": {
                    "bids": [[float(price), float(amount)] for price, amount in orderbook['bids'][:5]] if orderbook and 'bids' in orderbook else [],
                    "asks": [[float(price), float(amount)] for price, amount in orderbook['asks'][:5]] if orderbook and 'asks' in orderbook else [],
                    "bid_volume": sum(amount for _, amount in orderbook['bids'][:5]) if orderbook and 'bids' in orderbook else 0,
                    "ask_volume": sum(amount for _, amount in orderbook['asks'][:5]) if orderbook and 'asks' in orderbook else 0
                },
                "market_conditions": {
                    "trend": self._determine_trend(ohlcv),
                    "volatility": self._calculate_volatility(ohlcv),
                    "volume_profile": self._analyze_volume(ohlcv)
                }
            }
            
            # Add AI predictions if available
            if hasattr(self, 'ai_trader') and self.ai_trader:
                try:
                    # Prepare features for AI model
                    features = self.prepare_features_for_ai(ohlcv)
                    if not features.empty:
                        prediction = self.ai_trader.predict(features)
                        data["ai_prediction"] = prediction
                except Exception as e:
                    self.logger.error(f"Error generating AI prediction: {e}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in get_ai_market_data: {e}")
            return {"error": str(e)}
    
    def _determine_trend(self, ohlcv: pd.DataFrame) -> str:
        """Determine the current market trend"""
        if 'close' not in ohlcv.columns or len(ohlcv) < 50:
            return "neutral"
            
        close = ohlcv['close']
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        
        if len(sma_20) < 2 or len(sma_50) < 2:
            return "neutral"
            
        # Check if 20 SMA is above 50 SMA (uptrend) or below (downtrend)
        if sma_20.iloc[-1] > sma_50.iloc[-1] * 1.02:  # 2% threshold
            return "bullish"
        elif sma_20.iloc[-1] < sma_50.iloc[-1] * 0.98:  # 2% threshold
            return "bearish"
        return "neutral"
    
    def _calculate_volatility(self, ohlcv: pd.DataFrame, window: int = 14) -> float:
        """Calculate market volatility"""
        if 'close' not in ohlcv.columns or len(ohlcv) < window + 1:
            return 0.0
            
        returns = ohlcv['close'].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
            
        return float(returns.std() * (252 ** 0.5))  # Annualized volatility
    
    def _analyze_volume(self, ohlcv: pd.DataFrame, window: int = 20) -> dict:
        """Analyze volume profile"""
        if 'volume' not in ohlcv.columns or len(ohlcv) < window:
            return {"current": 0, "average": 0, "trend": "neutral"}
            
        volume = ohlcv['volume']
        current_volume = float(volume.iloc[-1])
        avg_volume = float(volume.rolling(window=window).mean().iloc[-1])
        
        trend = "neutral"
        if current_volume > avg_volume * 1.5:
            trend = "high"
        elif current_volume < avg_volume * 0.5:
            trend = "low"
            
        return {
            "current": current_volume,
            "average": avg_volume,
            "trend": trend
        }

    async def get_market_analysis(self, symbol: str = 'BTC/USDT', timeframe: str = '1h') -> str:
        """Get a detailed market analysis for a given symbol"""
        try:
            # Get comprehensive market data
            market_data = await self.get_ai_market_data(symbol, timeframe)
            
            if 'error' in market_data:
                return f"Error analyzing market: {market_data['error']}"
            
            # Generate analysis text
            price = market_data['price_data']
            indicators = market_data['indicators']
            conditions = market_data['market_conditions']
            
            analysis = f"""
=== {symbol} Market Analysis ===

## Price Action
- Current: {price['current']:.2f} (O: {price['open']:.2f}, H: {price['high']:.2f}, L: {price['low']:.2f}, C: {price['close']:.2f})
- 24h Volume: {price['volume']:.2f} {symbol.split('/')[1]}
- Spread: {price['spread']:.4f} ({price['spread']/price['current']*100:.2f}%)

## Technical Indicators
- RSI(14): {indicators['rsi']:.2f} {'(Overbought)' if indicators['rsi'] > 70 else '(Oversold)' if indicators['rsi'] < 30 else ''}
- MACD: {indicators['macd']:.4f} | Signal: {indicators['macd_signal']:.4f}
- Bollinger Bands: {indicators['bb_lower']:.2f} - {indicators['bb_upper']:.2f}
- Moving Averages: SMA20: {indicators['sma_20']:.2f} | SMA50: {indicators['sma_50']:.2f}

## Market Conditions
- Trend: {conditions['trend'].capitalize()}
- Volatility: {conditions['volatility']*100:.2f}% (annualized)
- Volume: {conditions['volume_profile']['current']:.2f} (Avg: {conditions['volume_profile']['average']:.2f}, Trend: {conditions['volume_profile']['trend']})
"""
            # Add AI prediction if available
            if 'ai_prediction' in market_data:
                pred = market_data['ai_prediction']
                analysis += f"\n## AI Prediction\n"
                for action, prob in pred.items():
                    analysis += f"- {action}: {prob*100:.1f}%\n"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in get_market_analysis: {str(e)}")
            return f"Error analyzing market: {str(e)}"
    
    async def start_chat_interface(self):
        """Start the interactive chat interface"""
        from chat_interface import TradingChatbot, chat_loop
        print("\n=== Starting Trading Bot Chat Interface ===")
        print("Loading AI model...")
        
        try:
            chatbot = TradingChatbot()
            await chat_loop(chatbot, trading_agent=self)
        except Exception as e:
            print(f"Error in chat interface: {str(e)}")
            import traceback
            traceback.print_exc()
        except KeyboardInterrupt:
            print("\nExiting chat interface...")

def show_menu():
    """Display the main menu"""
    print("\n=== Trading Bot ===")
    print("1. Start Trading")
    print("2. Start Chat Interface")
    print("3. Get Market Analysis")
    print("4. Exit")

async def main():
    """Main entry point"""
    agent = None
    try:
        # Create the trading agent
        agent = TradingAgent()
        
        while True:
            show_menu()
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == '1':
                # Start the trading loop
                logger.info("Starting trading bot...")
                await agent.run()
                break
                
            elif choice == '2':
                # Start the chat interface
                await agent.start_chat_interface()
                # After chat interface exits, return to menu
                continue
                
            elif choice == '3':
                # Show market analysis
                symbol = input("Enter trading pair (e.g., BTC/USDT): ").strip() or 'BTC/USDT'
                try:
                    analysis = await agent.analyze_market(symbol)
                    print("\n" + "="*50)
                    print(analysis)
                    print("="*50 + "\n")
                except Exception as e:
                    print(f"Error getting market analysis: {str(e)}")
                
            elif choice == '4':
                print("Exiting...")
                return
                
            else:
                print("Invalid choice. Please select 1-4.")
                
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        if agent is not None and hasattr(agent, 'exchange'):
            # CCXT doesn't have a close method, but we can clear any active connections
            if hasattr(agent.exchange, 'close'):
                await agent.exchange.close()
            logger.info("Trading bot stopped")

if __name__ == "__main__":
    asyncio.run(main())