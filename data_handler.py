import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import ta  # Using ta library instead of TA-Lib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, exchange=None):
        """Initialize DataHandler with an optional exchange instance.
        
        Args:
            exchange: Optional ccxt exchange instance. If None, will create a new one.
        """
        if exchange is None:
            load_dotenv()
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET_KEY', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # for spot trading
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                    'warnOnFetchCurrenciesWithoutPermission': False,
                    'fetchCurrencies': False,
                },
                'urls': {
                    'api': {
                        'public': 'https://api.binance.com/api/v3',
                        'private': 'https://api.binance.com/api/v3',
                        'sapi': 'https://api.binance.com/sapi/v1'
                    }
                },
                'timeout': 30000,  # 30 seconds timeout
                'enableRateLimit': True,
            })
        else:
            self.exchange = exchange
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 1000):
        """Fetch OHLCV data from exchange"""
        try:
            # Use the exchange's built-in method to fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No OHLCV data returned for {symbol} {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()  # Return empty DataFrame instead of None

    async def fetch_order_book(self, symbol: str, limit: int = 20):
        """Fetch order book data"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    async def fetch_ticker(self, symbol: str):
        """Fetch ticker data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators to the DataFrame using ta library"""
        if df.empty:
            return df
            
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        try:
            # Initialize indicators using the functional API
            # RSI
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            
            # MACD
            macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            macd = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            bb_upper = bollinger.bollinger_hband()
            bb_middle = bollinger.bollinger_mavg()
            bb_lower = bollinger.bollinger_lband()
            
            # Moving Averages
            sma_20 = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            sma_50 = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
            
            # Add indicators to DataFrame
            df['RSI'] = rsi
            df['MACD'] = macd
            df['Signal_Line'] = macd_signal
            df['BB_upper'] = bb_upper
            df['BB_middle'] = bb_middle
            df['BB_lower'] = bb_lower
            df['SMA_20'] = sma_20
            df['SMA_50'] = sma_50
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df  # Return the original DataFrame if there's an error

    @staticmethod
    def prepare_features(df):
        """Prepare features for model training/prediction"""
        # Normalize the data
        df_norm = (df - df.mean()) / df.std()
        return df_norm