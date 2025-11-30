import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from dotenv import load_dotenv
from autonomous_trader import AutonomousTrader

# Load environment variables
load_dotenv()

def fetch_historical_data(symbol: str = 'BTC/USDT', timeframe: str = '1h', days: int = 30):
    """Fetch historical OHLCV data from Binance (using public endpoint)"""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
        },
        'urls': {
            'api': {
                'public': 'https://api.binance.com/api/v3',
                'private': 'https://api.binance.com/api/v3',
            }
        }
    })
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for training by adding technical indicators"""
    # Ensure we have the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add returns
    data['returns'] = data['close'].pct_change()
    
    # Add moving averages
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    
    # Add RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Add MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    
    # Add Bollinger Bands
    data['bb_high'] = data['close'].rolling(window=20).mean() + 2 * data['close'].rolling(window=20).std()
    data['bb_mid'] = data['close'].rolling(window=20).mean()
    data['bb_low'] = data['close'].rolling(window=20).mean() - 2 * data['close'].rolling(window=20).std()
    
    # Add ATR (Average True Range)
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['atr'] = true_range.rolling(window=14).mean()
    
    # Add volume indicators
    data['volume_sma'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    # Add price change indicators
    data['price_change_1h'] = data['close'].pct_change(periods=1)
    data['price_change_4h'] = data['close'].pct_change(periods=4)
    data['price_change_24h'] = data['close'].pct_change(periods=24)
    
    # Add volatility
    data['volatility'] = data['close'].pct_change().rolling(window=20).std() * np.sqrt(24)  # 24h annualized
    
    # Ensure all required columns for the environment are present
    required_env_columns = ['returns', 'rsi', 'macd', 'macd_diff', 'bb_high', 'bb_mid', 'bb_low', 'atr', 'volume_ratio']
    for col in required_env_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column for environment: {col}")
    
    # Forward fill any remaining NaN values and drop any that can't be filled
    data = data.ffill().dropna()
    
    return data

def main():
    print("🚀 Starting model training...")
    
    # 1. Fetch historical data
    print("📊 Fetching historical data...")
    try:
        data = fetch_historical_data()
        print(f"✅ Successfully fetched {len(data)} rows of historical data")
    except Exception as e:
        print(f"❌ Error fetching historical data: {str(e)}")
        print("⚠️  Using sample data for training...")
        # Generate sample data if API fails
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='1h')
        data = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 1000).cumsum(),
            'high': np.random.normal(50200, 1000, 1000).cumsum(),
            'low': np.random.normal(49800, 1000, 1000).cumsum(),
            'close': np.random.normal(50000, 1000, 1000).cumsum(),
            'volume': np.random.uniform(10, 100, 1000)
        }, index=dates)
        data = prepare_training_data(data)
    
    # 2. Prepare data
    print("🔧 Preparing data...")
    data = prepare_training_data(data)
    
    # 3. Initialize the trader with a mock exchange for training
    print("🤖 Initializing autonomous trader with mock exchange...")
    class MockExchange:
        def __init__(self):
            self.symbols = ['BTC/USDT']
            self.timeframes = ['1h', '4h', '1d']
            
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            # Return mock data
            return []
            
        def load_markets(self, reload=False):
            return {}
    
    exchange = MockExchange()
    
    trader = AutonomousTrader(exchange)
    
    # 4. Train the model
    print("🎓 Training model (this may take a while)...")
    trader.train(data, total_timesteps=10000)  # Start with a small number for testing
    
    print("✅ Training completed! You can now run the trading app.")
    print("   Run: streamlit run autonomous_app.py")

if __name__ == "__main__":
    main()
