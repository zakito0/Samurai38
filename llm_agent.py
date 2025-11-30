import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
import ccxt
import ta
import yfinance as yf
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"

@dataclass
class MarketAnalysis(BaseModel):
    """Comprehensive market analysis result"""
    # Basic info
    symbol: str
    timeframe: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Trend analysis
    trend: str  # primary trend direction
    trend_strength: float  # 0-1 scale
    higher_timeframe_trend: Optional[str] = None
    
    # Key levels
    support_levels: List[float]
    resistance_levels: List[float]
    pivot_points: Dict[str, float] = {}  # Pivot, S1, S2, R1, R2, etc.
    
    # Volume analysis
    volume_trend: str  # increasing, decreasing, neutral
    volume_anomaly: Optional[bool] = None
    
    # Pattern recognition
    chart_patterns: List[Dict[str, Any]] = []  # List of detected patterns
    candlestick_patterns: List[Dict[str, Any]] = []  # List of candlestick patterns
    harmonic_patterns: List[Dict[str, Any]] = []  # Harmonic patterns
    fibonacci_levels: Dict[str, float] = {}  # Key Fibonacci levels
    
    # Market regime
    market_regime: str  # Trending, Ranging, Volatile, etc.
    volatility: float  # 0-1 scale
    
    # Indicators
    indicators: Dict[str, Any]
    
    # Elliott Wave (if detected)
    elliott_wave: Optional[Dict[str, Any]] = None
    
    # Wyckoff analysis
    wyckoff_phase: Optional[str] = None
    
    # Trading signals
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: List[float] = []
    risk_reward_ratio: Optional[float] = None
    
    # Final assessment
    confidence: float  # 0-1 scale
    recommendation: str  # Strong Buy, Buy, Neutral, Sell, Strong Sell
    reasoning: str  # Detailed explanation
    
    # Risk metrics
    position_size: Optional[float] = None  # Suggested position size
    risk_per_trade: Optional[float] = None  # % of capital to risk

class LLMTradingAgent:
    def __init__(
        self,
        model_name: str = "htngtech/deepseek-r1t2-chimera:free",  # Default to Mistral 7B on OpenRouter
        api_key: str = None,
        strategy: LLMStrategy = LLMStrategy.CHAIN_OF_THOUGHT,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        """
        Initialize the LLM-based trading agent with OpenRouter API.
        
        Args:
            model_name: Name of the LLM model to use (OpenRouter model name)
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY from .env)
            strategy: LLM strategy to use for decision making
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for repeating tokens
            presence_penalty: Penalty for new tokens
        """
        # Load environment variables
        load_dotenv()
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY not found in environment variables")
            
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        # HTTP session for API calls
        self.session = None
        
        # Cache for market data and analyses
        self.market_cache = {}
        self.analysis_cache = {}
        
        logger.info(f"Initialized LLM Trading Agent with OpenRouter model: {model_name}")
        
        # Initialize the API session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize the aiohttp session for OpenRouter API calls."""
        if self.session is None or self.session.closed:
            # Create a new event loop if one isn't running
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Create the session
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/yourusername/your-repo",
                    "X-Title": "NinjaZtrade",
                    "Content-Type": "application/json"
                },
                loop=loop
            )
    
    async def initialize_async(self):
        """Initialize any async components of the agent."""
        # This method can be used for any async initialization that needs to happen
        # after the agent is created but before it's used
        if self.session is None or self.session.closed:
            self._initialize_session()
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def analyze_market(self, symbol: str, timeframe: str = "1d", lookback: int = 30) -> MarketAnalysis:
        """
        Analyze the market for a given symbol using LLM.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis (e.g., '1d', '4h', '1h')
            lookback: Number of periods to look back
            
        Returns:
            MarketAnalysis object with the analysis results
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{lookback}"
            if cache_key in self.analysis_cache:
                cached = self.analysis_cache[cache_key]
                if (datetime.now() - cached.timestamp).total_seconds() < 300:  # 5-minute cache
                    return cached
            
            # Get market data
            ohlcv = await self._get_market_data(symbol, timeframe, lookback)
            if ohlcv.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(ohlcv)
            
            # Prepare prompt based on strategy
            prompt = self._prepare_analysis_prompt(symbol, ohlcv, indicators)
            
            # Generate analysis using LLM
            analysis_text = await self._generate_text(prompt)
            
            # Parse the LLM response
            analysis = self._parse_analysis(analysis_text, symbol, indicators)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis for {symbol}: {e}")
            # Return neutral analysis in case of error
            return MarketAnalysis(
                symbol=symbol,
                trend="neutral",
                confidence=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                indicators={},
                timestamp=datetime.now()
            )
    
    async def generate_trade_signal(
        self, 
        symbol: str, 
        portfolio: Dict[str, float], 
        current_price: float,
        position_size: float = 0.1,
        risk_per_trade: float = 0.01
    ) -> Dict[str, Any]:
        """
        Generate a trading signal using LLM analysis.
        
        Args:
            symbol: Trading pair symbol
            portfolio: Current portfolio holdings
            current_price: Current price of the asset
            position_size: Size of position as percentage of portfolio (0.0 to 1.0)
            risk_per_trade: Risk per trade as percentage of portfolio (0.0 to 1.0)
            
        Returns:
            Dictionary containing trade signal and metadata
        """
        try:
            # Get market analysis
            analysis = await self.analyze_market(symbol)
            
            # Prepare prompt for trade signal generation
            prompt = self._prepare_trade_prompt(
                symbol=symbol,
                analysis=analysis,
                portfolio=portfolio,
                current_price=current_price,
                position_size=position_size,
                risk_per_trade=risk_per_trade
            )
            
            # Generate trade signal using LLM
            signal_text = await self._generate_text(prompt)
            
            # Parse the trade signal
            signal = self._parse_trade_signal(signal_text, symbol, current_price, position_size, risk_per_trade)
            
            return {
                "symbol": symbol,
                "action": signal["action"],
                "size": signal["size"],
                "price": current_price,
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "timestamp": datetime.now().isoformat(),
                "indicators": analysis.indicators
            }
            
        except Exception as e:
            logger.error(f"Error generating trade signal for {symbol}: {e}")
            return {
                "symbol": symbol,
                "action": "HOLD",
                "size": 0.0,
                "price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "confidence": 0.0,
                "reasoning": f"Error generating signal: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "indicators": {}
            }
    
    async def _get_market_data(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Fetch market data for analysis."""
        try:
            # Try using yfinance first for more reliable data
            if '/' in symbol:
                yf_symbol = symbol.replace('/', '-')  # Convert BTC/USDT to BTC-USDT
            else:
                yf_symbol = symbol
                
            # Map timeframes to yfinance intervals
            tf_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '60m', '4h': '60m', '1d': '1d', '1w': '1wk'
            }
            
            interval = tf_map.get(timeframe, '1d')
            period = f"{lookback}d" if 'd' in interval else f"{lookback*int(interval[:-1])}m"
            
            # Download data
            data = yf.download(
                yf_symbol,
                period=period,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                # Fallback to CCXT if yfinance fails
                logger.warning(f"yfinance returned no data for {symbol}, trying CCXT")
                exchange = ccxt.binance()
                since = exchange.parse_timeframe(timeframe) * 1000 * lookback
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
                data = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate a comprehensive set of technical indicators and patterns."""
        if ohlcv.empty:
            return {}
            
        indicators = {}
        
        # Basic OHLCV data
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        volume = ohlcv['volume']
        
        # 1. Trend Indicators
        # Moving Averages
        indicators['sma_20'] = ta.trend.sma_indicator(close, window=20)
        indicators['sma_50'] = ta.trend.sma_indicator(close, window=50)
        indicators['sma_200'] = ta.trend.sma_indicator(close, window=200)
        indicators['ema_12'] = ta.trend.ema_indicator(close, window=12)
        indicators['ema_26'] = ta.trend.ema_indicator(close, window=26)
        
        # MACD
        macd = ta.trend.MACD(close)
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_diff'] = macd.macd_diff()
        
        # 2. Momentum Indicators
        indicators['rsi'] = ta.momentum.RSIIndicator(close).rsi()
        indicators['stoch_k'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
        indicators['stoch_d'] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
        indicators['cci'] = ta.trend.CCIIndicator(high, low, close).cci()
        indicators['awesome_oscillator'] = ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator()
        
        # 3. Volatility Indicators
        indicators['bb_high'] = ta.volatility.BollingerBands(close).bollinger_hband()
        indicators['bb_low'] = ta.volatility.BollingerBands(close).bollinger_lband()
        indicators['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        
        # 4. Volume Indicators
        indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        indicators['cmf'] = ta.volume.chaikin_money_flow(high, low, close, volume)
        indicators['mfi'] = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()
        
        # 5. Pattern Recognition
        indicators['patterns'] = self._detect_chart_patterns(ohlcv) if hasattr(self, '_detect_chart_patterns') else []
        indicators['candlestick_patterns'] = self._detect_candlestick_patterns(ohlcv) if hasattr(self, '_detect_candlestick_patterns') else []
        
        # 6. Support and Resistance Levels
        support_resistance = self._calculate_support_resistance(ohlcv) if hasattr(self, '_calculate_support_resistance') else {}
        indicators.update(support_resistance)
        
        # 7. Fibonacci Levels
        indicators['fibonacci'] = self._calculate_fibonacci_levels(ohlcv) if hasattr(self, '_calculate_fibonacci_levels') else {}
        
        # 8. Elliott Wave Analysis
        indicators['elliott_wave'] = self._analyze_elliott_waves(ohlcv) if hasattr(self, '_analyze_elliott_waves') else {}
        
        # 9. Wyckoff Analysis
        indicators['wyckoff'] = self._analyze_wyckoff(ohlcv) if hasattr(self, '_analyze_wyckoff') else {}
        
        return indicators
        
        trend = "neutral"
        if sma_20 > sma_50 and sma_50 > sma_200:
            trend = "strong uptrend"
        elif sma_20 < sma_50 and sma_50 < sma_200:
            trend = "strong downtrend"
        elif sma_20 > sma_50:
            trend = "short-term uptrend"
        elif sma_20 < sma_50:
            trend = "short-term downtrend"
        
        # Prepare the prompt based on strategy
        if self.strategy == LLMStrategy.ZERO_SHOT:
            prompt = f"""Analyze the current market conditions for {symbol} and provide a trading recommendation.
            
Current Price: ${price:,.2f}
24h Volume: {volume:,.2f} {symbol.split('/')[-1] if '/' in symbol else 'tokens'}

Technical Indicators:
- 20-day SMA: ${sma_20:,.2f}
- 50-day SMA: ${sma_50:,.2f}
- 200-day SMA: ${sma_200:,.2f}
- RSI: {indicators.get('rsi', 0):.2f}
- MACD: {indicators.get('macd', 0):.4f} (Signal: {indicators.get('macd_signal', 0):.4f})
- Bollinger Bands: ${indicators.get('bb_low', 0):,.2f} - ${indicators.get('bb_high', 0):,.2f}
- Volume (vs 20-day avg): {volume/indicators.get('volume_sma', 1)*100:.1f}%

Market Trend: {trend}

Based on this analysis, provide a trading recommendation with the following format:

ANALYSIS:
[Your analysis of the market conditions, trends, and key indicators]

RECOMMENDATION:
- Action: [BUY/SELL/HOLD]
- Confidence: [0-100]%
- Reason: [Brief explanation of the recommendation]
- Key Levels:
  - Support: [Price level]
  - Resistance: [Price level]
  - Stop Loss: [Price level]
  - Take Profit: [Price level]"""
            
        elif self.strategy == LLMStrategy.FEW_SHOT:
            prompt = f"""You are an expert cryptocurrency trading analyst. Analyze the following market data and provide a trading recommendation.

Example 1:
Current Price: $45,000
Market Trend: Uptrend
RSI: 65
MACD: Bullish crossover
Volume: Above average

ANALYSIS: The market is in a strong uptrend with healthy volume. RSI is slightly overbought but not extreme. MACD shows a recent bullish crossover.
RECOMMENDATION:
- Action: BUY
- Confidence: 75%
- Reason: Strong uptrend with confirmation from volume and MACD
- Key Levels:
  - Support: $43,200
  - Resistance: $46,500
  - Stop Loss: $42,800
  - Take Profit: $48,000

Now analyze this market:

Current Price: ${price:,.2f}
24h Volume: {volume:,.2f} {symbol.split('/')[-1] if '/' in symbol else 'tokens'}

Technical Indicators:
- 20-day SMA: ${sma_20:,.2f}
- 50-day SMA: ${sma_50:,.2f}
- 200-day SMA: ${sma_200:,.2f}
- RSI: {indicators.get('rsi', 0):.2f}
- MACD: {indicators.get('macd', 0):.4f} (Signal: {indicators.get('macd_signal', 0):.4f})
- Bollinger Bands: ${indicators.get('bb_low', 0):,.2f} - ${indicators.get('bb_high', 0):,.2f}
- Volume (vs 20-day avg): {volume/indicators.get('volume_sma', 1)*100:.1f}%

Market Trend: {trend}

ANALYSIS:"""
            
        else:  # CHAIN_OF_THOUGHT
            prompt = f"""Analyze the current market conditions for {symbol} step by step and provide a trading recommendation.

Current Price: ${price:,.2f}
24h Volume: {volume:,.2f} {symbol.split('/')[-1] if '/' in symbol else 'tokens'}

Technical Indicators:
- 20-day SMA: ${sma_20:,.2f}
- 50-day SMA: ${sma_50:,.2f}
- 200-day SMA: ${sma_200:,.2f}
- RSI: {indicators.get('rsi', 0):.2f}
- MACD: {indicators.get('macd', 0):.4f} (Signal: {indicators.get('macd_signal', 0):4f})
- Bollinger Bands: ${indicators.get('bb_low', 0):,.2f} - ${indicators.get('bb_high', 0):,.2f}
- Volume (vs 20-day avg): {volume/indicators.get('volume_sma', 1)*100:.1f}%

Market Trend: {trend}

Let's analyze this step by step:

1. Trend Analysis:
   - Short-term trend (20 vs 50 SMA): {'Uptrend' if sma_20 > sma_50 else 'Downtrend'}
   - Long-term trend (50 vs 200 SMA): {'Bullish' if sma_50 > sma_200 else 'Bearish'}
   - Overall trend strength: {trend}

2. Momentum Indicators:
   - RSI: {indicators.get('rsi', 0):.2f} ({'Overbought' if indicators.get('rsi', 0) > 70 else 'Oversold' if indicators.get('rsi', 0) < 30 else 'Neutral'})
   - MACD: {indicators.get('macd', 0):.4f} ({'Bullish' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'Bearish'})
   - MACD Histogram: {indicators.get('macd_hist', 0):.4f} ({'Increasing' if indicators.get('macd_hist', 0) > 0 else 'Decreasing'})

3. Volatility:
   - Bollinger Band Width: {indicators.get('bb_width', 0):.4f} ({'High' if indicators.get('bb_width', 0) > 0.05 else 'Low'} volatility)
   - Price position in BB: {((price - indicators.get('bb_low', 0)) / (indicators.get('bb_high', 1) - indicators.get('bb_low', 0)) * 100 if indicators.get('bb_high', 0) != indicators.get('bb_low', 0) else 50):.1f}%

4. Volume Analysis:
   - Current Volume: {volume:,.2f}
   - Volume vs 20-day avg: {volume/indicators.get('volume_sma', 1)*100:.1f}%
   - OBV Trend: {'Up' if (indicators.get('obv', 0) > ohlcv['OBV'].iloc[-10:-1].mean() if len(ohlcv) > 10 else False) else 'Down'}

5. Price Action:
   - Recent price change (1d): {indicators.get('price_change_1d', 0):.2f}%
   - Recent price change (7d): {indicators.get('price_change_7d', 0):.2f}%
   - Recent price change (30d): {indicators.get('price_change_30d', 0):.2f}%

Based on this analysis, provide a trading recommendation with the following format:

ANALYSIS:
[Your detailed analysis of the market conditions, trends, and key indicators]

RECOMMENDATION:
- Action: [BUY/SELL/HOLD]
- Confidence: [0-100]%
- Reason: [Brief explanation of the recommendation]
- Key Levels:
  - Support: [Price level]
  - Resistance: [Price level]
  - Stop Loss: [Price level]
  - Take Profit: [Price level]"""
        
        return prompt
    
    def _prepare_trade_prompt(
        self,
        symbol: str,
        analysis: MarketAnalysis,
        portfolio: Dict[str, float],
        current_price: float,
        position_size: float,
        risk_per_trade: float
    ) -> str:
        """Prepare the prompt for trade signal generation."""
        # Calculate position size based on risk
        account_balance = portfolio.get('USDT', portfolio.get('USDC', 1000))  # Default to 1000 if not found
        risk_amount = account_balance * risk_per_trade
        position_amount = account_balance * position_size
        
        # Calculate stop loss and take profit levels
        atr = analysis.indicators.get('atr', current_price * 0.02)  # Default to 2% ATR if not available
        stop_loss_pct = 0.02  # 2% stop loss by default
        take_profit_pct = 0.04  # 4% take profit by default (2:1 reward:risk)
        
        if analysis.trend == 'bullish':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            action = "BUY"
        elif analysis.trend == 'bearish':
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
            action = "SELL"
        else:
            stop_loss = current_price * 0.99  # 1% stop loss for neutral
            take_profit = current_price * 1.02  # 2% take profit for neutral
            action = "HOLD"
        
        # Prepare the prompt
        prompt = f"""You are an expert cryptocurrency trader. Based on the following market analysis, generate a trade signal.

SYMBOL: {symbol}
CURRENT PRICE: ${current_price:,.2f}
ACCOUNT BALANCE: ${account_balance:,.2f}
POSITION SIZE: {position_size*100:.1f}% (${position_amount:,.2f})
RISK PER TRADE: {risk_per_trade*100:.1f}% (${risk_amount:,.2f})

MARKET ANALYSIS:
{analysis.reasoning}

TECHNICAL INDICATORS:
"""
        
        # Add indicators to the prompt
        for key, value in analysis.indicators.items():
            if isinstance(value, (int, float)):
                if 0 < abs(value) < 0.01:
                    prompt += f"- {key}: {value:.6f}\n"
                else:
                    prompt += f"- {key}: {value:.2f}\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        prompt += f"""

Based on this analysis, generate a trade signal with the following format:

TRADE SIGNAL:
- Action: [BUY/SELL/HOLD]
- Size: [Position size in quote currency or base asset]
- Entry Price: [Entry price]
- Stop Loss: [Stop loss price]
- Take Profit: [Take profit price]
- Risk/Reward Ratio: [e.g., 1:2]
- Confidence: [0-100]%
        """
        return prompt
    
    async def _generate_text(self, prompt: str) -> str:
        """Generate text using the OpenRouter API."""
        if not self.session or self.session.closed:
            self._initialize_session()
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert cryptocurrency trading assistant. Provide clear, concise, and accurate analysis and trading recommendations."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
                
                result = await response.json()
                return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise
    
    def _parse_analysis(self, text: str, symbol: str, indicators: Dict[str, Any]) -> MarketAnalysis:
        """Parse the LLM's analysis text into a MarketAnalysis object."""
        try:
            # Default values
            trend = "neutral"
            confidence = 0.5
            reasoning = ""
            
            # Try to extract analysis and recommendation
            analysis_parts = text.split("ANALYSIS:")
            if len(analysis_parts) > 1:
                reasoning = analysis_parts[1].split("RECOMMENDATION:")[0].strip()
                
                # Extract action and confidence
                rec_parts = text.split("RECOMMENDATION:")
                if len(rec_parts) > 1:
                    rec_text = rec_parts[1]
                    
                    # Extract action
                    if "action:" in rec_text.lower():
                        action_line = [line for line in rec_text.split('\n') if 'action:' in line.lower()]
                        if action_line:
                            action = action_line[0].split(':')[-1].strip().upper()
                            if 'BUY' in action:
                                trend = "bullish"
                            elif 'SELL' in action:
                                trend = "bearish"
                    
                    # Extract confidence
                    if "confidence:" in rec_text.lower():
                        conf_line = [line for line in rec_text.split('\n') if 'confidence:' in line.lower()]
                        if conf_line:
                            try:
                                conf_text = conf_line[0].split(':')[-1].strip().strip('%')
                                confidence = float(conf_text) / 100.0
                                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                            except (ValueError, IndexError):
                                pass
            else:
                # Fallback to simple parsing if the format doesn't match
                text_lower = text.lower()
                if 'buy' in text_lower and 'sell' not in text_lower:
                    trend = "bullish"
                elif 'sell' in text_lower and 'buy' not in text_lower:
                    trend = "bearish"
                reasoning = text
            
            return MarketAnalysis(
                symbol=symbol,
                trend=trend,
                confidence=confidence,
                reasoning=reasoning,
                indicators=indicators,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing analysis: {e}")
            return MarketAnalysis(
                symbol=symbol,
                trend="neutral",
                confidence=0.0,
                reasoning=f"Error parsing analysis: {str(e)}",
                indicators=indicators,
                timestamp=datetime.now()
            )
    
    def _parse_trade_signal(
        self, 
        text: str, 
        symbol: str, 
        current_price: float,
        position_size: float,
        risk_per_trade: float
    ) -> Dict[str, Any]:
        """Parse the LLM's trade signal text into a dictionary."""
        try:
            # Default values
            action = "HOLD"
            size = 0.0
            stop_loss = None
            take_profit = None
            confidence = 0.5
            reasoning = ""
            
            # Try to extract signal information
            if "TRADE SIGNAL:" in text:
                signal_text = text.split("TRADE SIGNAL:")[1]
                
                # Extract action
                action_match = [line for line in signal_text.split('\n') if 'action:' in line.lower()]
                if action_match:
                    action = action_match[0].split(':')[-1].strip().upper()
                    if 'BUY' in action:
                        action = "BUY"
                    elif 'SELL' in action:
                        action = "SELL"
                    else:
                        action = "HOLD"
                
                # Extract size
                size_match = [line for line in signal_text.split('\n') if 'size:' in line.lower()]
                if size_match:
                    try:
                        size_text = size_match[0].split(':')[-1].strip()
                        if '%' in size_text:
                            size = float(size_text.strip('%')) / 100.0 * position_size
                        elif '$' in size_text:
                            size = float(size_text.strip('$').replace(',', '')) / current_price
                        else:
                            size = float(size_text)
                    except (ValueError, IndexError):
                        size = position_size
                else:
                    size = position_size
                
                # Extract stop loss
                sl_match = [line for line in signal_text.split('\n') if 'stop loss' in line.lower()]
                if sl_match:
                    try:
                        sl_text = sl_match[0].split(':')[-1].strip()
                        if '$' in sl_text:
                            stop_loss = float(sl_text.strip('$').replace(',', ''))
                        else:
                            stop_loss = float(sl_text)
                    except (ValueError, IndexError):
                        pass
                
                # Extract take profit
                tp_match = [line for line in signal_text.split('\n') if 'take profit' in line.lower()]
                if tp_match:
                    try:
                        tp_text = tp_match[0].split(':')[-1].strip()
                        if '$' in tp_text:
                            take_profit = float(tp_text.strip('$').replace(',', ''))
                        else:
                            take_profit = float(tp_text)
                    except (ValueError, IndexError):
                        pass
                
                # Extract confidence
                conf_match = [line for line in signal_text.split('\n') if 'confidence:' in line.lower()]
                if conf_match:
                    try:
                        conf_text = conf_match[0].split(':')[-1].strip().strip('%')
                        confidence = float(conf_text) / 100.0
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    except (ValueError, IndexError):
                        pass
                
                # Extract reasoning
                reason_match = [line for line in signal_text.split('\n') if 'reasoning:' in line.lower()]
                if reason_match:
                    reasoning = ':'.join(reason_match[0].split(':')[1:]).strip()
            
            # Set default stop loss and take profit if not provided
            if stop_loss is None:
                stop_loss = current_price * (0.98 if action == "BUY" else 1.02)
            
            if take_profit is None:
                take_profit = current_price * (1.04 if action == "BUY" else 0.96)
            
            return {
                "action": action,
                "size": size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error parsing trade signal: {e}")
            return {
                "action": "HOLD",
                "size": 0.0,
                "stop_loss": None,
                "take_profit": None,
                "confidence": 0.0,
                "reasoning": f"Error parsing signal: {str(e)}"
            }
