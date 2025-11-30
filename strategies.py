from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import ta  # Using ta library instead of talib
from enum import Enum

class Action(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class StrategyResult:
    def __init__(self, action: Action, confidence: float, indicators: Dict[str, float]):
        self.action = action
        self.confidence = confidence
        self.indicators = indicators

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        """Analyze market data and return trading signal"""
        raise NotImplementedError("Subclasses must implement this method")

class MovingAverageCrossover(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"MA_Crossover_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        if len(df) < self.long_window + 1:
            return StrategyResult(Action.HOLD, 0.0, {})
        
        # Calculate moving averages
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Get the last two values for crossover detection
        prev_short = df['SMA_short'].iloc[-2]
        prev_long = df['SMA_long'].iloc[-2]
        curr_short = df['SMA_short'].iloc[-1]
        curr_long = df['SMA_long'].iloc[-1]
        
        # Check for crossover
        if prev_short <= prev_long and curr_short > curr_long:
            # Golden cross (short crosses above long)
            return StrategyResult(
                Action.BUY,
                0.7,
                {'short_ma': curr_short, 'long_ma': curr_long}
            )
        elif prev_short >= prev_long and curr_short < curr_long:
            # Death cross (short crosses below long)
            return StrategyResult(
                Action.SELL,
                0.7,
                {'short_ma': curr_short, 'long_ma': curr_long}
            )
        
        return StrategyResult(
            Action.HOLD,
            0.5,
            {'short_ma': curr_short, 'long_ma': curr_long}
        )

class RSIStrategy(BaseStrategy):
    """Relative Strength Index based strategy"""
    
    def __init__(self, rsi_period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        super().__init__(f"RSI_{rsi_period}")
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
    
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        if len(df) < self.rsi_period + 1:
            return StrategyResult(Action.HOLD, 0.0, {})
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < self.oversold:
            return StrategyResult(
                Action.BUY,
                min(1.0, (self.oversold - current_rsi) / self.oversold * 0.5 + 0.5),
                {'rsi': current_rsi}
            )
        elif current_rsi > self.overbought:
            return StrategyResult(
                Action.SELL,
                min(1.0, (current_rsi - self.overbought) / (100 - self.overbought) * 0.5 + 0.5),
                {'rsi': current_rsi}
            )
        
        return StrategyResult(
            Action.HOLD,
            0.5,
            {'rsi': current_rsi}
        )

class MACDStrategy(BaseStrategy):
    """Moving Average Convergence Divergence strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        if len(df) < self.slow_period + self.signal_period:
            return StrategyResult(Action.HOLD, 0.0, {})
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        prev_macd = macd.iloc[-2]
        prev_signal = signal.iloc[-2]
        
        # Check for crossover
        if prev_macd <= prev_signal and current_macd > current_signal:
            # Bullish crossover
            return StrategyResult(
                Action.BUY,
                0.7,
                {'macd': current_macd, 'signal': current_signal}
            )
        elif prev_macd >= prev_signal and current_macd < current_signal:
            # Bearish crossover
            return StrategyResult(
                Action.SELL,
                0.7,
                {'macd': current_macd, 'signal': current_signal}
            )
        
        return StrategyResult(
            Action.HOLD,
            0.5,
            {'macd': current_macd, 'signal': current_signal}
        )

class CompositeStrategy(BaseStrategy):
    """Combine multiple strategies with weighted voting"""
    
    def __init__(self, strategies: List[Tuple[BaseStrategy, float]]):
        """
        Initialize with a list of (strategy, weight) tuples.
        Higher weights mean more influence on the final decision.
        """
        super().__init__("Composite_Strategy")
        self.strategies = strategies
        # Normalize weights to sum to 1
        total_weight = sum(weight for _, weight in strategies)
        self.weights = [weight / total_weight for _, weight in strategies]
    
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        if not self.strategies:
            return StrategyResult(Action.HOLD, 0.5, {})
        
        results = []
        all_indicators = {}
        
        for (strategy, weight), norm_weight in zip(self.strategies, self.weights):
            result = strategy.analyze(df)
            results.append((result, norm_weight))
            all_indicators[f"{strategy.name}_action"] = result.action.name
            all_indicators[f"{strategy.name}_confidence"] = result.confidence
            all_indicators.update({f"{strategy.name}_{k}": v for k, v in result.indicators.items()})
        
        # Weighted voting
        vote = 0.0
        total_confidence = 0.0
        
        for result, weight in results:
            vote += weight * result.action.value * result.confidence
            total_confidence += weight * result.confidence
        
        # Determine final action
        if abs(vote) > 0.2:  # Threshold for decision
            action = Action.BUY if vote > 0 else Action.SELL
            confidence = min(1.0, abs(vote))
        else:
            action = Action.HOLD
            confidence = 0.5
        
        return StrategyResult(
            action,
            confidence,
            all_indicators
        )