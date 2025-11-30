import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from typing import Dict, List, Tuple, Optional, Any, Union
import gym
from gym import spaces
import ccxt
import ta
from datetime import datetime, timedelta
import time
import json
import logging
import warnings
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dotenv import load_dotenv
import pytz
import matplotlib.pyplot as plt
from matplotlib import style
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import threading
import queue
import sys
import traceback
import io
import math
from collections import deque

# Import the LLM agent
from llm_agent import LLMTradingAgent, LLMStrategy, MarketAnalysis

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging with UTF-8 safe handlers
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_console_stream = sys.stdout or sys.__stdout__
if _console_stream is not None and hasattr(_console_stream, "buffer"):
    try:
        _console_stream = io.TextIOWrapper(_console_stream.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_trading.log', encoding='utf-8'),
        logging.StreamHandler(_console_stream)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG = {
    "trading_fee": 0.001,  # 0.1% trading fee
    "initial_balance": 10000.0,
    "episode_length": 1000,  # Number of steps per episode
    "window_size": 50,  # Lookback window for the state
    "symbols": [],
    "timeframe": "1h",
    "order_type": "MARKET",
    "take_profit_pct": 0.02,
    "stop_loss_pct": 0.01,
    "quote_asset": "USDC",
    "use_all_quote_pairs": True,
    "symbol_refresh_interval": 1800,
    "paper_trading": False,
    "trading_enabled": True,
    "fixed_quote_order_amount": 100.0,
    "train_freq": 1,  # Train every n steps
    "batch_size": 64,  # Batch size for training
    "learning_rate": 0.0003,  # Learning rate for the optimizer
    "gamma": 0.99,  # Discount factor
    "ent_coef": 0.01,  # Entropy coefficient for exploration
    "clip_range": 0.2,  # Clip range for PPO
    "n_steps": 2048,  # Number of steps to run for each environment per update
    "n_epochs": 10,  # Number of epochs when optimizing the surrogate loss
    "gae_lambda": 0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    "max_grad_norm": 0.5,  # The maximum value for the gradient clipping
    "vf_coef": 0.5,  # Value function coefficient for the loss calculation
    "max_symbols_per_cycle": 25,
    "trend_refresh_interval": 300,
    "trend_threshold_pct": 0.5,
    "trend_benchmark_symbol": None,
}

class TradingAction(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

@dataclass
class TradeSignal:
    action: TradingAction
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    confidence: float = 1.0
    reason: str = ""
    metadata: Optional[Dict[str, Any]] = None

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_drawdown = config.get("max_drawdown", 0.2)  # 20% max drawdown
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% of portfolio per position
        self.max_daily_loss = config.get("max_daily_loss", 0.05)  # 5% max daily loss
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        
    def reset_daily(self):
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        
    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        
    def check_risk(self, portfolio_value: float, current_drawdown: float) -> bool:
        """Check if risk limits are exceeded"""
        now = datetime.now()
        
        # Reset daily PnL if it's a new day
        if now.date() > self.last_reset.date():
            self.reset_daily()
            
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * portfolio_value:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False
            
        # Check max drawdown
        if current_drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown limit reached: {current_drawdown:.2%}")
            return False
            
        return True

class AutonomousTrader:
    def __init__(self, exchange: ccxt.Exchange, config: Optional[Dict] = None):
        """
        Initialize the autonomous trading agent.
        
        Args:
            exchange: CCXT exchange instance
            config: Configuration dictionary
        """
        self.exchange = exchange
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir="runs/autonomous_trader")
        self.risk_manager = RiskManager(self.config)
        self.model = None
        self.env = None
        self.is_running = False
        self.training_mode = False
        self.current_episode = 0
        self.best_model_score = -float('inf')
        self.log_queue = queue.Queue()
        self.log_history = deque(maxlen=500)
        self.symbol_filters: Dict[str, Dict[str, float]] = {}
        self._cached_symbols: List[str] = []
        self._last_symbol_refresh: float = 0.0
        self._trend_cache: Dict[str, Dict[str, float]] = {}
        self._last_trend_refresh: float = 0.0
        self._last_market_trend: Tuple[str, float] = ("neutral", 0.0)
        
        # Initialize LLM agent (will be fully initialized in initialize_async)
        self.llm_agent = LLMTradingAgent(
            model_name=os.getenv("LLM_MODEL", "htngtech/deepseek-r1t2-chimera:free"),
            strategy=LLMStrategy.CHAIN_OF_THOUGHT,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            api_key=os.getenv("OPENROUTER_API_KEY")  # Get API key from environment
        )
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize state
        self.reset()
        self._log_event("INFO", "Trader initialized")
        self._refresh_symbol_filters()
    
    async def initialize_async(self):
        """Initialize async components of the trader."""
        try:
            if self.llm_agent:
                await self.llm_agent.initialize_async()
                logger.info(f"Initialized LLM agent with model: {self.llm_agent.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM agent: {e}")
            raise

        # Harden exchange options to avoid unsupported endpoints (esp. testnet SAPI)
        try:
            self.exchange.options = {
                **getattr(self.exchange, 'options', {}),
                'warnOnFetchCurrenciesWithoutPermission': False,
                'fetchCurrencies': False,
                'warnOnFetchOpenOrdersWithoutSymbol': False,
            }
        except Exception as opt_err:
            logger.warning(f"Unable to set exchange options: {opt_err}")

        # Initialize state
        self.reset()
        self._log_event("INFO", "Trader initialized")
        self._refresh_symbol_filters()

    def reset(self):
        """Reset the trader's state"""
        self.portfolio_value = self.config["initial_balance"]
        self.cash = self.config["initial_balance"]
        self.positions = {}  # symbol -> quantity
        self.position_prices = {}  # symbol -> avg entry price
        self.trades = []
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.episode_start_time = datetime.now()
        self.portfolio_history = []

    def _get_symbols(self) -> List[str]:
        quote = self._get_quote_asset()
        configured = self.config.get("symbols")
        if not isinstance(configured, (list, tuple)):
            configured = [configured] if configured else []

        if configured:
            base_universe = [
                sym for sym in configured
                if isinstance(sym, str) and sym.upper().endswith(f"/{quote}")
            ]
        else:
            base_universe = self._refresh_symbol_universe()

        filtered = list(base_universe)
        # Handle both old (float) and new (dict) position formats
        holding_symbols = []
        for sym, pos in self.positions.items():
            if isinstance(pos, dict):
                # New format: {'amount': float, 'value': float, 'price': float}
                if pos.get('amount', 0) > 1e-8:
                    holding_symbols.append(sym)
            elif isinstance(pos, (int, float)) and pos > 1e-8:
                # Old format: float
                holding_symbols.append(sym)
                
        # Add any holding symbols not already in the filtered list
        for sym in holding_symbols:
            if sym.upper().endswith(f"/{quote}") and sym not in filtered:
                filtered.append(sym)
        if not filtered:
            fallback = self.config.get("symbol")
            if fallback and isinstance(fallback, str) and fallback.upper().endswith(f"/{quote}"):
                filtered = [fallback]
            else:
                filtered = [f"BTC/{quote}"]
        return filtered

    def _get_quote_asset(self) -> str:
        return str(self.config.get("quote_asset", "USDC")).upper()

    def _refresh_symbol_universe(self, force: bool = False) -> List[str]:
        quote = self._get_quote_asset()
        use_all_pairs = bool(self.config.get("use_all_quote_pairs", False))

        if not use_all_pairs or not hasattr(self, "exchange"):
            if not self._cached_symbols:
                self._cached_symbols = [f"BTC/{quote}", f"ETH/{quote}"]
            return self._cached_symbols

        refresh_interval = int(self.config.get("symbol_refresh_interval", 1800))
        now = time.time()
        if (
            not force
            and self._last_symbol_refresh
            and (now - self._last_symbol_refresh) < refresh_interval
            and self._cached_symbols
        ):
            return self._cached_symbols

        try:
            markets = self.exchange.load_markets()
            discovered = sorted(
                symbol
                for symbol, market in markets.items()
                if isinstance(symbol, str)
                and symbol.upper().endswith(f"/{quote}")
                and market.get("active", True)
            )
            if discovered:
                self._cached_symbols = discovered
                self._last_symbol_refresh = now
                logger.info(
                    "Discovered %d %s-quoted markets for autonomous trader",
                    len(discovered),
                    quote,
                )
                return self._cached_symbols
            logger.warning("Exchange returned no active %s pairs; keeping previous universe", quote)
        except Exception as err:
            logger.warning("Unable to refresh symbol universe: %s", err)

        if not self._cached_symbols:
            self._cached_symbols = [f"BTC/{quote}", f"ETH/{quote}"]
        return self._cached_symbols

    def _determine_market_trend(self) -> Tuple[str, float]:
        benchmark = self.config.get("trend_benchmark_symbol") or f"BTC/{self._get_quote_asset()}"
        threshold = float(self.config.get("trend_threshold_pct", 0.5))
        try:
            ticker = self.exchange.fetch_ticker(benchmark)
            change = float(ticker.get("percentage") or ticker.get("change") or 0.0)
        except Exception as trend_err:
            logger.warning(f"Unable to fetch benchmark trend ({benchmark}): {trend_err}")
            return self._last_market_trend

        if change >= threshold:
            trend = "bullish"
        elif change <= -threshold:
            trend = "bearish"
        else:
            trend = "neutral"
        self._last_market_trend = (trend, change)
        return self._last_market_trend

    def _refresh_trend_cache(self, symbols: List[str]) -> None:
        refresh_interval = int(self.config.get("trend_refresh_interval", 300))
        now = time.time()
        if self._trend_cache and (now - self._last_trend_refresh) < refresh_interval:
            return

        try:
            tickers = self.exchange.fetch_tickers(symbols)
            cache: Dict[str, Dict[str, float]] = {}
            for sym in symbols:
                info = tickers.get(sym, {}) if isinstance(tickers, dict) else {}
                pct = float(info.get("percentage") or info.get("change") or 0.0)
                volume = float(
                    info.get("quoteVolume")
                    or info.get("baseVolume")
                    or info.get("info", {}).get("quoteVolume", 0)
                )
                cache[sym] = {"pct": pct, "volume": volume}
            if cache:
                self._trend_cache = cache
                self._last_trend_refresh = now
        except Exception as trend_err:
            logger.warning(f"Unable to refresh trend cache: {trend_err}")

    def _rank_symbols_by_trend(self, symbols: List[str], market_trend: str) -> List[str]:
        if not symbols:
            return []
        self._refresh_trend_cache(symbols)

        def get_metrics(sym: str) -> Tuple[float, float]:
            data = self._trend_cache.get(sym, {})
            return data.get("pct", 0.0), data.get("volume", 0.0)

        if market_trend == "bullish":
            ranked = sorted(
                symbols,
                key=lambda sym: (get_metrics(sym)[0], get_metrics(sym)[1]),
                reverse=True,
            )
        elif market_trend == "bearish":
            ranked = sorted(
                symbols,
                key=lambda sym: (get_metrics(sym)[0], -get_metrics(sym)[1]),
            )
        else:
            ranked = sorted(
                symbols,
                key=lambda sym: (abs(get_metrics(sym)[0]), get_metrics(sym)[1]),
                reverse=True,
            )
        return ranked

    def _select_symbols_for_cycle(self) -> List[str]:
        universe = self._get_symbols()
        if not universe:
            return []
        market_trend, change = self._determine_market_trend()
        ranked = self._rank_symbols_by_trend(universe, market_trend)
        max_symbols = max(1, int(self.config.get("max_symbols_per_cycle", 25)))
        selected = ranked[:max_symbols] if ranked else universe[:max_symbols]
        self._log_event(
            "DEBUG",
            "Selected symbol batch",
            {
                "market_trend": market_trend,
                "benchmark_change_pct": change,
                "batch_size": len(selected),
                "max_batch": max_symbols,
            },
        )
        return selected

    def _is_symbol_allowed(self, symbol: str) -> bool:
        if not isinstance(symbol, str):
            return False
        sym = symbol.upper()
        if not sym.endswith(f"/{self._get_quote_asset()}"):
            return False
        if not self.symbol_filters:
            self._refresh_symbol_filters()
        return sym in self.symbol_filters

    def set_fixed_order_amount(self, amount: float) -> None:
        sanitized = max(float(amount), 0.0)
        previous = self.config.get("fixed_quote_order_amount")
        self.config["fixed_quote_order_amount"] = sanitized
        if previous != sanitized:
            logger.info(f"Fixed order amount set to {sanitized:.2f} {self._get_quote_asset()}")

    def _log_event(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.now(),
            "level": level.upper(),
            "message": message,
            "metadata": metadata or {}
        }
        self.log_history.append(entry)
        try:
            self.log_queue.put_nowait(entry)
        except queue.Full:
            pass
        
    def _process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process market data and add technical indicators"""
        if data.empty:
            return data
            
        # Add technical indicators
        data['returns'] = data['close'].pct_change()
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['sma_200'] = ta.trend.sma_indicator(data['close'], window=200)
        data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'])
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_mid'] = bollinger.bollinger_mavg()
        data['bb_low'] = bollinger.bollinger_lband()
        
        # Volume indicators
        data['volume_sma'] = ta.trend.sma_indicator(data['volume'], window=20)
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Drop NaN values
        data = data.dropna()
        
        return data
        
    def get_state(self, data: pd.DataFrame) -> np.ndarray:
        """Get the current state from market data"""
        if len(data) < self.config["window_size"]:
            return np.zeros((self.config["window_size"], len(self._get_feature_columns())))
            
        # Get the most recent window of data
        feature_cols = self._get_feature_columns()
        state_data = data.iloc[-self.config["window_size"]:][feature_cols].copy()
        
        # Normalize the data
        state_data = (state_data - state_data.mean()) / (state_data.std() + 1e-8)
        
        # Convert to numpy array
        state = state_data.values
        
        return state
        
    def _get_feature_columns(self) -> List[str]:
        """Get the list of feature columns to use for the state"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'sma_20', 'sma_50', 'rsi', 'macd',
            'macd_signal', 'macd_diff', 'bb_high', 'bb_mid',
            'bb_low', 'volume_ratio'
        ]
        
    def _execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade based on the signal"""
        try:
            symbol = signal.symbol
            action = signal.action
            price = signal.price
            quantity = signal.quantity
            
            if quantity <= 0:
                logger.warning("Skipping trade with non-positive quantity")
                return False

            if not self._is_symbol_allowed(symbol):
                logger.warning(f"Symbol {symbol} is not quoted in {self._get_quote_asset()} – trade skipped")
                return False

            # Check if we have enough cash/position
            required_quote = self._required_quote(price, quantity)
            if action == TradingAction.BUY and self.cash + 1e-8 < required_quote:
                logger.warning(f"Insufficient {self._get_quote_asset()} to buy {quantity} {symbol} at {price}")
                return False
                
            if action == TradingAction.SELL and self.positions.get(symbol, 0) < quantity:
                logger.warning(f"Insufficient position to sell {quantity} {symbol}")
                return False
            
            max_positions = self.config.get("max_open_positions")
            open_positions = sum(1 for qty in self.positions.values() if qty > 1e-8)

            # Execute the trade
            if action == TradingAction.BUY:
                if symbol not in self.positions and max_positions and open_positions >= max_positions:
                    logger.warning("Max open positions reached; skipping new BUY order")
                    return False
                cost = required_quote
                self.cash -= cost
                prev_qty = self.positions.get(symbol, 0.0)
                new_qty = prev_qty + quantity
                self.positions[symbol] = new_qty
                prev_price = self.position_prices.get(symbol, price)
                if new_qty > 0:
                    self.position_prices[symbol] = ((prev_qty * prev_price) + (quantity * price)) / new_qty
                logger.info(f"Bought {quantity} {symbol} at {price:.2f} (cost: {cost:.2f})")
                
            elif action == TradingAction.SELL:
                revenue = price * quantity * (1 - self.config["trading_fee"])
                self.cash += revenue
                prev_qty = self.positions.get(symbol, 0.0)
                new_qty = prev_qty - quantity
                if new_qty <= 1e-8:
                    self.positions.pop(symbol, None)
                    self.position_prices.pop(symbol, None)
                else:
                    self.positions[symbol] = new_qty
                    self.position_prices[symbol] = self.position_prices.get(symbol, price)
                logger.info(f"Sold {quantity} {symbol} at {price:.2f} (revenue: {revenue:.2f})")
                
            elif action == TradingAction.CLOSE:
                # Close the entire position
                existing_qty = self.positions.get(symbol, 0.0)
                if existing_qty > 0:
                    revenue = price * existing_qty * (1 - self.config["trading_fee"])
                    self.cash += revenue
                    self.positions.pop(symbol, None)
                    self.position_prices.pop(symbol, None)
                    logger.info(f"Closed long position of {existing_qty} {symbol} at {price:.2f}")
                else:
                    logger.warning(f"No open position to close for {symbol}")
                    return False
            
            # Record the trade
            self.trades.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action.name,
                'price': price,
                'quantity': quantity,
                'portfolio_value': self.get_portfolio_value(),
                'reason': signal.reason,
                'confidence': signal.confidence
            })
            self._log_event("INFO", "Trade executed", {
                "symbol": symbol,
                "action": action.name,
                "price": price,
                "quantity": quantity
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _generate_trade_signal(self, symbol: str, data: pd.DataFrame) -> TradeSignal:
        """
        Generate a trade signal based on the current market data.
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV data
            
        Returns:
            TradeSignal object with the generated signal
        """
        try:
            # Get the current price and indicators
            current_price = data['close'].iloc[-1]
            
            # Get the current position
            position = self.positions.get(symbol, 0.0)
            
            # Get the current portfolio value
            portfolio_value = self._get_portfolio_value()
            
            # Generate a trade signal using the LLM agent
            signal = await self.llm_agent.generate_trade_signal(
                symbol=symbol,
                portfolio={"USDT": portfolio_value},
                current_price=current_price,
                position_size=0.1,  # 10% of portfolio
                risk_per_trade=0.01  # 1% risk per trade
            )
            
            # Create a TradeSignal object
            action = TradingAction.HOLD
            if signal["action"] == "BUY":
                action = TradingAction.BUY
            elif signal["action"] == "SELL":
                action = TradingAction.SELL
            
            return TradeSignal(
                action=action,
                symbol=symbol,
                price=current_price,
                quantity=signal["size"],
                timestamp=datetime.now(),
                confidence=signal["confidence"],
                reason=signal["reasoning"],
                metadata={
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "indicators": signal.get("indicators", {})
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating trade signal for {symbol}: {e}")
            return TradeSignal(
                action=TradingAction.HOLD,
                symbol=symbol,
                price=0.0,
                quantity=0.0,
                timestamp=datetime.now(),
                confidence=0.0,
                reason=f"Error generating signal: {str(e)}",
                metadata={}
            )

    def _sync_account_state(self):
        """Synchronize cash and positions with the exchange when live trading."""
        try:
            if not self.config.get('trading_enabled'):
                return
                
            # Fetch the latest balance from the exchange
            balance = self.exchange.fetch_balance()
            quote_asset = self._get_quote_asset()
            
            # Update available cash in the quote currency (USDC/USDT)
            quote_entry = balance.get(quote_asset, {})
            self.cash = float(quote_entry.get('free', quote_entry.get('total', self.cash)))
            
            # Track total portfolio value starting with available cash
            total_value = self.cash
            
            # Update positions and calculate their value
            tracked_symbols = self._get_symbols()
            self.positions = {}
            
            # First, get all assets with non-zero balance
            for currency, amount in balance['total'].items():
                if float(amount) > 0 and currency != quote_asset:
                    # Find a trading pair for this asset to the quote currency
                    symbol = f"{currency}/{quote_asset}"
                    if symbol in tracked_symbols:
                        try:
                            # Get current price for the asset
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = ticker['last']
                            position_value = float(amount) * price
                            total_value += position_value
                            self.positions[symbol] = {
                                'amount': float(amount),
                                'value': position_value,
                                'price': price
                            }
                        except Exception as e:
                            logger.warning(f"Could not fetch price for {symbol}: {e}")
            
            # Update the portfolio value
            self.portfolio_value = total_value
            
            # Log the updated values
            logger.info(f"Updated portfolio - Cash: {self.cash:.2f} {quote_asset}, "
                       f"Positions: {len(self.positions)}, Total Value: {total_value:.2f} {quote_asset}")
                        
        except Exception as sync_err:
            logger.warning(f"Account sync failed: {sync_err}")
            if "symbol filters" in str(sync_err).lower():
                self.symbol_filters = {}
            # Re-raise the exception to be handled by the caller
            raise

    def _required_quote(self, price: float, quantity: float) -> float:
        fee_rate = self.config.get("trading_fee", 0.0)
        return price * quantity * (1 + fee_rate)

    def _refresh_symbol_filters(self) -> None:
        try:
            markets = self.exchange.load_markets()
            quote = self._get_quote_asset()
            filters: Dict[str, Dict[str, float]] = {}
            for sym, market in markets.items():
                if '/' not in sym:
                    continue
                if sym.split('/')[1].upper() != quote:
                    continue
                info_filters = {f.get('filterType'): f for f in market.get('info', {}).get('filters', []) if isinstance(f, dict)}
                limits = market.get('limits', {})
                min_amount = float(info_filters.get('LOT_SIZE', {}).get('minQty', limits.get('amount', {}).get('min', 0)) or 0)
                step = float(info_filters.get('LOT_SIZE', {}).get('stepSize', 0) or market.get('lot') or 0)
                min_notional = float(info_filters.get('MIN_NOTIONAL', {}).get('minNotional', limits.get('cost', {}).get('min', 0)) or 0)
                precision = market.get('precision', {}).get('amount')
                filters[sym.upper()] = {
                    'min_amount': min_amount,
                    'min_notional': min_notional,
                    'step': step,
                    'precision': precision
                }
            if filters:
                self.symbol_filters = filters
        except Exception as err:
            logger.warning(f"Unable to refresh symbol filters: {err}")

    def _get_symbol_filter(self, symbol: str) -> Optional[Dict[str, float]]:
        if not self.symbol_filters:
            self._refresh_symbol_filters()
        return self.symbol_filters.get(symbol.upper())

    def _ensure_minimums(
        self,
        symbol: str,
        price: float,
        quantity: float,
        *,
        grow: bool = True,
        max_available: Optional[float] = None,
    ) -> float:
        """Ensure order quantities meet exchange minimums and account for available balance.
        
        Args:
            symbol: Trading pair symbol
            price: Current price of the asset
            quantity: Desired quantity to trade
            grow: If True, allow increasing position size
            max_available: Maximum available quantity to trade (for sells/closes)
            
        Returns:
            Adjusted quantity that meets exchange requirements and available balance
        """
        # Get exchange filters for the symbol
        filt = self._get_symbol_filter(symbol)
        if not filt or quantity <= 0:
            return max(quantity, 0.0)
        
        # Get minimum order requirements
        min_amount = float(filt.get('min_amount', 0.0))
        min_notional = float(filt.get('min_notional', 0.0))
        step = float(filt.get('step', 0.0))
        precision = filt.get('precision')
        
        # Apply minimum amount and notional
        adj_qty = max(quantity, min_amount)
        if min_notional and price > 0:
            min_qty = min_notional / price
            adj_qty = max(adj_qty, min_qty)
        
        # Apply step size and precision
        if step and step > 0:
            adj_qty = math.floor(adj_qty / step) * step
        elif precision is not None:
            adj_qty = round(adj_qty, precision)
        
        # Handle max available quantity for sells/closes
        if max_available is not None:
            if isinstance(max_available, dict):
                max_available = max_available.get('amount', 0.0)
            max_available = float(max_available or 0)
            
            if not grow:
                # For non-growing orders (like sells), cap at max_available
                adj_qty = min(adj_qty, max_available)
                if adj_qty < min_amount:
                    return 0.0
            else:
                # For growing orders (like buys), ensure we don't exceed max_available
                adj_qty = min(adj_qty, max_available) if max_available > 0 else adj_qty
        
        return max(adj_qty, 0.0)

    def _fetch_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as price_err:
            logger.error(f"Failed to fetch price for {symbol}: {price_err}")
            return self.position_prices.get(symbol, 0.0)

    def _fetch_market_window(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        window = self.config.get("window_size", 50)
        limit = limit or max(window + 200, window * 3)
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = self._process_market_data(df)
            return df
        except Exception as err:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {err}")
            self._log_event("ERROR", "OHLCV fetch failed", {"symbol": symbol, "error": str(err)})
            return None

    def get_portfolio_value(self, symbol: Optional[str] = None, price_override: Optional[float] = None) -> float:
        total = self.cash
        # Copy items to avoid "dictionary changed size during iteration" when other threads update positions
        for sym, position in list(self.positions.items()):
            if isinstance(position, dict):
                # New format: {'amount': float, 'value': float, 'price': float}
                amount = position.get('amount', 0)
                if abs(amount) < 1e-8:
                    continue
                if symbol and sym == symbol and price_override is not None:
                    total += amount * price_override
                else:
                    total += position.get('value', 0)
            else:
                # Old format: float
                if abs(position) < 1e-8:
                    continue
                use_price = price_override if symbol and sym == symbol and price_override is not None else self._fetch_price(sym)
                total += position * use_price
                
        self.portfolio_history.append({'timestamp': datetime.now(), 'value': total})
        return total
        
    def get_current_drawdown(self) -> float:
        """Calculate the current drawdown"""
        if not self.trades:
            return 0.0
            
        peak = max(trade['portfolio_value'] for trade in self.trades) if self.trades else self.portfolio_value
        current = self.get_portfolio_value()
        return (peak - current) / peak if peak > 0 else 0.0
        
    def get_current_price(self, symbol: Optional[str] = None) -> float:
        """Get the current price of a tracked asset"""
        target = symbol or self._get_symbols()[0]
        return self._fetch_price(target)
        
    def run_episode(self, data: pd.DataFrame) -> float:
        """Run a single episode of trading"""
        self.reset()
        episode_reward = 0
        
        for i in range(len(data) - self.config["window_size"] - 1):
            if self.done:
                break
                
            # Get the current state
            state = self.get_state(data.iloc[i:i + self.config["window_size"]])
            
            # Get action from the model
            action, _ = self.model.predict(state, deterministic=True)
            
            # Execute the action
            signal = self._action_to_signal(action, data.iloc[i + self.config["window_size"]])
            self._execute_trade(signal)
            
            # Get the next state and reward
            next_state = self.get_state(data.iloc[i+1:i + self.config["window_size"] + 1])
            reward = self._calculate_reward(signal, data.iloc[i + self.config["window_size"]])
            
            # Update the model
            if self.training_mode:
                self.model.learn(
                    total_timesteps=1,
                    reset_num_timesteps=False,
                    tb_log_name="autonomous_trader"
                )
            
            # Update episode reward
            episode_reward += reward
            
            # Update step counter
            self.current_step += 1
            
            # Log progress
            if self.current_step % 100 == 0:
                logger.info(f"Step {self.current_step}, Reward: {reward:.2f}, Portfolio: {self.get_portfolio_value():.2f}")
        
        return episode_reward
        
    def _get_action_reason(self, action: TradingAction, data: pd.Series) -> str:
        """Generate a human-readable reason for the action based on market conditions"""
        reasons = []

        rsi = data.get('rsi')
        if rsi is not None and not np.isnan(rsi):
            if rsi > 70:
                reasons.append("RSI indicates overbought conditions")
            elif rsi < 30:
                reasons.append("RSI indicates oversold conditions")

        if 'macd' in data and 'macd_signal' in data:
            macd_val = data.get('macd')
            macd_sig = data.get('macd_signal')
            if macd_val is not None and macd_sig is not None:
                if macd_val > macd_sig:
                    reasons.append("MACD above signal line (bullish)")
                else:
                    reasons.append("MACD below signal line (bearish)")

        close_price = data.get('close')
        sma_50 = data.get('sma_50')
        sma_200 = data.get('sma_200') if 'sma_200' in data else None
        if close_price is not None:
            if sma_50 is not None and not np.isnan(sma_50):
                if close_price > sma_50:
                    reasons.append("Price above 50-period SMA (short-term bullish)")
                else:
                    reasons.append("Price below 50-period SMA (short-term bearish)")
            if sma_200 is not None and not np.isnan(sma_200):
                if close_price > sma_200:
                    reasons.append("Price above 200-period SMA (long-term bullish)")
                else:
                    reasons.append("Price below 200-period SMA (long-term bearish)")

        if not reasons:
            reasons.append("Insufficient indicator data; defaulting to model policy")

        if action == TradingAction.BUY:
            reasons.append("AI signals a buying opportunity")
        elif action == TradingAction.SELL:
            reasons.append("AI signals a selling opportunity")
        elif action == TradingAction.CLOSE:
            reasons.append("AI suggests closing the position")
        else:
            reasons.append("AI suggests holding current position")

        return "; ".join(reasons)

    def _action_to_signal(self, action: int, data: pd.Series, symbol_override: Optional[str] = None) -> TradeSignal:
        """Convert an action index to a trade signal with detailed logging"""
        action_enum = TradingAction(action)
        symbol = symbol_override or self._get_symbols()[0]
        price = data['close']

        position_size = self._calculate_position_size(symbol, price, action_enum)
        grow = action_enum == TradingAction.BUY
        
        # Handle max_available for SELL/CLOSE actions
        max_available = None
        if action_enum in (TradingAction.SELL, TradingAction.CLOSE):
            position = self.positions.get(symbol, 0.0)
            if isinstance(position, dict):
                max_available = position.get('amount', 0.0)
            else:
                max_available = float(position) if position else 0.0
                
        quantity = self._ensure_minimums(symbol, price, position_size, grow=grow, max_available=max_available)

        if action_enum in (TradingAction.SELL, TradingAction.CLOSE) and quantity <= 0:
            logger.info(f"Skipping {action_enum.name} for {symbol}: insufficient holdings to satisfy exchange minimums")
        if action_enum == TradingAction.BUY and quantity > position_size + 1e-9 and quantity * price > self.cash:
            logger.warning(f"Adjusted BUY size for {symbol} exceeds available cash; capping to balance")
            affordable_qty = max(self.cash / price, 0.0)
            quantity = self._ensure_minimums(symbol, price, affordable_qty, grow=False)

        signal_qty = quantity if quantity > 0 else 0.0

        reason = self._get_action_reason(action_enum, data)

        signal = TradeSignal(
            action=action_enum,
            symbol=symbol,
            price=price,
            quantity=signal_qty,
            timestamp=datetime.now(),
            confidence=1.0,
            reason=reason
        )

        self._log_market_snapshot(signal, data)
        self._log_event("INFO", "Signal generated", {"symbol": symbol, "action": action_enum.name, "qty": position_size})
        return signal

    def _log_market_snapshot(self, signal: TradeSignal, data: pd.Series) -> None:
        """Log the market context and AI decision for observability"""
        logger.info("────────── AI DECISION ──────────")
        logger.info(f"Time: {signal.timestamp.isoformat()} | Symbol: {signal.symbol}")
        logger.info(f"Action: {signal.action.name} | Price: {signal.price:.2f} | Quantity: {signal.quantity:.6f}")
        logger.info(f"Reason: {signal.reason}")
        logger.info(
            "Indicators ⇒ RSI: %.2f | MACD: %.4f | Signal: %.4f | BB Mid: %.2f" % (
                data.get('rsi', float('nan')),
                data.get('macd', float('nan')),
                data.get('macd_signal', float('nan')),
                data.get('bb_mid', float('nan')),
            )
        )
        logger.info("─────────────────────────────────")
        
    def _calculate_position_size(self, symbol: str, price: float, action: TradingAction) -> float:
        """Calculate position size based on risk management rules and holdings
        
        Args:
            symbol: The trading pair symbol
            price: Current price of the asset
            action: The trading action to take
            
        Returns:
            float: The calculated position size in the asset's unit
        """
        if price <= 0:
            return 0.0

        # Get the current position, handling both dict and float formats
        position = self.positions.get(symbol, 0.0)
        if isinstance(position, dict):
            held = position.get('amount', 0.0)
        else:
            held = float(position) if position else 0.0
            
        if action == TradingAction.CLOSE:
            return abs(held)  # Return absolute value for closing

        # Calculate position size based on risk management rules
        fixed_amount = float(self.config.get("fixed_quote_order_amount", 0.0) or 0.0)
        risk_quote = self.portfolio_value * self.risk_manager.max_position_size
        spendable_quote = min(risk_quote, self.cash)
        
        if fixed_amount > 0:
            spendable_quote = min(spendable_quote, fixed_amount)

        # Calculate position size based on action
        if action == TradingAction.BUY:
            return max(spendable_quote / price, 0.0)

        if action == TradingAction.SELL:
            if held <= 0:
                return 0.0
            desired = spendable_quote / price if spendable_quote > 0 else abs(held)
            return min(max(desired, 0.0), abs(held))

        return 0.0
        
    def _calculate_reward(self, signal: TradeSignal, data: pd.Series) -> float:
        """Calculate the reward for the taken action with detailed logging"""
        prev_value = self.portfolio_value
        current_value = self.get_portfolio_value(symbol=signal.symbol, price_override=data['close'])
        pnl = current_value - prev_value
        pnl_pct = (pnl / prev_value * 100) if prev_value > 0 else 0
        reward = pnl_pct / 100

        logger.info("🎯 Reward Calculation")
        logger.info(f"  Previous Portfolio Value: ${prev_value:,.2f}")
        logger.info(f"  Current Portfolio Value:  ${current_value:,.2f}")
        logger.info(f"  P&L: {pnl:+,.2f} USD ({pnl_pct:+.2f}%)")

        drawdown = self.get_current_drawdown()
        if drawdown > self.risk_manager.max_drawdown:
            penalty = 10.0
            reward -= penalty
            logger.warning(
                f"  ⚠️ Drawdown penalty applied: -{penalty:.2f} (Current drawdown: {drawdown * 100:.2f}%)"
            )
            self._log_event("WARNING", "Drawdown penalty applied", {"drawdown_pct": drawdown * 100})

        self.portfolio_value = current_value
        logger.info(f"  Final Reward: {reward:+.6f}")
        self._log_event("INFO", "Reward updated", {"pnl": pnl, "reward": reward})
        return reward

    def _execute_live_trade(self, signal: TradeSignal, order_type: Optional[str] = None) -> Optional[dict]:
        """Send an order to the exchange or simulate it depending on config"""
        order_type = (order_type or self.config.get("order_type", "MARKET")).upper()
        symbol = signal.symbol
        side = 'buy' if signal.action in (TradingAction.BUY,) else 'sell'

        if signal.quantity <= 0:
            logger.info("No live order placed because calculated quantity is zero")
            return None

        if not self._is_symbol_allowed(symbol):
            logger.warning(f"Rejected order for {symbol}: not quoted in {self._get_quote_asset()}")
            return None

        if signal.action in (TradingAction.HOLD, TradingAction.CLOSE):
            logger.info("No live order required for HOLD/CLOSE action – updating internal state only")
            self._execute_trade(signal)
            self._log_event("INFO", "Hold/Close action processed", {"symbol": symbol})
            return None

        # Default to paper trading unless explicitly enabled
        if not self.config.get("trading_enabled") or self.config.get("paper_trading", True):
            logger.info(
                f"[PAPER TRADE] {order_type} {side.upper()} {signal.quantity:.6f} {symbol} @ {signal.price:.2f}"
            )
            self._execute_trade(signal)
            self._log_event("INFO", "Paper trade executed", {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": signal.quantity,
            })
            return {
                'id': f"PAPER_{int(time.time() * 1000)}",
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'price': signal.price,
                'amount': signal.quantity,
                'timestamp': int(time.time() * 1000)
            }

        try:
            logger.info(
                f"Submitting {order_type} {side.upper()} order for {signal.quantity:.6f} {symbol}"
            )
            if order_type == 'MARKET':
                order = self.exchange.create_market_order(symbol, side, signal.quantity)
            elif order_type == 'LIMIT':
                order = self.exchange.create_limit_order(symbol, side, signal.quantity, signal.price)
            elif order_type == 'OCO':
                params = self._build_oco_params(signal, side)
                order = self.exchange.private_post_order_oco(params)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            logger.info(f"✅ Order accepted by exchange: {order.get('id', 'UNKNOWN_ID')}")
            self._execute_trade(signal)
            self._log_event("INFO", "Live order accepted", {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "order_id": order.get('id')
            })
            return order
        except Exception as exc:
            logger.error(f"❌ Failed to place {order_type} order: {exc}")
            logger.exception("Order submission error")
            self._log_event("ERROR", "Order submission failed", {"error": str(exc)})
            return None

    def _build_oco_params(self, signal: TradeSignal, side: str) -> Dict[str, Any]:
        """Construct parameters for an OCO order using config risk settings"""
        take_profit_pct = self.config.get("take_profit_pct", 0.02)
        stop_loss_pct = self.config.get("stop_loss_pct", 0.01)
        price = signal.price

        if side == 'buy':
            tp_price = price * (1 + take_profit_pct)
            sl_price = price * (1 - stop_loss_pct)
        else:
            tp_price = price * (1 - take_profit_pct)
            sl_price = price * (1 + stop_loss_pct)

        params = {
            'symbol': signal.symbol.replace('/', ''),
            'side': side.upper(),
            'quantity': signal.quantity,
            'price': f"{tp_price:.2f}",
            'stopPrice': f"{sl_price:.2f}",
            'stopLimitPrice': f"{sl_price:.2f}",
            'stopLimitTimeInForce': 'GTC'
        }
        logger.info(
            f"OCO params ⇒ TP: {tp_price:.2f}, SL: {sl_price:.2f}, Qty: {signal.quantity:.6f}"
        )
        return params
        
    def load_model(self, model_path: str = "models/autonomous_trader_final"):
        """Load a trained model from disk"""
        try:
            # Check if the model file exists
            if not os.path.exists(f"{model_path}.zip"):
                raise FileNotFoundError(f"Model file not found at {model_path}.zip")
                
            # Load the model
            self.model = PPO.load(model_path, device=self.device)
            logger.info(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    def train(self, data: pd.DataFrame, total_timesteps: int = 10000):
        """Train the trading agent"""
        self.training_mode = True
        
        # Create the environment
        env = self._create_environment(data)
        
        # Initialize the model
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            vf_coef=self.config["vf_coef"],
            device=self.device
        )
        
        # Set up callbacks
        eval_callback = EvalCallback(
            env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=1000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="autonomous_trader"
        )
        
        # Save the final model
        self.model.save("models/autonomous_trader_final")
        
        self.training_mode = False
        
    def _create_environment(self, data: pd.DataFrame):
        """Create a trading environment"""
        # This is a simplified version - in a real implementation, you would use a custom environment
        # that inherits from gym.Env
        class TradingEnv(gym.Env):
            def __init__(self, data, window_size):
                super(TradingEnv, self).__init__()
                self.data = data
                self.window_size = window_size
                self.current_step = window_size
                
                # Define action and observation space
                self.action_space = spaces.Discrete(len(TradingAction))
                self.observation_space = spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(window_size, len(self._get_feature_columns())), 
                    dtype=np.float32
                )
                
            def _get_feature_columns(self):
                return [
                    'open', 'high', 'low', 'close', 'volume',
                    'returns', 'sma_20', 'sma_50', 'rsi', 'macd',
                    'macd_signal', 'macd_diff', 'bb_high', 'bb_mid',
                    'bb_low', 'volume_ratio'
                ]
                
            def reset(self):
                self.current_step = self.window_size
                return self._get_observation()
                
            def step(self, action):
                # Execute the action
                # ...
                
                # Get the next state
                self.current_step += 1
                next_state = self._get_observation()
                
                # Calculate reward
                reward = 0.0  # Implement reward calculation
                
                # Check if episode is done
                done = self.current_step >= len(self.data) - 1
                
                # Additional info
                info = {}
                
                return next_state, reward, done, info
                
            def _get_observation(self):
                # Get the current window of data
                obs = self.data.iloc[
                    self.current_step - self.window_size : self.current_step
                ][self._get_feature_columns()].values
                
                return obs.astype(np.float32)
                
        return DummyVecEnv([lambda: TradingEnv(data, self.config["window_size"])])
        
    def start_trading(self, live_data_callback=None):
        """Start the autonomous trading loop with live data and full logging"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.is_running = True
        timeframe = self.config.get("timeframe", "1h")
        refresh_seconds = self.config.get("data_refresh_interval", 60)
        current_symbols = self._get_symbols()
        self._log_event("INFO", "Trading loop started", {"symbols": current_symbols})

        def trading_loop():
            try:
                while self.is_running:
                    self._sync_account_state()
                    symbols = self._select_symbols_for_cycle()
                    if not symbols:
                        logger.warning("No symbols available for this cycle; sleeping")
                        time.sleep(refresh_seconds)
                        continue
                    for symbol in symbols:
                        if not self._is_symbol_allowed(symbol):
                            logger.warning(f"Skipping unsupported symbol {symbol}")
                            continue
                        try:
                            df = self._fetch_market_window(symbol, timeframe)
                            if df is None or len(df) < self.config["window_size"]:
                                logger.warning(f"Not enough data returned from exchange for {symbol}; retrying...")
                                self._log_event("WARNING", "Insufficient market data", {"symbol": symbol})
                                time.sleep(2)
                                continue

                            state = self.get_state(df)
                            action, _ = self.model.predict(state, deterministic=True)
                            latest_row = df.iloc[-1]
                            signal = self._action_to_signal(action, latest_row, symbol_override=symbol)

                            order_type = self.config.get("order_type", "MARKET")
                            self._execute_live_trade(signal, order_type=order_type)

                            if live_data_callback:
                                live_data_callback(latest_row, signal)
                        except Exception as loop_err:
                            logger.error(f"Trading loop iteration failed for {symbol}: {loop_err}")
                            logger.exception("Loop error details")
                            self._log_event("ERROR", "Trading iteration failed", {"symbol": symbol, "error": str(loop_err)})
                            time.sleep(5)
                    time.sleep(refresh_seconds)
            finally:
                logger.info("🛑 Autonomous trading loop stopped")
                self._log_event("INFO", "Trading loop stopped")

        self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
        self.trading_thread.start()
        
    def stop_trading(self):
        """Stop the autonomous trading loop"""
        self.is_running = False
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join()
            
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get current trading statistics"""
        self._sync_account_state()
        portfolio_value = self.get_portfolio_value()
        num_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        returns = [t['portfolio_value'] / self.config["initial_balance"] - 1 for t in self.trades] if self.trades else []
        total_return = (portfolio_value / self.config["initial_balance"] - 1) * 100

        peak = max([self.config["initial_balance"]] + [t['portfolio_value'] for t in self.trades])
        max_drawdown = 0
        for t in self.trades:
            if t['portfolio_value'] > peak:
                peak = t['portfolio_value']
            drawdown = (peak - t['portfolio_value']) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns) if returns else 0,
            'current_drawdown_pct': self.get_current_drawdown() * 100,
            'is_running': self.is_running,
            'trading_mode': 'Live' if not self.config.get('paper_trading', True) else 'Paper',
            'last_trade': self.trades[-1] if self.trades else None,
            'positions_snapshot': self.get_positions_snapshot(),
            'performance_history': self.get_performance_history()
        }

    def emergency_sell_all(self):
        """Immediately liquidate all positions to USDC.
        
        This method will attempt to sell all non-USDC positions at market price.
        """
        try:
            self._log_event("WARNING", "Initiating emergency sell of all positions")
            
            # Get current positions
            self._sync_account_state()
            
            # Get the quote currency (USDC or USDT)
            quote_asset = self._get_quote_asset()
            
            # Get all symbols that can be traded to the quote asset
            symbols = self._get_symbols()
            
            # Filter symbols that can be directly converted to the quote asset
            target_symbols = [s for s in symbols if s.endswith(f'/{quote_asset}')]
            
            # For each position, create a market sell order
            for symbol, amount in self.positions.items():
                if abs(amount) > 0:  # Only process non-zero positions
                    try:
                        # If we have a direct trading pair to the quote asset
                        if symbol in target_symbols:
                            order_side = 'sell' if amount > 0 else 'buy'
                            order_amount = abs(amount)
                            
                            # Get current price and calculate order size
                            ticker = self.exchange.fetch_ticker(symbol)
                            current_price = ticker['last']
                            
                            # Ensure we don't exceed available balance
                            if order_side == 'sell':
                                order_amount = min(order_amount, self.positions[symbol])
                            
                            # Place the market order
                            if order_amount > 0:
                                self._log_event("WARNING", f"Placing {order_side} order for {order_amount} {symbol} at market price")
                                order = self.exchange.create_market_order(
                                    symbol=symbol,
                                    side=order_side,
                                    amount=order_amount
                                )
                                self._log_event("INFO", f"Emergency order placed: {order}")
                        
                    except Exception as e:
                        self._log_event("ERROR", f"Error during emergency sell of {symbol}: {str(e)}")
                        continue
            
            # Update positions after selling
            self._sync_account_state()
            self._log_event("INFO", "Emergency sell operation completed")
            
        except Exception as e:
            self._log_event("ERROR", f"Failed to execute emergency sell: {str(e)}")
            raise

    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get historical performance metrics for analysis and visualization.
        
        Returns:
            List of dictionaries containing performance metrics at each time point
        """
        history = []
        
        # If no trades, return empty list
        if not hasattr(self, 'trades') or not self.trades:
            return []
            
        # Initialize starting values
        initial_balance = self.config.get('initial_balance', 10000)
        running_balance = initial_balance
        peak_balance = initial_balance
        max_drawdown = 0
        
        # Add initial point
        history.append({
            'timestamp': datetime.now() - timedelta(days=1),  # Start from yesterday
            'balance': initial_balance,
            'equity': initial_balance,
            'drawdown': 0.0,
            'daily_pnl': 0.0,
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0
        })
        
        # Process each trade
        for i, trade in enumerate(self.trades):
            if 'portfolio_value' in trade:
                equity = trade['portfolio_value']
                daily_pnl = equity - running_balance
                
                # Update peak and drawdown
                if equity > peak_balance:
                    peak_balance = equity
                
                current_drawdown = (peak_balance - equity) / peak_balance if peak_balance > 0 else 0
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Calculate win rate
                winning_trades = sum(1 for t in self.trades[:i+1] if t.get('pnl', 0) > 0)
                win_rate = (winning_trades / (i + 1)) * 100 if (i + 1) > 0 else 0
                
                # Add to history
                history.append({
                    'timestamp': trade.get('timestamp', datetime.now() - timedelta(minutes=len(self.trades)-i)),
                    'balance': running_balance,
                    'equity': equity,
                    'drawdown': current_drawdown * 100,  # as percentage
                    'daily_pnl': daily_pnl,
                    'total_return': ((equity / initial_balance) - 1) * 100,  # as percentage
                    'num_trades': i + 1,
                    'win_rate': win_rate
                })
                
                running_balance = equity
                
        return history
        
    def get_positions_snapshot(self) -> List[Dict[str, Any]]:
        snapshots = []
        for symbol, position in self.positions.items():
            if isinstance(position, dict):
                # New format: {'amount': float, 'value': float, 'price': float}
                amount = position.get('amount', 0)
                if abs(amount) < 1e-8:
                    continue
                current_price = position.get('price', 0)
                entry_price = self.position_prices.get(symbol, current_price)
                pnl_value = (current_price - entry_price) * amount
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                snapshots.append({
                    'symbol': symbol,
                    'size': amount,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl_value': pnl_value,
                    'pnl_pct': pnl_pct,
                    'value': position.get('value', 0)
                })
            else:
                # Old format: float
                if abs(position) < 1e-8:
                    continue
                entry_price = self.position_prices.get(symbol, 0.0)
                current_price = self._fetch_price(symbol)
                pnl_value = (current_price - entry_price) * position
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                snapshots.append({
                    'symbol': symbol,
                    'size': position,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl_value': pnl_value,
                    'pnl_pct': pnl_pct,
                    'value': position * current_price
                })
class AutonomousTradingUI:
    """Enhanced Streamlit UI for the autonomous trading system"""
    
    def __init__(self, trader):
        self.trader = trader
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the enhanced Streamlit UI with additional controls"""
        st.set_page_config(
            page_title="🤖 Advanced Trading System",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .main-header { font-size: 24px; font-weight: bold; color: #1f77b4; }
            .metric-card { 
                border-radius: 5px; 
                padding: 15px; 
                background-color: #f0f2f6; 
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .positive { color: #2ecc71; }
            .negative { color: #e74c3c; }
            .log-entry { 
                padding: 8px; 
                margin: 2px 0; 
                border-radius: 3px; 
                background-color: #f8f9fa;
                font-family: monospace;
                font-size: 0.9em;
            }
            .section {
                margin-top: 20px;
                padding: 15px;
                background-color: #ffffff;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
        </style>
        """, unsafe_allow_html=True)
        
    def render(self):
        """Render the main UI"""
        st.title("🤖 Advanced Trading System")
        
        # Sidebar with controls
        self._render_sidebar()
        
        # Main content area
        self._render_main_content()
        
        # Emergency controls at the bottom
        self._render_emergency_controls()
    
    def _render_sidebar(self):
        """Render the sidebar controls"""
        with st.sidebar:
            st.header("🛠️ Trading Controls")
            
            # Timeframe selection
            self.timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                index=4
            )
            
            # Position sizing
            st.subheader("Position Management")
            self.position_size = st.slider(
                "Position Size (% of portfolio)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1
            ) / 100.0
            
            # Risk management
            st.subheader("Risk Management")
            self.risk_per_trade = st.slider(
                "Risk per Trade (% of portfolio)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            ) / 100.0
            
            self.take_profit = st.number_input(
                "Take Profit (%)",
                min_value=0.1,
                max_value=100.0,
                value=2.0,
                step=0.1
            )
            
            self.stop_loss = st.number_input(
                "Stop Loss (%)",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1
            )
            
            # Strategy selection
            st.subheader("Trading Strategy")
            self.strategy = st.selectbox(
                "Trading Strategy",
                ["Trend Following", "Mean Reversion", "Breakout", "Swing Trading"],
                index=0
            )
            
            # AI Settings
            st.subheader("AI Settings")
            self.ai_confidence = st.slider(
                "Minimum Confidence Level",
                min_value=50,
                max_value=100,
                value=70,
                step=1
            )
            
            # Start/Stop trading
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                    self._start_trading()
            with col2:
                if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
                    self._stop_trading()
    
    def _render_main_content(self):
        """Render the main content area"""
        # Market overview
        self._render_market_overview()
        
        # Trading view
        self._render_trading_view()
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Trading signals
        self._render_trading_signals()
    
    def _render_emergency_controls(self):
        """Render emergency controls at the bottom"""
        st.markdown("---")
        st.subheader("🚨 Emergency Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛑 Emergency Sell All", type="primary", 
                        help="Immediately close all positions at market price"):
                self._emergency_sell_all()
        
        with col2:
            if st.button("⏸️ Pause Trading", 
                        help="Pause all trading activity"):
                self._pause_trading()
        
        with col3:
            if st.button("🔄 Reset Strategy", 
                        help="Reset all strategy parameters to default"):
                self._reset_strategy()
    
    def _render_market_overview(self):
        """Render the market overview section"""
        with st.container():
            st.subheader("📊 Market Overview")
            
            # Market data
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BTC/USDT", "$45,321.50", "+2.3%")
            with col2:
                st.metric("ETH/USDT", "$2,345.67", "+1.8%")
            with col3:
                st.metric("24h Volume", "$45.2B", "+5.2%")
            with col4:
                st.metric("Market Sentiment", "Bullish", "3.2%")
            
            # Price chart
            st.line_chart(pd.DataFrame({
                'Price': [40000, 40500, 39800, 41000, 41500, 42000, 42500, 43000, 43500, 44000],
                'SMA 20': [None, None, None, 39800, 40200, 40800, 41200, 41800, 42200, 42800],
                'SMA 50': [None] * 5 + [40500, 40800, 41200, 41500, 41800, 42000]
            }))
    
    def _render_trading_view(self):
        """Render the trading view section"""
        with st.expander("📈 Advanced Chart", expanded=True):
            # This would integrate with TradingView or another charting library
            st.write("TradingView chart would be embedded here")
            
            # Technical analysis indicators
            st.subheader("Technical Indicators")
            indicators = st.multiselect(
                "Add Indicators",
                ["RSI", "MACD", "Bollinger Bands", "Volume", "VWAP", "Ichimoku Cloud"],
                ["RSI", "MACD"]
            )
            
            # Pattern recognition
            st.subheader("Pattern Recognition")
            patterns = st.multiselect(
                "Detect Patterns",
                ["Head & Shoulders", "Double Top/Bottom", "Triangles", "Flags", "Wedges"],
                ["Head & Shoulders", "Double Top/Bottom"]
            )
    
    def _render_performance_metrics(self):
        """Render performance metrics"""
        with st.container():
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", "+24.5%", "2.1% today")
            with col2:
                st.metric("Win Rate", "68%", "Last 30 days")
            with col3:
                st.metric("Max Drawdown", "8.2%", "Current drawdown: 2.1%")
            with col4:
                st.metric("Sharpe Ratio", "1.85", "Risk-adjusted return")
            
            # Equity curve
            st.line_chart(pd.DataFrame({
                'Equity': [10000, 10200, 10500, 10300, 10800, 11200, 11000, 11500, 11800, 12450],
                'Benchmark': [10000, 10100, 10200, 10050, 10300, 10500, 10400, 10600, 10800, 11000]
            }))
    
    def _render_trading_signals(self):
        """Render trading signals and recommendations"""
        with st.container():
            st.subheader("📈 Trading Signals")
            
            # Current signals
            signals = [
                {"Symbol": "BTC/USDT", "Signal": "BUY", "Confidence": "85%", "Price": "43,200", "Target": "47,500", "Stop": "41,800"},
                {"Symbol": "ETH/USDT", "Signal": "HOLD", "Confidence": "72%", "Price": "2,320", "Target": "2,450", "Stop": "2,250"},
                {"Symbol": "SOL/USDT", "Signal": "SELL", "Confidence": "68%", "Price": "145.60", "Target": "130.00", "Stop": "152.00"},
            ]
            
            st.dataframe(
                pd.DataFrame(signals),
                use_container_width=True,
                hide_index=True
            )
    
    def _start_trading(self):
        """Start the trading bot"""
        try:
            # Update trader settings
            self.trader.position_size = self.position_size
            self.trader.risk_per_trade = self.risk_per_trade
            self.trader.take_profit = self.take_profit / 100
            self.trader.stop_loss = self.stop_loss / 100
            self.trader.timeframe = self.timeframe
            self.trader.strategy = self.strategy
            self.trader.min_confidence = self.ai_confidence / 100
            
            # Start trading
            self.trader.start()
            st.success("✅ Trading started successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to start trading: {str(e)}")
    
    def _stop_trading(self):
        """Stop the trading bot"""
        try:
            self.trader.stop()
            st.success("⏹️ Trading stopped successfully!")
        except Exception as e:
            st.error(f"❌ Failed to stop trading: {str(e)}")
    
    def _emergency_sell_all(self):
        """Emergency sell all positions"""
        if st.button("⚠️ CONFIRM EMERGENCY SELL", type="primary", 
                    help="This will immediately close all positions at market price"):
            try:
                self.trader.close_all_positions()
                st.success("✅ All positions closed successfully!")
            except Exception as e:
                st.error(f"❌ Failed to close positions: {str(e)}")
    
    def _pause_trading(self):
        """Pause trading"""
        try:
            self.trader.pause()
            st.success("⏸️ Trading paused")
        except Exception as e:
            st.error(f"❌ Failed to pause trading: {str(e)}")
    
    def _reset_strategy(self):
        """Reset strategy to default"""
        if st.button("🔄 CONFIRM RESET", type="secondary",
                    help="This will reset all strategy parameters to default"):
            try:
                self.trader.reset_strategy()
                st.success("🔄 Strategy reset to default settings")
            except Exception as e:
                st.error(f"❌ Failed to reset strategy: {str(e)}")
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Trading signals
        self._render_trading_signals()
    
    def _render_emergency_controls(self):
        """Render emergency controls at the bottom"""
        st.markdown("---")
        st.subheader("🚨 Emergency Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛑 Emergency Sell All", type="primary", 
                        help="Immediately close all positions at market price"):
                self._emergency_sell_all()
        
        with col2:
            if st.button("⏸️ Pause Trading", 
                        help="Pause all trading activity"):
                self._pause_trading()
        
        with col3:
            if st.button("🔄 Reset Strategy", 
                        help="Reset all strategy parameters to default"):
                self._reset_strategy()
    
    def _render_market_overview(self):
        """Render the market overview section"""
        with st.container():
            st.subheader("📊 Market Overview")
            
            # Market data
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BTC/USDT", "$45,321.50", "+2.3%")
            with col2:
                st.metric("ETH/USDT", "$2,345.67", "+1.8%")
            with col3:
                st.metric("24h Volume", "$45.2B", "+5.2%")
            with col4:
                st.metric("Market Sentiment", "Bullish", "3.2%")
            
            # Price chart
            st.line_chart(pd.DataFrame({
                'Price': [40000, 40500, 39800, 41000, 41500, 42000, 42500, 43000, 43500, 44000],
                'SMA 20': [None, None, None, 39800, 40200, 40800, 41200, 41800, 42200, 42800],
                'SMA 50': [None] * 5 + [40500, 40800, 41200, 41500, 41800, 42000]
            }))
    
    def _render_trading_view(self):
        """Render the trading view section"""
        with st.expander("📈 Advanced Chart", expanded=True):
            # This would integrate with TradingView or another charting library
            st.write("TradingView chart would be embedded here")
            
            # Technical analysis indicators
            st.subheader("Technical Indicators")
            indicators = st.multiselect(
                "Add Indicators",
                ["RSI", "MACD", "Bollinger Bands", "Volume", "VWAP", "Ichimoku Cloud"],
                ["RSI", "MACD"]
            )
            
            # Pattern recognition
            st.subheader("Pattern Recognition")
            patterns = st.multiselect(
                "Detect Patterns",
                ["Head & Shoulders", "Double Top/Bottom", "Triangles", "Flags", "Wedges"],
                ["Head & Shoulders", "Double Top/Bottom"]
            )
    
    def _render_performance_metrics(self):
        """Render performance metrics"""
        with st.container():
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", "+24.5%", "2.1% today")
            with col2:
                st.metric("Win Rate", "68%", "Last 30 days")
            with col3:
                st.metric("Max Drawdown", "8.2%", "Current drawdown: 2.1%")
            with col4:
                st.metric("Sharpe Ratio", "1.85", "Risk-adjusted return")
            
            # Equity curve
            st.line_chart(pd.DataFrame({
                'Equity': [10000, 10200, 10500, 10300, 10800, 11200, 11000, 11500, 11800, 12450],
                'Benchmark': [10000, 10100, 10200, 10050, 10300, 10500, 10400, 10600, 10800, 11000]
            }))
    
    def _render_trading_signals(self):
        """Render trading signals and recommendations"""
        with st.container():
            st.subheader("📈 Trading Signals")
            
            # Current signals
            signals = [
                {"Symbol": "BTC/USDT", "Signal": "BUY", "Confidence": "85%", "Price": "43,200", "Target": "47,500", "Stop": "41,800"},
                {"Symbol": "ETH/USDT", "Signal": "HOLD", "Confidence": "72%", "Price": "2,320", "Target": "2,450", "Stop": "2,250"},
                {"Symbol": "SOL/USDT", "Signal": "SELL", "Confidence": "68%", "Price": "145.60", "Target": "130.00", "Stop": "152.00"},
            ]
            
            st.dataframe(
                pd.DataFrame(signals),
                use_container_width=True,
                hide_index=True
            )
    
    def _start_trading(self):
        """Start the trading bot"""
        try:
            # Update trader settings
            self.trader.position_size = self.position_size
            self.trader.risk_per_trade = self.risk_per_trade
            self.trader.take_profit = self.take_profit / 100
            self.trader.stop_loss = self.stop_loss / 100
            self.trader.timeframe = self.timeframe
            self.trader.strategy = self.strategy
            self.trader.min_confidence = self.ai_confidence / 100
            
            # Start trading
            self.trader.start()
            st.success("✅ Trading started successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to start trading: {str(e)}")
    
    def _stop_trading(self):
        """Stop the trading bot"""
        try:
            self.trader.stop()
            st.success("⏹️ Trading stopped successfully!")
        except Exception as e:
            st.error(f"❌ Failed to stop trading: {str(e)}")
    
    def _emergency_sell_all(self):
        """Emergency sell all positions"""
        if st.button("⚠️ CONFIRM EMERGENCY SELL", type="primary", 
                    help="This will immediately close all positions at market price"):
            try:
                self.trader.close_all_positions()
                st.success("✅ All positions closed successfully!")
            except Exception as e:
                st.error(f"❌ Failed to close positions: {str(e)}")
    
    def _pause_trading(self):
        """Pause trading"""
        try:
            self.trader.pause()
            st.success("⏸️ Trading paused")
        except Exception as e:
            st.error(f"❌ Failed to pause trading: {str(e)}")
    
    def _reset_strategy(self):
        """Reset strategy to default"""
        if st.button("🔄 CONFIRM RESET", type="secondary",
                    help="This will reset all strategy parameters to default"):
            try:
                self.trader.reset_strategy()
                st.success("🔄 Strategy reset to default settings")
            except Exception as e:
                st.error(f"❌ Failed to reset strategy: {str(e)}")
    
    def _render_logs(self):
        """Render the trading logs"""
        log_container = st.empty()
        
        # In a real implementation, this would read from a log file or queue
        logs = [
            "[INFO] 2023-01-01 10:00:00 - Started trading session",
            "[INFO] 2023-01-01 10:00:01 - Connected to Binance API",
            "[TRADE] 2023-01-01 10:00:05 - BUY 0.1 BTC/USDT @ 40000.00",
            "[INFO] 2023-01-01 12:30:00 - Market conditions changed, adjusting strategy",
            "[TRADE] 2023-01-01 12:30:15 - SELL 1.5 ETH/USDT @ 3000.00",
        ]
        
        log_text = "\n\n".join(f"<div class='log-entry'>{log}</div>" for log in logs[-50:])  # Show last 50 logs
        log_container.markdown(log_text, unsafe_allow_html=True)

# Example usage
async def main():
    # Initialize the exchange
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # or 'spot' for spot trading
        },
    })
    
    # Initialize the trader
    trader = AutonomousTrader(exchange)
    
    # Initialize async components
    try:
        await trader.initialize_async()
        logger.info("Successfully initialized trader with LLM agent")
        return trader
    except Exception as e:
        logger.error(f"Failed to initialize trader: {e}")
        raise

    # Load some sample data for demonstration
    # In a real implementation, you would fetch this from the exchange
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, 1000),
        'high': np.random.normal(105, 10, 1000),
        'low': np.random.normal(95, 10, 1000),
        'close': np.random.normal(100, 10, 1000),
        'volume': np.random.normal(1000, 100, 1000)
    })
    
    # Process the data
    data = trader._process_market_data(data)
    
    # Train the model (in a real implementation, you would do this separately)
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            trader.train(data, total_timesteps=10000)
        st.success("Model trained successfully!")
    
    # Create and render the UI
    ui = AutonomousTradingUI(trader)
    ui.render()
