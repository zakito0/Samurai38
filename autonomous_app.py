import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os
import json
import logging
from autonomous_trader import AutonomousTrader, TradingAction, TradeSignal
import ccxt
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousTradingApp:
    def __init__(self):
        self.trader = None
        self.initialized = False
        self.setup_page()
        self.initialize_trader()
        
    def setup_page(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="🤖 Autonomous Trading System",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .main-header { 
                font-size: 2.5rem; 
                font-weight: bold; 
                color: #1f77b4;
                margin-bottom: 1rem;
            }
            .status-card { 
                border-radius: 10px; 
                padding: 1.5rem; 
                background-color: #f8f9fa;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 5px solid #1f77b4;
            }
            .metric-card { 
                border-radius: 10px; 
                padding: 1rem; 
                background-color: #f8f9fa; 
                margin: 0.5rem 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .positive { 
                color: #2ecc71; 
                font-weight: bold;
            }
            .negative { 
                color: #e74c3c; 
                font-weight: bold;
            }
            .log-entry { 
                padding: 0.5rem; 
                margin: 0.25rem 0; 
                border-radius: 5px; 
                background-color: #f1f3f5;
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
            }
            .strategy-active {
                border-left: 5px solid #2ecc71;
            }
            .strategy-inactive {
                border-left: 5px solid #95a5a6;
            }
        </style>
        """, unsafe_allow_html=True)
        
    def initialize_trader(self):
        """Initialize the autonomous trader and load the trained model"""
        if not self.initialized:
            try:
                load_dotenv()
                exchange = ccxt.binance({
                    'apiKey': os.getenv('BINANCE_API_KEY', ''),
                    'secret': os.getenv('BINANCE_SECRET_KEY', ''),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True,
                        'warnOnFetchCurrenciesWithoutPermission': False,
                        'fetchCurrencies': False,
                        'warnOnFetchOpenOrdersWithoutSymbol': False,
                    },
                    'urls': {
                        'api': {
                            'public': 'https://api.binance.com/api/v3',
                            'private': 'https://api.binance.com/api/v3',
                            'sapi': 'https://api.binance.com/sapi/v1'
                        }
                    }
                })
                
                # Initialize the trader
                self.trader = AutonomousTrader(exchange)
                
                # Try to load the trained model
                model_loaded = self.trader.load_model("models/autonomous_trader_final")
                if not model_loaded:
                    st.warning("No trained model found. Please train a model first.")
                    return False
                
                self.initialized = True
                logger.info("Autonomous trader initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize trader: {e}")
                st.error(f"Failed to initialize trader: {e}")
                return False
    
    def render_sidebar(self):
        """Render the sidebar controls"""
        with st.sidebar:
            st.title("⚙️ Control Panel")

            # AI Agent Status
            st.subheader("🤖 AI Agent")
            ai_status = "🟢 Active" if hasattr(self, 'trader') and hasattr(self.trader, 'llm_agent') and self.trader.llm_agent is not None else "🔴 Inactive"
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #2e3b4e; color: white; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>AI Agent Status:</span>
                    <span style="font-weight: bold;">{ai_status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Trading control
            st.subheader("🎮 Trading Control")
            
            # Timeframe selection
            st.selectbox(
                "⏱️ AI Agent Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
                key="ai_timeframe",
                index=2,  # Default to 15m
                help="Select the timeframe for AI agent analysis and actions"
            )
            
            # Price per trade and risk-reward
            col1, col2 = st.columns(2)
            with col1:
                price_per_trade = st.number_input(
                    "💰 Price per Trade (USDC)",
                    min_value=10.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0,
                    help="Amount of USDC to allocate per trade"
                )
            
            with col2:
                risk_reward = st.number_input(
                    "🎯 Risk/Reward Ratio",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Desired risk/reward ratio for trades"
                )
            
            # Trading actions
            col1, col2 = st.columns(2)
            trading_active = hasattr(self.trader, 'is_running') and self.trader.is_running
            
            with col1:
                if st.button("▶️ Start Trading" if not trading_active else "⏸️ Pause Trading", 
                           type="primary" if not trading_active else "secondary",
                           use_container_width=True):
                    try:
                        if not trading_active:
                            # Update trader settings
                            self.trader.config['price_per_trade'] = price_per_trade
                            self.trader.config['risk_reward_ratio'] = risk_reward
                            self.trader.config['timeframe'] = st.session_state.get('ai_timeframe', '15m')
                            self.trader.start_trading()
                            st.rerun()
                        else:
                            self.trader.stop_trading()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col2:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.rerun()
            
            # Emergency sell button
            if st.button("🆘 SELL ALL POSITIONS (USDC)", 
                        type="primary",
                        use_container_width=True,
                        help="Immediately liquidate all positions to USDC"):
                try:
                    if hasattr(self.trader, 'emergency_sell_all'):
                        self.trader.emergency_sell_all()
                        st.success("Emergency sell order placed for all positions")
                        st.rerun()
                    else:
                        st.error("Emergency sell functionality not available")
                except Exception as e:
                    st.error(f"Error during emergency sell: {e}")
            
            # Account summary
            st.subheader("💰 Account Summary")
            available_usdc = 0.0
            total_balance = 0.0
            pnl = 0.0
            
            if self.trader is not None:
                try:
                    self.trader._sync_account_state()
                    available_usdc = getattr(self.trader, "cash", 0.0)
                    portfolio_value = getattr(self.trader, "portfolio_value", 0)
                    # Use portfolio_value directly instead of adding to available_usdc to avoid double-counting
                    total_balance = portfolio_value if hasattr(self.trader, "portfolio_value") else available_usdc
                    pnl = getattr(self.trader, "total_pnl", 0)
                except Exception as e:
                    logger.warning(f"Unable to refresh balances: {e}")
            
            st.metric("Total Balance", f"${total_balance:,.2f}", delta=f"${pnl:+,.2f} P&L")
            
            # Performance metrics
            st.metric("Available USDC", f"${available_usdc:,.2f}")
            if hasattr(self.trader, 'portfolio_value'):
                st.metric("Invested Value", f"${(total_balance - available_usdc):,.2f}")
            
            # Current positions
            if hasattr(self.trader, 'positions') and self.trader.positions:
                st.subheader("📊 Open Positions")
                for symbol, position_info in self.trader.positions.items():
                    if isinstance(position_info, dict) and 'amount' in position_info:
                        amount = position_info['amount']
                        value = position_info.get('value', 0)
                        price = position_info.get('price', 0)
                        
                        st.metric(
                            f"{symbol} Position",
                            f"{amount:.4f} @ ${price:.4f}",
                            delta=f"${value:,.2f} Value"
                        )
                    elif position_info != 0:  # Fallback for backward compatibility
                        st.metric(
                            f"{symbol} Position",
                            f"{position_info:.4f}",
                            delta=f"${getattr(self.trader, 'unrealized_pnl', {}).get(symbol, 0):+,.2f}"
                        )
            
            # Status indicators
            st.subheader("🔍 System Status")
            self._render_status_indicators()
    
    def _render_status_indicators(self):
        """Render status indicators in the sidebar"""
        if not hasattr(self, 'trader') or self.trader is None:
            st.warning("Trader not initialized")
            return
            
        # Status cards with icons
        status_cards = [
            {
                "title": "Exchange",
                "status": "Connected" if self.initialized else "Disconnected",
                "icon": "🔌",
                "color": "#2ecc71" if self.initialized else "#e74c3c"
            },
            {
                "title": "Trading",
                "status": "Active" if hasattr(self.trader, 'is_running') and self.trader.is_running else "Stopped",
                "icon": "📈" if hasattr(self.trader, 'is_running') and self.trader.is_running else "⏸️",
                "color": "#2ecc71" if hasattr(self.trader, 'is_running') and self.trader.is_running else "#e74c3c"
            },
            {
                "title": "AI Agent",
                "status": "Ready" if hasattr(self.trader, 'llm_agent') and self.trader.llm_agent is not None else "Disabled",
                "icon": "🤖",
                "color": "#3498db" if hasattr(self.trader, 'llm_agent') and self.trader.llm_agent is not None else "#95a5a6"
            },
            {
                "title": "Last Signal",
                "status": self.trader.last_signal_time.strftime('%H:%M:%S') if hasattr(self.trader, 'last_signal_time') else "N/A",
                "icon": "📡",
                "color": "#9b59b6"
            }
        ]
        
        # Render status cards
        for card in status_cards:
            st.markdown(f"""
            <div style="
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                background-color: #2d3748;
                color: white;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-left: 4px solid {card['color']};
            ">
                <div>
                    <div style="font-size: 0.9rem; color: #a0aec0;">{card['title']}</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">{card['status']}</div>
                </div>
                <div style="font-size: 1.5rem;">{card['icon']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Last update and refresh
        st.markdown(f"""
        <div style="
            margin-top: 20px; 
            padding: 10px;
            background-color: #2d3748;
            border-radius: 8px;
            color: #a0aec0;
            font-size: 0.8rem;
            text-align: center;
        ">
            <div>Last updated: {datetime.now().strftime('%H:%M:%S')}</div>
            <div style="margin-top: 5px;">🔄 Auto-refresh: 30s</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_metrics(self):
        """Render the main metrics and charts"""
        if not hasattr(self, 'trader') or self.trader is None:
            st.warning("Trader not initialized. Please check your API keys and connection.")
            return
            
        # Get trading stats
        stats = self.trader.get_trading_stats()
        
        # Main metrics row
        st.markdown("## 📊 Trading Dashboard")
        
        # Portfolio overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", 
                     f"${stats.get('portfolio_value', 0):,.2f}",
                     delta=f"${stats.get('daily_pnl', 0):+,.2f} today")
            
        with col2:
            return_pct = stats.get('total_return_pct', 0)
            st.metric("Total Return", 
                     f"{return_pct:+.2f}%",
                     delta=f"{stats.get('daily_return_pct', 0):+.2f}% today")
            
        with col3:
            win_rate = stats.get('win_rate', 0)
            st.metric("Win Rate", 
                     f"{win_rate:.1f}%",
                     delta=f"{stats.get('win_rate_change', 0):+.1f}%")
            
        with col4:
            sharpe = stats.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", 
                     f"{sharpe:.2f}",
                     delta="Higher is better" if sharpe > 1.5 else "Needs improvement")
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trades_count = stats.get('total_trades', 0)
            st.metric("Total Trades", 
                     f"{trades_count:,}",
                     delta=f"{stats.get('today_trades', 0)} today")
            
        with col2:
            avg_trade = stats.get('avg_trade_return', 0)
            st.metric("Avg. Trade Return", 
                     f"{avg_trade:+.2f}%",
                     delta=f"${stats.get('avg_trade_pnl', 0):+,.2f}")
            
        with col3:
            max_drawdown = stats.get('max_drawdown_pct', 0)
            st.metric("Max Drawdown", 
                     f"{max_drawdown:.2f}%",
                     delta=f"${stats.get('max_drawdown_amt', 0):,.2f}")
            
        with col4:
            st.metric("Volatility", 
                     f"{stats.get('volatility', 0):.2f}%",
                     delta="30d")
        
        # Performance charts
        st.markdown("## 📈 Performance Charts")
        self._render_performance_charts()
        
        # Trading activity
        tab1, tab2, tab3 = st.tabs(["💼 Active Positions", "📊 Trade History", "📋 Trading Logs"])
        
        with tab1:
            self._render_positions()
            
        with tab2:
            self._render_trade_history()
            
        with tab3:
            self._render_logs()
    
    def _render_trade_history(self):
        """Render the trade history table"""
        if not self.trader or not hasattr(self.trader, 'trades') or not self.trader.trades:
            st.info("No trade history available yet.")
            return
            
        # Prepare trade history data
        trades = []
        for i, trade in enumerate(self.trader.trades):
            trade_data = {
                'ID': i + 1,
                'Date': trade.get('timestamp', 'N/A'),
                'Symbol': trade.get('symbol', 'N/A'),
                'Type': trade.get('side', 'N/A'),
                'Price': trade.get('price', 0),
                'Quantity': trade.get('amount', 0),
                'Cost': trade.get('cost', 0),
                'Fee': f"{trade.get('fee', {}).get('cost', 0)} {trade.get('fee', {}).get('currency', '')}",
                'PnL': trade.get('pnl', 0),
                'PnL %': trade.get('pnl_pct', 0),
                'Status': trade.get('status', 'N/A')
            }
            trades.append(trade_data)
            
        if not trades:
            st.info("No trades have been executed yet.")
            return
            
        # Create a DataFrame and display it
        df = pd.DataFrame(trades)
        
        # Format numbers for better display
        df['Price'] = df['Price'].apply(lambda x: f"${x:,.4f}" if isinstance(x, (int, float)) else x)
        df['Cost'] = df['Cost'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
        df['PnL'] = df['PnL'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
        df['PnL %'] = df['PnL %'].apply(
            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
        )
        
        # Display the table with better formatting
        st.dataframe(
            df,
            column_config={
                'ID': 'ID',
                'Date': 'Date',
                'Symbol': 'Symbol',
                'Type': st.column_config.TextColumn(
                    'Type',
                    help="Trade type (Buy/Sell)"
                ),
                'Price': 'Price',
                'Quantity': 'Qty',
                'Cost': 'Cost',
                'Fee': 'Fee',
                'PnL': st.column_config.TextColumn(
                    'PnL',
                    help="Profit and Loss"
                ),
                'PnL %': st.column_config.TextColumn(
                    'PnL %',
                    help="Profit and Loss Percentage"
                ),
                'Status': 'Status'
            },
            use_container_width=True,
            hide_index=True
        )

    def _render_performance_charts(self):
        """Render performance charts"""
        # Create tabs for different chart views
        tab1, tab2, tab3 = st.tabs(["📊 Portfolio Value", "📈 Returns", "📉 Drawdown"])
        
        with tab1:
            # Portfolio value over time
            if hasattr(self.trader, 'portfolio_history') and len(self.trader.portfolio_history) > 1:
                df = pd.DataFrame({
                    'Date': list(self.trader.portfolio_history.keys()),
                    'Portfolio Value': list(self.trader.portfolio_history.values())
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Portfolio Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#4e79a7', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(78, 121, 167, 0.1)'
                ))
                
                # Add buy/sell markers if available
                if hasattr(self.trader, 'trade_history'):
                    buys = [t for t in self.trader.trade_history if t['type'] == 'BUY']
                    sells = [t for t in self.trader.trade_history if t['type'] == 'SELL']
                    
                    if buys:
                        buy_dates = [t['timestamp'] for t in buys]
                        buy_values = [self.trader.portfolio_history.get(d, 0) for d in buy_dates]
                        fig.add_trace(go.Scatter(
                            x=buy_dates,
                            y=buy_values,
                            mode='markers',
                            name='Buy',
                            marker=dict(color='#59a14f', size=10, symbol='triangle-up')
                        ))
                    
                    if sells:
                        sell_dates = [t['timestamp'] for t in sells]
                        sell_values = [self.trader.portfolio_history.get(d, 0) for d in sell_dates]
                        fig.add_trace(go.Scatter(
                            x=sell_dates,
                            y=sell_values,
                            mode='markers',
                            name='Sell',
                            marker=dict(color='#e15759', size=10, symbol='triangle-down')
                        ))
                
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value (USDT)',
                    template='plotly_dark',
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No portfolio history data available")
        
        with tab2:
            # Daily returns
            if hasattr(self.trader, 'daily_returns') and len(self.trader.daily_returns) > 1:
                df = pd.DataFrame({
                    'Date': list(self.trader.daily_returns.keys()),
                    'Return': list(self.trader.daily_returns.values())
                })
                
                # Calculate cumulative returns
                df['Cumulative Return'] = (1 + df['Return']).cumprod() - 1
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Daily returns (bar chart)
                fig.add_trace(
                    go.Bar(
                        x=df['Date'],
                        y=df['Return'] * 100,
                        name='Daily Return',
                        marker_color=['#59a14f' if x >= 0 else '#e15759' for x in df['Return']],
                        opacity=0.7
                    ),
                    secondary_y=False
                )
                
                # Cumulative return (line)
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Cumulative Return'] * 100,
                        name='Cumulative Return',
                        line=dict(color='#4e79a7', width=2)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title='Daily Returns',
                    xaxis_title='Date',
                    yaxis_title='Daily Return (%)',
                    yaxis2_title='Cumulative Return (%)',
                    template='plotly_dark',
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No return data available")
        
        with tab3:
            # Drawdown chart
            if hasattr(self.trader, 'drawdown') and len(self.trader.drawdown) > 1:
                df = pd.DataFrame({
                    'Date': list(self.trader.drawdown.keys()),
                    'Drawdown': list(self.trader.drawdown.values())
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#e15759', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(225, 87, 89, 0.2)'
                ))
                
                fig.update_layout(
                    title='Portfolio Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    template='plotly_dark',
                    hovermode='x',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No drawdown data available")
        
        # Sample data for demonstration
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        portfolio_values = np.cumprod(1 + np.random.normal(0.001, 0.02, 30)) * 10000
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_values, name="Portfolio Value", line=dict(color='#3498db')),
            secondary_y=False,
        )
        
        # Add shape for drawdowns
        peak = portfolio_values[0]
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i] > peak:
                peak = portfolio_values[i]
            else:
                fig.add_shape(
                    type="rect",
                    x0=dates[i-1], y0=peak,
                    x1=dates[i], y1=portfolio_values[i],
                    fillcolor="rgba(231, 76, 60, 0.2)",
                    line=dict(width=0),
                    layer="below"
                )
        
        # Add layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, title="Portfolio Value (USD)"),
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", "1.25", "+0.05")
        with col2:
            st.metric("Max Drawdown", "8.2%", "-0.3%")
        with col3:
            st.metric("Win Rate", "62%", "+2%")
        with col4:
            st.metric("Profit Factor", "1.8", "+0.1")
    
    def _render_positions(self):
        """Render the current positions table"""
        # Sample data - in a real implementation, this would come from the trader
        positions = [
            {"Symbol": "BTC/USDT", "Size": 0.5, "Entry Price": 40000, "Current Price": 41250, "P&L ($)": 625, "P&L (%)": 3.12, "Leverage": 1.0},
            {"Symbol": "ETH/USDT", "Size": 5.2, "Entry Price": 2800, "Current Price": 2750, "P&L ($)": -260, "P&L (%)": -1.79, "Leverage": 1.0},
        ]
        
        if positions:
            df = pd.DataFrame(positions)
            
            # Format the DataFrame for display
            df_display = df.copy()
            df_display["P&L ($)"] = df_display["P&L ($)"].apply(lambda x: f"${x:,.2f}")
            df_display["P&L (%)"] = df_display["P&L (%)"].apply(lambda x: f"{x:+.2f}%")
            df_display["Entry Price"] = df_display["Entry Price"].apply(lambda x: f"${x:,.2f}")
            df_display["Current Price"] = df_display["Current Price"].apply(lambda x: f"${x:,.2f}")
            
            # Display the table with conditional formatting
            st.dataframe(
                df_display,
                column_config={
                    "Symbol": "Symbol",
                    "Size": "Size",
                    "Entry Price": "Entry Price",
                    "Current Price": "Current Price",
                    "P&L ($)": "P&L ($)",
                    "P&L (%)": "P&L (%)",
                    "Leverage": "Leverage"
                },
                hide_index=True,
                width='stretch'
            )
        else:
            st.info("No open positions.")
    
    def _render_logs(self):
        """Render the trading logs"""
        # In a real implementation, this would read from a log file or queue
        logs = [
            {"timestamp": "2023-11-16 10:00:00", "level": "INFO", "message": "Started trading session"},
            {"timestamp": "2023-11-16 10:00:01", "level": "INFO", "message": "Connected to Binance API"},
            {"timestamp": "2023-11-16 10:00:05", "level": "TRADE", "message": "BUY 0.1 BTC/USDT @ 40000.00"},
            {"timestamp": "2023-11-16 10:30:22", "level": "INFO", "message": "Market conditions changed, adjusting strategy"},
            {"timestamp": "2023-11-16 11:15:45", "level": "TRADE", "message": "SELL 1.5 ETH/USDT @ 3000.00"},
            {"timestamp": "2023-11-16 12:00:10", "level": "WARNING", "message": "High volatility detected"},
            {"timestamp": "2023-11-16 12:30:00", "level": "INFO", "message": "Taking profits on BTC position"},
            {"timestamp": "2023-11-16 13:00:00", "level": "INFO", "message": "Rebalancing portfolio"},
        ]
        
        log_container = st.container()
        
        for log in logs[-50:]:  # Show last 50 logs
            # Determine color based on log level
            if log["level"] == "INFO":
                color = "#3498db"
            elif log["level"] == "TRADE":
                color = "#2ecc71"
            elif log["level"] == "WARNING":
                color = "#f39c12"
            else:
                color = "#7f8c8d"
                
            log_container.markdown(
                f"""
                <div class="log-entry">
                    <span style="color: {color}; font-weight: bold;">[{log['level']}]</span>
                    <span style="color: #7f8c8d; font-size: 0.85rem;">{log['timestamp']}</span>
                    <span style="margin-left: 10px;">{log['message']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def run(self):
        """Run the Streamlit app"""
        st.markdown("<div class='main-header'>🤖 Autonomous Trading System</div>", unsafe_allow_html=True)
        
        # Initialize trader if not already done
        if not self.initialized:
            self.initialize_trader()
        
        # Render the UI
        self.render_sidebar()
        self.render_metrics()

if __name__ == "__main__":
    app = AutonomousTradingApp()
    app.run()
