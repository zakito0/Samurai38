import streamlit as st
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import existing modules
from chat_interface import TradingChatbot
from main import TradingAgent

# Page config
st.set_page_config(
    page_title="NinjaZ Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for black text and better UI
st.markdown("""
    <style>
    /* Main text color */
    .stApp, .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
        color: #000000 !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        color: #000000 !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton>button {
        color: white !important;
        border-color: #4F8BF9 !important;
        background-color: #4F8BF9 !important;
    }
    
    /* Strategy cards */
    .strategy-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    
    .strategy-card.active {
        border-left: 4px solid #4F8BF9;
        background-color: #f0f7ff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #4F8BF9 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Basic styles
st.markdown(
    """
    <style>
    .main { padding: 1rem; }
    .chat-message { padding: 0.75rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .user-message { background: #f0f2f6; margin-left: 20%; border-left: 4px solid #4CAF50; }
    .bot-message { background: #e3f2fd; margin-right: 20%; border-left: 4px solid #2196F3; }
    .strategy-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #fafafa; }
    .strategy-card.active { border-left: 4px solid #4CAF50; background: #f1f8f4; }
    .tab-content { padding-top: 0.5rem; }
    .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helpers

def run_async(coro):
    """Run an async coroutine from Streamlit sync context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If an event loop is already running (unlikely in Streamlit), create a new task
        loop = asyncio.get_event_loop()
        return loop.create_task(coro)

# Session bootstrap
if 'agent' not in st.session_state:
    st.session_state.agent = TradingAgent('config.json')
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = TradingChatbot()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, timestamp}
if 'active_strategies' not in st.session_state:
    st.session_state.active_strategies = set()
if 'trading_enabled' not in st.session_state:
    st.session_state.trading_enabled = False
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'BTC/USDT'

agent: TradingAgent = st.session_state.agent
chatbot: TradingChatbot = st.session_state.chatbot

# UI Components

def render_message(role: str, content: str):
    css = 'user-message' if role == 'user' else 'bot-message'
    st.markdown(
        f'<div class="chat-message {css}"><strong>{"You" if role=="user" else "Assistant"}:</strong><br>{content}</div>',
        unsafe_allow_html=True,
    )


def format_market_analysis(analysis: Dict[str, Any], symbol: str) -> str:
    if not analysis:
        return f"Could not analyze {symbol}."
    parts = []
    price = analysis.get('price')
    if price is not None:
        parts.append(f"Price: ${price:,.2f}")
    change = analysis.get('change_24h')
    if change is not None:
        parts.append(f"24h Change: {change}%")
    vol = analysis.get('volume_24h')
    if vol is not None:
        parts.append(f"24h Volume: ${vol:,.2f}")
    action = analysis.get('action', 'HOLD')
    conf = analysis.get('confidence', 0)
    lines = [f"📊 {symbol} Analysis", "", *parts, "", f"Recommendation: {action} (Confidence: {conf*100:.1f}%)"]
    return "\n".join(lines)


def render_chat_tab():
    st.subheader("💬 Trading Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image" in message:
                st.image(message["image"], width='stretch')
                if message["text"]:
                    st.write(message["text"])
            else:
                st.write(message["content"])
    
    # Image uploader
    uploaded_file = st.file_uploader(
        "Upload a chart to analyze",
        type=['png', 'jpg', 'jpeg'],
        key="file_uploader"
    )
    
    # Handle image upload
    if uploaded_file is not None and uploaded_file != st.session_state.get("last_uploaded_file"):
        st.session_state.last_uploaded_file = uploaded_file
        st.session_state.chat_history.append({
            "role": "user",
            "content": "Analyzing the uploaded chart...",
            "image": uploaded_file,
            "text": ""
        })
        st.rerun()
    
    # Text input
    if prompt := st.chat_input("Ask about the chart or request analysis..."):
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "text": prompt
        })
        
        # Get the last uploaded image if it exists
        image_data = None
        if st.session_state.get("last_uploaded_file"):
            image_data = st.session_state.last_uploaded_file.getvalue()
        
        # Get response
        with st.spinner("Analyzing..."):
            response = asyncio.run(st.session_state.chatbot.get_response(
                prompt,
                image=image_data
            ))
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "text": response
            })
            st.session_state.last_uploaded_file = None  # Reset after processing
            st.rerun()


def render_strategy_card(sid: str, name: str, desc: str, risk: str, timeframe: str, pairs):
    active = sid in st.session_state.active_strategies
    with st.container():
        st.markdown(f"<div class='strategy-card {'active' if active else ''}'>", unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"#### {name}")
            st.caption(desc)
            st.text(f"Risk: {risk} | Timeframe: {timeframe} | Pairs: {', '.join(pairs)}")
        with c2:
            toggled = st.toggle("Active", value=active, key=f"toggle_{sid}")
            if toggled:
                st.session_state.active_strategies.add(sid)
            else:
                st.session_state.active_strategies.discard(sid)
        st.markdown("</div>", unsafe_allow_html=True)


def render_autonomous_tab():
    st.subheader("🤖 Autonomous Trading")

    st.markdown("### 🎯 Strategies")
    render_strategy_card(
        'trend_following', 'Trend Following',
        'Follows market trends using moving averages', 'Medium', '1h', ['BTC/USDT', 'ETH/USDT']
    )
    render_strategy_card(
        'mean_reversion', 'Mean Reversion',
        'Capitalizes on price returning to the mean', 'High', '15m', ['BTC/USDT']
    )
    render_strategy_card(
        'breakout', 'Breakout',
        'Trades breakouts from key levels', 'High', '4h', ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    )

    st.markdown("### 🎮 Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Start Trading", type="primary"):
            st.session_state.trading_enabled = True
            st.success("Autonomous trading started (UI flag).")
    with c2:
        if st.button("🛑 Stop Trading"):
            st.session_state.trading_enabled = False
            st.warning("Autonomous trading stopped (UI flag).")

    status = "ACTIVE" if st.session_state.trading_enabled else "INACTIVE"
    st.markdown(f"**Status:** {'✅' if status=='ACTIVE' else '⏸️'} {status}")

    st.markdown("### 📊 Stats")
    c = st.columns(5)
    c[0].metric("Total Trades", "-")
    c[1].metric("Win Rate", "-")
    c[2].metric("Total P&L", "-")
    c[3].metric("Daily P&L", "-")
    c[4].metric("Active Positions", "-")


# Main layout
st.title("🤖 NinjaZ Trading Bot")

# Tabs
chat_tab, auto_tab = st.tabs(["💬 Assistant", "🤖 Autonomous Trading"])
with chat_tab:
    render_chat_tab()
with auto_tab:
    render_autonomous_tab()
