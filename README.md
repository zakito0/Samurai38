# AI-Powered Cryptocurrency Trading Agent

An autonomous trading agent that uses machine learning and technical analysis to trade cryptocurrencies on Binance.

-## Features

- **Multiple Trading Strategies**: Implements various technical indicators including Moving Averages, RSI, and MACD
- **AI-Powered Analysis**: Uses Hugging Face's Transformers for market sentiment analysis
- **Django Frontend**: Modern assistant + autonomous trading dashboard served by Django
- **Risk Management**: Implements position sizing and stop-loss/take-profit mechanisms
- **Paper Trading**: Test strategies without risking real funds
- **Real-time Data**: Fetches and processes real-time market data from Binance

## Prerequisites

- Python 3.8+
- Binance API key and secret (for live trading)
- TA-Lib (Technical Analysis Library)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd NinjaZtrade
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install TA-Lib (required for technical indicators):
   - Windows: Download the appropriate wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - macOS: `brew install ta-lib`
   - Linux: `sudo apt-get install ta-lib`

## Configuration

1. Create a `.env` file in the project root with your Binance API credentials and Django secrets:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   OPENROUTER_API_KEY=optional_openrouter_key
   DJANGO_SECRET_KEY=generate_a_strong_value
   DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
   ```

2. (Optional) Edit `config.json` to customize trading parameters:
   ```json
   {
     "symbols": ["BTC/USDT", "ETH/USDT"],
     "timeframe": "1h",
     "max_position_size": 0.1,
     "stop_loss_pct": 0.05,
     "take_profit_pct": 0.1,
     "trading_enabled": false,
     "paper_balance": 10000.0,
     "risk_free_rate": 0.02,
     "max_drawdown_pct": 0.2,
     "data_refresh_interval": 300
   }
   ```

## Usage

### Running the Trading Bot

```bash
python main.py
```

### Monitoring

The bot logs all activities to `trading_bot.log`. You can monitor the bot's actions in real-time using:

```bash
tail -f trading_bot.log
```

### Web Interface (Django)

A modern UI for the assistant + autonomous trading controls now runs on Django:

```bash
python manage.py migrate  # first run
python manage.py runserver 0.0.0.0:8000
```

Visit `http://127.0.0.1:8000/` and interact with the chat assistant, upload charts, toggle strategies, and flip the autonomous trading status flag. The view reads the same `TradingChatbot` logic as the Streamlit build.

> Legacy Streamlit dashboards (`autonomous_app.py`) remain available for rapid prototyping and can still be launched with `streamlit run autonomous_app.py` if desired.

### Deployment

#### Local / Virtualenv
1. Install dependencies: `pip install -r requirements_autonomous.txt`
2. Put your Binance API keys in `.env`
3. Ensure `models/autonomous_trader_final.zip` exists (or train via `train_model.py`)
4. Start Streamlit: `streamlit run autonomous_app.py`

#### Docker

Build and run the containerized app (ideal for VPS/cloud hosting):

```bash
docker build -t ninjaztrade .
docker run -p 8501:8501 --env-file .env ninjaztrade
```

The container uses `requirements_autonomous.txt`, exposes port `8501`, and launches `autonomous_app.py`.

#### Streamlit Cloud (for demos)
1. Push this repo to GitHub (include `requirements_autonomous.txt` and `Dockerfile`)
2. In Streamlit Cloud, create a new app pointing to `autonomous_app.py`
3. Add `BINANCE_API_KEY`/`BINANCE_SECRET_KEY` as app secrets

> ⚠️ Streamlit Cloud sessions sleep on inactivity; for continuous live trading prefer Docker on a VPS.

### Configuration Notes
- The default quote asset is **USDC** and all symbols must end with `/USDC`
- `config.json` exposes `fixed_quote_order_amount` to cap per-trade spend
- Exchange filters (min notional / step size) are fetched automatically; if you add new symbols ensure Binance lists them for your quote asset

## Strategies

The bot comes with several built-in strategies:

1. **Moving Average Crossover**: Identifies trend changes using two moving averages
2. **RSI Strategy**: Uses Relative Strength Index to identify overbought/oversold conditions
3. **MACD Strategy**: Uses Moving Average Convergence Divergence for trend following
4. **Composite Strategy**: Combines multiple strategies with weighted voting

## Risk Management

- Maximum position size per trade: 10% of available balance (configurable)
- Stop loss: 5% (configurable)
- Take profit: 10% (configurable)
- Maximum drawdown: 20% (configurable)

## Backtesting

To backtest a strategy:

```bash
python backtest.py --strategy rsi --symbol BTC/USDT --timeframe 1h --start 2023-01-01 --end 2023-12-31
```

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred while using this software. Always test with paper trading before using real funds.
