import os
import json
import logging
import aiohttp
from typing import Optional, Dict, Any, List, Union
from PIL import Image
import base64
from io import BytesIO

import ccxt

class LLMAnalyzer:
    def __init__(self, 
                api_key: Optional[str] = None,
                model: str = "openai/gpt-oss-20b:free"):
        """
        Initialize the LLM analyzer with OpenRouter API.
        
        Args:
            api_key: Your OpenRouter API key. If not provided, will try to get from OPENROUTER_API_KEY environment variable.
            model: The model to use for text generation. Default is gpt-4-turbo-preview.
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://ninjaz-trading-bot.com",
            "X-Title": "NinjaZ Trading Bot"
        }
        
        # Chat history
        self.chat_history = []
        self.max_history = 10
        
        # System prompt with emphasis on Binance data
        self.system_prompt = """You are NinjaZ, an AI trading assistant specialized in Binance cryptocurrency trading. 
        Your primary focus is analyzing Binance market data to provide actionable trading insights.
        
        When analyzing market data:
        1. Always prioritize Binance order book depth, recent trades, and candlestick patterns
        2. Focus on key support/resistance levels, volume profiles, and order flow
        3. Consider Binance-specific indicators like funding rates and open interest
        4. Look for arbitrage opportunities between Binance and other exchanges
        5. Pay special attention to Binance launchpad/launchpool announcements
        
        Be concise, professional, and data-driven in your responses.
        Always provide clear reasoning and specific price levels for your analysis."""
        
        if not self.api_key:
            self.logger.warning("No OpenRouter API key provided. Set OPENROUTER_API_KEY environment variable.")

    def _fetch_binance_market_snapshot(self, symbol: str = "BTC/USDT", timeframe: str = "1h") -> Optional[Dict[str, Any]]:
        """Return current ticker + simple indicator data from Binance."""
        try:
            exchange = ccxt.binance({"enableRateLimit": True})
            ticker = exchange.fetch_ticker(symbol)

            ohlcv = []
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=50)
            except Exception as sub_err:  # pragma: no cover - best-effort data
                self.logger.debug(f"Unable to fetch {symbol} {timeframe} OHLCV: {sub_err}")

            closes = [candle[4] for candle in ohlcv[-10:]] if ohlcv else []
            latest_close = closes[-1] if closes else ticker.get("last")
            prev_close = closes[-2] if len(closes) > 1 else ticker.get("open")
            avg_close = sum(closes) / len(closes) if closes else ticker.get("last")

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "price": ticker.get("last"),
                "high": ticker.get("high"),
                "low": ticker.get("low"),
                "volume": ticker.get("baseVolume"),
                "quoteVolume": ticker.get("quoteVolume"),
                "change": ticker.get("percentage"),
                "latest_close": latest_close,
                "previous_close": prev_close,
                "avg_close": avg_close,
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask")
            }
        except Exception as err:
            self.logger.warning(f"Couldn't fetch Binance data for {symbol}: {err}")
            return None

    def _format_market_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """Convert snapshot dict into a concise textual context."""
        if not snapshot:
            return ""

        return (
            f"Symbol: {snapshot['symbol']} ({snapshot['timeframe']})\n"
            f"Last Price: ${snapshot['price']:,.2f}\n"
            f"24h High/Low: ${snapshot['high']:,.2f} / ${snapshot['low']:,.2f}\n"
            f"24h Change: {snapshot['change'] or 0:.2f}%\n"
            f"Bid/Ask: ${snapshot['bid'] or 0:,.2f} / ${snapshot['ask'] or 0:,.2f}\n"
            f"Volume (base/quote): {snapshot['volume'] or 0:,.2f} / ${snapshot['quoteVolume'] or 0:,.2f}\n"
            f"Latest Close vs Avg(10): ${snapshot['latest_close'] or 0:,.2f} / ${snapshot['avg_close'] or 0:,.2f}"
        )

    async def _make_openrouter_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the OpenRouter API."""
        if not self.api_key:
            return {"error": "OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."}
            
        url = f"{self.base_url}/chat/completions"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
                    return await response.json()
        except Exception as e:
            self.logger.error(f"Error making request to OpenRouter: {str(e)}")
            return {"error": f"Failed to connect to OpenRouter: {str(e)}"}

    def _image_to_base64(self, image: Union[str, Image.Image, bytes]) -> str:
        """Convert an image to base64 string."""
        if isinstance(image, str):
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        else:
            raise ValueError("Unsupported image format")

    async def analyze_text(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response to the given text prompt using OpenRouter, with emphasis on Binance data.
        
        Args:
            prompt: User's input prompt
            context: Optional context dictionary with additional information
            
        Returns:
            Generated response with Binance-focused analysis
        """
        """Generate a response to the given text prompt using OpenRouter."""
        try:
            context = context or {}
            symbol = context.get('symbol', 'BTC/USDT')
            timeframe = context.get('timeframe', '1h')

            market_snapshot = self._fetch_binance_market_snapshot(symbol, timeframe)

            # Prepare messages with Binance context
            binance_context = {
                'exchange': 'Binance',
                'market_type': context.get('market_type', 'spot'),
                'preferred_pairs': context.get('preferred_pairs', ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
                'timeframes': context.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
            }
            
            # Add Binance-specific context to the system message
            binance_system_msg = f"{self.system_prompt}\n\nCurrent Binance Context:\n"
            binance_system_msg += "\n".join(f"{k}: {v}" for k, v in binance_context.items())
            
            if market_snapshot:
                binance_system_msg += "\n\nLive Market Snapshot:\n"
                binance_system_msg += self._format_market_snapshot(market_snapshot)
            else:
                binance_system_msg += "\n\nLive Market Snapshot: unavailable (Binance fetch failed)."
            
            messages = [{"role": "system", "content": binance_system_msg}]
            
            # Add context if provided
            if context:
                context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
                messages.append({"role": "system", "content": f"Context: {context_str}"})
            
            # Add chat history
            for msg in self.chat_history[-self.max_history:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # Make the API request
            response = await self._make_openrouter_request(payload)
            
            if "error" in response:
                return f"Error: {response['error']}"
                
            # Extract the response text
            response_text = response["choices"][0]["message"]["content"]
            
            # Update chat history
            self._update_chat_history(prompt, response_text)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error in analyze_text: {str(e)}")
            return "I encountered an error processing your request. Please try again."

    async def analyze_image(self, image: Union[str, Image.Image, bytes], prompt: str = None) -> str:
        """Analyze a trading chart image using OpenRouter's vision capabilities with Binance data priority."""
        try:
            # Convert image to base64
            if isinstance(image, str):
                with open(image, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            elif isinstance(image, Image.Image):
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            elif isinstance(image, bytes):
                image_b64 = base64.b64encode(image).decode('utf-8')
            else:
                return "Unsupported image format"

            # Get current market data from Binance
            market_snapshot = self._fetch_binance_market_snapshot()
            if market_snapshot:
                formatted_snapshot = self._format_market_snapshot(market_snapshot)
                market_context = (
                    "IMPORTANT: Use this real-time Binance data for your analysis:\n"
                    f"{formatted_snapshot}\n\n"
                    "Base your analysis PRIMARILY on this data. The chart is for reference only.\n"
                )
            else:
                market_context = ""

            # Prepare the prompt
            if not prompt:
                prompt = f"""{market_context}Analyze this trading chart, but IGNORE any price/volume data in the image.
Use ONLY the Binance data provided above for your analysis.

Focus on:
1. Current trend based on the Binance data
2. Key levels from the chart that align with the current price
3. Trading opportunities based on the real market data
4. Risk assessment using the actual market data

Remember: The chart is just for pattern recognition - use the Binance data for all price/volume information.
"""

            # Prepare the request
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"}
                ]
            }]

            payload = {
                "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.3
            }

            # Make the API request
            response = await self._make_openrouter_request(payload)
            
            if "error" in response:
                return f"Error analyzing image: {response['error']}"

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            self.logger.error(f"Error in analyze_image: {str(e)}", exc_info=True)
            return f"Error processing image: {str(e)}"
    
    def _update_chat_history(self, user_input: str, assistant_response: str):
        """Update the chat history with the latest exchange."""
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": assistant_response})
        
        # Trim history if it gets too long
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history*2:]
    
    def reset_chat(self):
        """Reset the chat history."""
        self.chat_history = []
        return "Chat history has been reset."

# Singleton instance with API key
llm_analyzer = LLMAnalyzer(api_key="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")
