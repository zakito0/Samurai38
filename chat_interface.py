from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import asyncio
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from llm_analyzer import llm_analyzer

class TradingChatbot:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the TradingChatbot with OpenRouter integration.
        
        Args:
            api_key: Optional OpenRouter API key. If not provided, will use the one from environment variables.
        """
        self.logger = logging.getLogger(__name__)
        self.llm = llm_analyzer
        if api_key:
            self.llm.api_key = api_key
        self.logger.info("Trading chatbot initialized with OpenRouter LLM")

    async def get_response(self, user_input: str, context: Optional[Dict[str, Any]] = None,
                         image: Optional[Union[str, Image.Image, bytes]] = None) -> str:
        """Get a response from the chatbot, optionally using an image."""
        try:
            if image is not None:
                return await self.llm.analyze_image(image, user_input or "Analyze this trading chart.")
            return await self.llm.analyze_text(user_input, context)
        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    async def analyze_market_data(self, market_data: Dict[str, Any]) -> str:
        """
        Analyze market data and provide insights using the LLM.
        
        Args:
            market_data: Dictionary containing market data (OHLCV, indicators, etc.)
            
        Returns:
            Analysis of the market data
        """
        try:
            # Format the market data for the LLM
            context = {
                'symbol': market_data.get('symbol', 'Unknown'),
                'timeframe': market_data.get('timeframe', '1h'),
                'current_price': market_data.get('close', 'N/A'),
                'volume': market_data.get('volume', 'N/A'),
                'rsi': market_data.get('rsi', 'N/A') if 'rsi' in market_data else 'N/A',
                'macd': market_data.get('macd', 'N/A') if 'macd' in market_data else 'N/A',
                'support_levels': market_data.get('support_levels', 'N/A'),
                'resistance_levels': market_data.get('resistance_levels', 'N/A'),
                'indicators': market_data.get('indicators', {})
            }
            
            prompt = ("Analyze this market data and provide trading insights. "
                     "Focus on key levels, trends, and potential trading opportunities. "
                     "Include technical analysis, support/resistance levels, and potential entry/exit points. "
                     "Be concise but thorough in your analysis.")
            
            return await self.llm.analyze_text(prompt, context)
            
        except Exception as e:
            self.logger.error(f"Error in analyze_market_data: {str(e)}")
            return "I couldn't analyze the market data at this time. Please try again later."

    async def suggest_strategy(self, market_conditions: Optional[Dict[str, Any]] = None) -> str:
        """
        Suggest a trading strategy based on current market conditions.
        
        Args:
            market_conditions: Optional dictionary with market conditions
            
        Returns:
            Strategy suggestion with detailed reasoning
        """
        try:
            if not market_conditions:
                market_conditions = {
                    'trend': 'neutral',
                    'volatility': 'medium',
                    'volume': 'average',
                    'market_sentiment': 'neutral',
                    'timeframe': '1h',
                    'key_levels': {}
                }
            
            prompt = ("Based on the current market conditions, please suggest a detailed trading strategy. "
                     "Include the following in your response:\n"
                     "1. Overall market assessment\n"
                     "2. Recommended strategy (e.g., trend following, mean reversion, breakout, etc.)\n"
                     "3. Specific entry conditions\n"
                     "4. Stop loss levels with reasoning\n"
                     "5. Take profit targets with risk-reward ratio\n"
                     "6. Position sizing recommendations\n"
                     "7. Key levels to watch\n\n"
                     f"Market Conditions: {market_conditions}")
            
            return await self.llm.analyze_text(prompt)
            
        except Exception as e:
            self.logger.error(f"Error in suggest_strategy: {str(e)}")
            return "I couldn't generate a strategy at this time. Please try again later."

    def reset_chat(self) -> str:
        """Reset the chat history."""
        return self.llm.reset_chat()
        
    def set_api_key(self, api_key: str):
        """
        Update the OpenRouter API key.
        
        Args:
            api_key: The new OpenRouter API key
        """
        self.llm.api_key = api_key
        self.llm.headers["Authorization"] = f"Bearer {api_key}"
        self.logger.info("OpenRouter API key updated")

async def chat_loop(chatbot, trading_agent=None):
    """Interactive chat loop for the trading bot"""
    print("\n=== Trading Bot Chat Interface ===")
    print("Type 'exit' to end the chat")
    print("Type 'suggest' to get a strategy suggestion")
    print("Type 'analyze' to analyze current market")
    print("Type 'symbol X/Y' to analyze a specific pair (e.g., 'symbol BTC/USDT')")
    print("Type 'reset' to reset the conversation\n")
    
    current_symbol = 'BTC/USDT'  # Default trading pair
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print("Bot: Goodbye! Happy trading!")
                break
                
            elif user_input.lower() == 'reset':
                print("Bot:", chatbot.reset_chat())
                continue
                
            elif user_input.lower().startswith('symbol '):
                # Change the current trading pair
                new_symbol = user_input[7:].strip().upper()
                if '/' in new_symbol:  # Basic validation
                    current_symbol = new_symbol
                    print(f"Bot: Switched to {current_symbol}")
                else:
                    print("Bot: Please specify a valid trading pair (e.g., 'symbol BTC/USDT')")
                continue
                
            elif user_input.lower() == 'suggest':
                market_data = {}
                if trading_agent:
                    try:
                        # Get current market data if trading agent is available
                        ohlcv = await trading_agent.data_handler.get_ohlcv(current_symbol, limit=100)
                        if not ohlcv.empty:
                            market_data = {
                                'close': ohlcv['close'].iloc[-1],
                                'volume': ohlcv['volume'].iloc[-1],
                                'rsi': trading_agent.data_handler.calculate_rsi(ohlcv['close']).iloc[-1]
                            }
                        print(f"\nAnalyzing {current_symbol} market conditions...")
                    except Exception as e:
                        print(f"Couldn't fetch market data: {str(e)}")
                print("\nSuggested Strategy:")
                print(chatbot.suggest_strategy(market_data))
                continue
                
            elif user_input.lower() == 'analyze':
                if trading_agent and hasattr(trading_agent, 'analyze_market'):
                    try:
                        print(f"\nCurrent Market Analysis for {current_symbol}:")
                        analysis = await trading_agent.analyze_market(current_symbol)
                        print(analysis)
                    except Exception as e:
                        print(f"Error analyzing market: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("Market analysis not available")
                continue
                
            # Get response from the chatbot
            response = chatbot.get_response(user_input)
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nType 'exit' to quit the chat interface")
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
