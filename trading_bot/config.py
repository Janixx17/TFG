# DeepSeek API Configuration
# =========================
# TODO: Fill in your actual DeepSeek API configuration

# DeepSeek API Settings
DEEPSEEK_API_KEY = "sk-77ed50b04745406c9f6a9cb8827dc9dc"  # TODO: Replace with your actual API key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"   # TODO: Verify the correct base URL
DEEPSEEK_MODEL = "deepseek-chat"                 # TODO: Use the correct model name

# API Rate Limiting
MIN_REQUEST_INTERVAL = 1.0  # Minimum seconds between requests
CACHE_DURATION = 300        # Cache responses for 5 minutes

# LLM Feature Configuration
ENABLE_LLM_FEATURES = True  # TODO: Set to True when API is configured
MAX_TOKENS = 100            # Maximum tokens for API responses
TEMPERATURE = 0.1           # Low temperature for consistent responses

# Sentiment Analysis Prompts

# Enhanced Sentiment Analysis Prompt with News
SENTIMENT_PROMPT_TEMPLATE = """
Analyze the sentiment for stock {symbol} based on the following comprehensive information:

Technical Data:
- Current Price: ${price:.2f}
- Price Change: {price_change:.2f}%

Recent News Analysis, check before if the news is relevant:
{news_text}

Instructions:
1. Analyze the news sentiment (positive, negative, or neutral)
2. Consider the price movement in context of the news
3. Evaluate potential impact on future price movement
4. Provide overall sentiment rating from one of the following: very positive/positive/neutral/negative/very negative

Focus on actionable insights for trading decisions. Consider both short-term market reaction and longer-term fundamental impact.

Respond with: [SENTIMENT: very positive/positive/neutral/negative/very negative]
"""

# Risk Assessment Prompts
# TODO: Customize these prompts based on your requirements
RISK_PROMPT_TEMPLATE = """
Assess the risk level for stock {symbol} with the following metrics:
- Volatility: {volatility:.4f}
- Volume ratio: {volume_ratio:.2f}

Please provide a risk assessment considering market volatility, 
trading volume, and current market conditions.

Respond with: [RISK: very high/high/medium/low/very low]
"""

# Trading Agent Configuration
TRADING_CONFIG = {
    "model_path": "../models/agent_cppo_deepseek_100_epochs_20k_steps_01.pth",
    "stocks_file": "../stock_csv/top100.csv",
    "ibkr_host": "127.0.0.1",
    "ibkr_port": 7497,
    "client_id": 1,
    "initial_cash": 1000000.0,
    "max_position_size": 0.01,    # 1% max position size
    "trading_frequency": 300,     # 5 minutes
    "enable_llm_features": ENABLE_LLM_FEATURES,
    "deepseek_api_key": DEEPSEEK_API_KEY
}

# TODO: Instructions for setup:
# 1. Get your DeepSeek API key from https://platform.deepseek.com/
# 2. Replace DEEPSEEK_API_KEY with your actual key
# 3. Verify the correct API base URL and model name
# 4. Test the API connection before enabling LLM features
# 5. Customize the prompts for your specific use case
# 6. Set ENABLE_LLM_FEATURES to True when ready
