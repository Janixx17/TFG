#!/usr/bin/env python3
"""
Test script for DeepSeek API integration
========================================

This script tests the DeepSeek API client to verify it's working correctly
before running the full trading agent.

TODO: Configure your API key in config.py before running this test.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_agent import DeepSeekAPIClient
from config import DEEPSEEK_API_KEY, ENABLE_LLM_FEATURES
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deepseek_api():
    """Test the DeepSeek API client"""
    
    if not ENABLE_LLM_FEATURES:
        logger.warning("LLM features are disabled in config.py")
        logger.info("To enable: Set ENABLE_LLM_FEATURES = True in config.py")
        return False
    
    if DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
        logger.error("DeepSeek API key not configured!")
        logger.info("TODO: Set your actual API key in config.py")
        return False
    
    try:
        # Initialize the client
        client = DeepSeekAPIClient(api_key=DEEPSEEK_API_KEY)
        logger.info("DeepSeek API client initialized")
        
        # Test sentiment analysis
        logger.info("Testing sentiment analysis...")
        test_symbol = "AAPL"
        test_price = 150.0
        test_change = 2.5
        
        sentiment = client.get_sentiment_analysis(
            symbol=test_symbol,
            current_price=test_price,
            price_change=test_change
        )
        
        logger.info(f"Sentiment for {test_symbol}: {sentiment}")
        
        # Test risk assessment
        logger.info("Testing risk assessment...")
        test_volatility = 0.02
        test_volume_ratio = 1.5
        
        risk = client.get_risk_assessment(
            symbol=test_symbol,
            volatility=test_volatility,
            volume_ratio=test_volume_ratio
        )
        
        logger.info(f"Risk assessment for {test_symbol}: {risk}")
        
        logger.info("[SUCCESS] DeepSeek API test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] DeepSeek API test failed: {e}")
        logger.info("TODO: Check your API key and network connection")
        return False

def main():
    """Main test function"""
    logger.info("=== DeepSeek API Test ===")
    
    success = test_deepseek_api()
    
    if success:
        logger.info("[SUCCESS] All tests passed! You can now enable LLM features in the trading agent.")
    else:
        logger.error("[ERROR] Tests failed. Please check the TODO items above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
