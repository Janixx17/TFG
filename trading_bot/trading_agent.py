"""
Agent de Trading per a Integració amb IBKR
==========================================

Aquest mòdul implementa un agent de trading que utilitza el model CPPO-DeepSeek 
entrenat per prendre decisions de trading i executar-les a través d'Interactive 
Brokers (IBKR).

L'agent:
1. Es connecta a IBKR TWS/Gateway
2. Obté dades de mercat en temps real per als 84 principals valors
3. Calcula indicadors tècnics i característiques basades en LLM
4. Utilitza el model entrenat per predir accions de trading
5. Executa ordres de compra/venda a través d'IBKR
6. Monitoritza la cartera i les mètriques de risc

Estructura d'entrada del model:
- Espai d'observació: 1009 dimensions
- Espai d'accions: 84 dimensions (una per valor)
- Composició d'estat: efectiu + preus + posicions + indicadors + característiques LLM

Basat en el notebook de backtesting FinRL-DeepSeek i test_agent_no_spinup.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium.spaces import Box
from torch.distributions.normal import Normal
from ib_insync import *
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import requests
import json
import os
from functools import lru_cache
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    MIN_REQUEST_INTERVAL, CACHE_DURATION, MAX_TOKENS, TEMPERATURE,
    SENTIMENT_PROMPT_TEMPLATE, RISK_PROMPT_TEMPLATE,
    TRADING_CONFIG, ENABLE_LLM_FEATURES
)
warnings.filterwarnings('ignore')

# Configuració del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepSeekAPIClient:
    """
    Client de l'API DeepSeek per a l'anàlisi de sentiment i avaluació de risc
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        # Utilitza els valors de configuració com a predeterminats
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Limitació de velocitat des de la configuració
        self.last_request_time = 0
        self.min_request_interval = MIN_REQUEST_INTERVAL
        
        # Memòria cau per a les respostes de l'API per evitar crides repetides
        self.sentiment_cache = {}
        self.risk_cache = {}
        self.cache_duration = CACHE_DURATION
        
    def _make_request(self, prompt: str, max_tokens: int = None) -> str:
        """Fa una petició a l'API DeepSeek"""
        try:
            # Limitació de velocitat
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            # Utilitza els valors de configuració
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens or MAX_TOKENS,
                "temperature": TEMPERATURE
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",  # TODO: Utilitzar l'endpoint correcte
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                # TODO: Ajustar segons l'estructura real de la resposta de l'API
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"Error de l'API DeepSeek: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error en fer la petició a l'API DeepSeek: {e}")
            return ""
    
    def _parse_sentiment_response(self, response: str) -> float:
        """Analitza la resposta de sentiment i retorna una puntuació normalitzada"""
        try:
            # TODO: Implementar l'anàlisi adequada basada en l'estructura del prompt
            # Aquesta és una implementació provisional
            response_lower = response.lower()
            
            # Anàlisi simple basada en paraules clau (substituir per anàlisi adequada)
            if "very positive" in response_lower or "bullish" in response_lower:
                return 0.8
            elif "positive" in response_lower:
                return 0.4
            elif "negative" in response_lower:
                return -0.4
            elif "very negative" in response_lower or "bearish" in response_lower:
                return -0.8
            else:
                return 0.0  # Neutral
                
        except Exception as e:
            logger.error(f"Error en analitzar la resposta de sentiment: {e}")
            return 0.0
    
    def _parse_risk_response(self, response: str) -> float:
        """Analitza la resposta de risc i retorna una puntuació normalitzada"""
        try:
            # TODO: Implementar l'anàlisi adequada basada en l'estructura del prompt
            # Aquesta és una implementació provisional
            response_lower = response.lower()
            
            # Anàlisi simple basada en paraules clau (substituir per anàlisi adequada)
            if "very high risk" in response_lower or "high volatility" in response_lower:
                return 0.8
            elif "high risk" in response_lower:
                return 0.6
            elif "medium risk" in response_lower or "moderate risk" in response_lower:
                return 0.4
            elif "low risk" in response_lower:
                return 0.2
            else:
                return 0.0  # Risc molt baix
                
        except Exception as e:
            logger.error(f"Error en analitzar la resposta de risc: {e}")
            return 0.0
    
    @lru_cache(maxsize=100)
    def get_sentiment_analysis(self, symbol: str, current_price: float, price_change: float) -> float:
        """Obté l'anàlisi de sentiment per a un valor"""
        try:
            # Comprova primer la memòria cau
            cache_key = f"{symbol}_{int(time.time() // self.cache_duration)}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Utilitza la plantilla de prompt de configuració
            prompt = SENTIMENT_PROMPT_TEMPLATE.format(
                symbol=symbol,
                price=current_price,
                price_change=price_change
            )
            
            response = self._make_request(prompt)
            sentiment_score = self._parse_sentiment_response(response)
            
            # Guarda el resultat a la memòria cau
            self.sentiment_cache[cache_key] = sentiment_score
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error en obtenir el sentiment per {symbol}: {e}")
            return 0.0
    
    @lru_cache(maxsize=100)
    def get_risk_assessment(self, symbol: str, volatility: float, volume_ratio: float) -> float:
        """Obté l'avaluació de risc per a un valor"""
        try:
            # Comprova primer la memòria cau
            cache_key = f"{symbol}_risk_{int(time.time() // self.cache_duration)}"
            if cache_key in self.risk_cache:
                return self.risk_cache[cache_key]
            
            # Utilitza la plantilla de prompt de configuració
            prompt = RISK_PROMPT_TEMPLATE.format(
                symbol=symbol,
                volatility=volatility,
                volume_ratio=volume_ratio
            )
            
            response = self._make_request(prompt)
            risk_score = self._parse_risk_response(response)
            
            # Guarda el resultat a la memòria cau
            self.risk_cache[cache_key] = risk_score
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error en obtenir l'avaluació de risc per {symbol}: {e}")
            return 0.0

class MLPActor(nn.Module):
    """
    Xarxa neuronal Actor MLP simple per a espais d'acció continus
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        
        # Construeix les capes de la xarxa per coincidir amb l'estructura del model guardat
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:  # No afegir activació després de l'última capa
                layers.append(activation())
        
        self.mu_net = nn.Sequential(*layers)  # Utilitza mu_net per coincidir amb el model guardat
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32))

    def _distribution(self, obs):
        mu = self.mu_net(obs)  # Utilitza mu_net en lloc de pi
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCritic(nn.Module):
    """
    Simple MLP Critic network for value function approximation
    """
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        
        # Build the network layers
        sizes = [obs_dim] + list(hidden_sizes) + [1]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:  # Don't add activation after last layer
                layers.append(activation())
        
        self.v_net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    """
    Combined Actor-Critic network
    """
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        else:
            raise NotImplementedError("Only continuous action spaces supported")

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class TechnicalIndicators:
    """
    Technical indicators calculator matching FinRL implementation
    """
    
    @staticmethod
    def calculate_macd(close_prices, window_slow=26, window_fast=12, window_signal=9):
        """Calculate MACD"""
        exp1 = close_prices.ewm(span=window_fast).mean()
        exp2 = close_prices.ewm(span=window_slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=window_signal).mean()
        return macd.iloc[-1] if len(macd) > 0 else 0.0
    
    @staticmethod
    def calculate_rsi(close_prices, window=14):
        """Calculate RSI"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50.0
    
    @staticmethod
    def calculate_bollinger_bands(close_prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = close_prices.rolling(window=window).mean()
        rolling_std = close_prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        current_price = close_prices.iloc[-1]
        current_upper = upper_band.iloc[-1] if len(upper_band) > 0 else current_price * 1.02
        current_lower = lower_band.iloc[-1] if len(lower_band) > 0 else current_price * 0.98
        
        # Return normalized position within bands
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        return max(0.0, min(1.0, bb_position))
    
    @staticmethod
    def calculate_cci(high_prices, low_prices, close_prices, window=14):
        """Calculate Commodity Channel Index"""
        tp = (high_prices + low_prices + close_prices) / 3
        sma = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci.iloc[-1] if len(cci) > 0 else 0.0
    
    @staticmethod
    def calculate_adx(high_prices, low_prices, close_prices, window=14):
        """Calculate Average Directional Index"""
        high_diff = high_prices.diff()
        low_diff = -low_prices.diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = np.maximum(high_prices - low_prices, 
                       np.maximum(np.abs(high_prices - close_prices.shift(1)),
                                 np.abs(low_prices - close_prices.shift(1))))
        
        atr = tr.rolling(window=window).mean()
        pos_di = (pos_dm.rolling(window=window).mean() / atr) * 100
        neg_di = (neg_dm.rolling(window=window).mean() / atr) * 100
        
        dx = (np.abs(pos_di - neg_di) / (pos_di + neg_di)) * 100
        adx = dx.rolling(window=window).mean()
        
        return adx.iloc[-1] if len(adx) > 0 else 0.0

class TradingAgent:
    """
    Main trading agent that integrates with IBKR
    """
    
    def __init__(self, 
                 model_path: str = None,
                 stocks_file: str = None,
                 ibkr_host: str = None,
                 ibkr_port: int = None,
                 client_id: int = None,
                 initial_cash: float = None,
                 max_position_size: float = None,
                 trading_frequency: int = None,
                 enable_llm_features: bool = None,
                 deepseek_api_key: str = None):
        
        # Use config values as defaults
        self.model_path = model_path or TRADING_CONFIG["model_path"]
        self.stocks_file = stocks_file or TRADING_CONFIG["stocks_file"]
        self.ibkr_host = ibkr_host or TRADING_CONFIG["ibkr_host"]
        self.ibkr_port = ibkr_port or TRADING_CONFIG["ibkr_port"]
        self.client_id = client_id or TRADING_CONFIG["client_id"]
        self.initial_cash = initial_cash or TRADING_CONFIG["initial_cash"]
        self.max_position_size = max_position_size or TRADING_CONFIG["max_position_size"]
        self.trading_frequency = trading_frequency or TRADING_CONFIG["trading_frequency"]
        self.enable_llm_features = enable_llm_features if enable_llm_features is not None else TRADING_CONFIG["enable_llm_features"]
        deepseek_api_key = deepseek_api_key or TRADING_CONFIG["deepseek_api_key"]
        
        # Model parameters (from analysis)
        self.obs_dim = 1009
        self.act_dim = 84
        
        # Initialize components
        self.model = None
        self.ib = None
        self.stocks_list = []
        self.current_prices = {}
        self.current_holdings = {}
        self.portfolio_value = initial_cash
        self.cash_balance = initial_cash
        
        # Technical indicators
        self.tech_indicators = TechnicalIndicators()
        
        # DeepSeek API client
        if self.enable_llm_features:
            self.deepseek_client = DeepSeekAPIClient(api_key=deepseek_api_key)
            logger.info("DeepSeek API client initialized")
        else:
            self.deepseek_client = None
            logger.info("LLM features disabled - using placeholder values")
        
        # Initialize
        self._load_stocks()
        self._load_model()
        
    def _load_stocks(self):
        """Load the list of stocks to trade"""
        try:
            stocks_df = pd.read_csv(self.stocks_file)
            self.stocks_list = stocks_df['Symbol'].tolist()[:84]  # Top 84 stocks
            logger.info(f"Loaded {len(self.stocks_list)} stocks for trading")
        except Exception as e:
            logger.error(f"Error loading stocks: {e}")
            raise
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Create model architecture
            observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            action_space = Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)
            
            self.model = MLPActorCritic(
                observation_space=observation_space,
                action_space=action_space,
                hidden_sizes=(512, 512),
                activation=torch.nn.ReLU
            )
            
            # Load trained weights
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def connect_to_ibkr(self):
        """Connect to IBKR TWS/Gateway"""
        try:
            self.ib = IB()
            self.ib.connect(self.ibkr_host, self.ibkr_port, clientId=self.client_id)
            logger.info("Connected to IBKR successfully")
            
            # Initialize portfolio tracking
            self._update_portfolio_status()
            
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            raise
    
    def disconnect_from_ibkr(self):
        """Disconnect from IBKR"""
        if self.ib:
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")
    
    def _update_portfolio_status(self):
        """Update current portfolio status"""
        try:
            # Get account summary
            account_summary = self.ib.accountSummary()
            for item in account_summary:
                if item.tag == 'TotalCashValue':
                    self.cash_balance = float(item.value)
                elif item.tag == 'NetLiquidation':
                    self.portfolio_value = float(item.value)
            
            # Get current positions
            positions = self.ib.positions()
            self.current_holdings = {}
            
            for position in positions:
                symbol = position.contract.symbol
                if symbol in self.stocks_list:
                    self.current_holdings[symbol] = position.position
            
            logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}, Cash: ${self.cash_balance:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def _get_market_data(self, lookback_days: int = 30) -> Dict:
        """Get market data for all stocks"""
        try:
            market_data = {}
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get data for all stocks
            for symbol in self.stocks_list:
                try:
                    # Get historical data from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) > 0:
                        market_data[symbol] = {
                            'price': hist['Close'].iloc[-1],
                            'high': hist['High'],
                            'low': hist['Low'],
                            'close': hist['Close'],
                            'volume': hist['Volume'].iloc[-1]
                        }
                        
                        # Update current prices
                        self.current_prices[symbol] = market_data[symbol]['price']
                    else:
                        logger.warning(f"No data found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    continue
            
            logger.info(f"Retrieved market data for {len(market_data)} stocks")
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def _calculate_technical_indicators(self, market_data: Dict) -> Dict:
        """Calculate technical indicators for all stocks"""
        indicators = {}
        
        for symbol, data in market_data.items():
            try:
                close_prices = data['close']
                high_prices = data['high']
                low_prices = data['low']
                
                # Calculate indicators (matching FinRL)
                indicators[symbol] = {
                    'macd': self.tech_indicators.calculate_macd(close_prices),
                    'rsi': self.tech_indicators.calculate_rsi(close_prices),
                    'cci': self.tech_indicators.calculate_cci(high_prices, low_prices, close_prices),
                    'adx': self.tech_indicators.calculate_adx(high_prices, low_prices, close_prices),
                    'bb_position': self.tech_indicators.calculate_bollinger_bands(close_prices),
                    'sma_20': close_prices.rolling(20).mean().iloc[-1] if len(close_prices) >= 20 else close_prices.iloc[-1],
                    'sma_50': close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else close_prices.iloc[-1],
                    'volatility': close_prices.pct_change().rolling(20).std().iloc[-1] if len(close_prices) >= 20 else 0.02,
                    'momentum': (close_prices.iloc[-1] / close_prices.iloc[-10] - 1) if len(close_prices) >= 10 else 0.0,
                    'volume_ratio': data['volume'] / close_prices.rolling(20).mean().iloc[-1] if len(close_prices) >= 20 else 1.0
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
                indicators[symbol] = self._get_default_indicators()
        
        return indicators
    
    def _get_default_indicators(self) -> Dict:
        """Get default indicator values"""
        return {
            'macd': 0.0,
            'rsi': 50.0,
            'cci': 0.0,
            'adx': 0.0,
            'bb_position': 0.5,
            'sma_20': 0.0,
            'sma_50': 0.0,
            'volatility': 0.02,
            'momentum': 0.0,
            'volume_ratio': 1.0
        }
    
    def _construct_observation(self, market_data: Dict, indicators: Dict, 
                             sentiment_scores: Dict, risk_scores: Dict) -> np.ndarray:
        """
        Construct observation vector for the model
        Structure: [cash_balance, prices, holdings, tech_indicators, llm_sentiment, llm_risk]
        Total: 1009 dimensions
        """
        try:
            observation = []
            
            # 1. Cash balance (normalized)
            normalized_cash = self.cash_balance / self.initial_cash
            observation.append(normalized_cash)
            
            # 2. Stock prices (84 dimensions)
            for symbol in self.stocks_list:
                if symbol in market_data:
                    # Normalize price (you may want to adjust this)
                    price = market_data[symbol]['price'] / 100.0  # Simple normalization
                    observation.append(price)
                else:
                    observation.append(0.0)
            
            # 3. Stock holdings (84 dimensions)
            for symbol in self.stocks_list:
                holdings = self.current_holdings.get(symbol, 0)
                # Normalize holdings
                normalized_holdings = holdings / 1000.0  # Adjust based on typical position sizes
                observation.append(normalized_holdings)
            
            # 4. Technical indicators (10 indicators × 84 stocks = 840 dimensions)
            indicator_names = ['macd', 'rsi', 'cci', 'adx', 'bb_position', 
                             'sma_20', 'sma_50', 'volatility', 'momentum', 'volume_ratio']
            
            for indicator_name in indicator_names:
                for symbol in self.stocks_list:
                    if symbol in indicators:
                        value = indicators[symbol].get(indicator_name, 0.0)
                        # Normalize indicators
                        if indicator_name == 'rsi':
                            value = value / 100.0
                        elif indicator_name in ['macd', 'cci', 'adx', 'momentum']:
                            value = np.tanh(value / 10.0)  # Squash to [-1, 1]
                        elif indicator_name in ['volatility', 'volume_ratio']:
                            value = np.tanh(value)
                        elif indicator_name in ['sma_20', 'sma_50']:
                            value = value / 100.0
                        observation.append(value)
                    else:
                        observation.append(0.0)
            
            # 5. LLM sentiment (84 dimensions) - now using actual DeepSeek API
            for symbol in self.stocks_list:
                sentiment = sentiment_scores.get(symbol, 0.0)
                observation.append(sentiment)
            
            # 6. LLM risk (84 dimensions) - now using actual DeepSeek API
            for symbol in self.stocks_list:
                risk = risk_scores.get(symbol, 0.0)
                observation.append(risk)
            
            # Ensure exact dimensions
            if len(observation) != self.obs_dim:
                logger.warning(f"Observation dimension mismatch: {len(observation)} vs {self.obs_dim}")
                # Pad or truncate as needed
                if len(observation) < self.obs_dim:
                    observation.extend([0.0] * (self.obs_dim - len(observation)))
                else:
                    observation = observation[:self.obs_dim]
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error constructing observation: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    def _predict_actions(self, observation: np.ndarray) -> np.ndarray:
        """Predict trading actions using the model"""
        try:
            obs_tensor = torch.tensor(observation).unsqueeze(0)
            
            with torch.no_grad():
                actions = self.model.act(obs_tensor)
            
            return actions.squeeze()
            
        except Exception as e:
            logger.error(f"Error predicting actions: {e}")
            return np.zeros(self.act_dim)
    
    def _execute_trades(self, actions: np.ndarray, market_data: Dict):
        """Execute trades based on model predictions"""
        try:
            executed_trades = 0
            
            for i, symbol in enumerate(self.stocks_list):
                if symbol not in market_data:
                    continue
                
                action = actions[i]
                current_price = market_data[symbol]['price']
                current_position = self.current_holdings.get(symbol, 0)
                
                # Calculate target position based on action
                # Action is in [-1, 1], map to position size
                max_position_value = self.portfolio_value * self.max_position_size
                target_shares = int((action * max_position_value) / current_price)
                
                # Calculate trade size
                trade_size = target_shares - current_position
                
                # Execute trade if significant
                if abs(trade_size) > 0:  # Minimum trade size
                    try:
                        contract = Stock(symbol, 'SMART', 'USD')
                        self.ib.qualifyContracts(contract)
                        
                        if trade_size > 0:
                            # Buy order
                            order = MarketOrder('BUY', trade_size)
                            logger.info(f"Placing BUY order: {trade_size} shares of {symbol}")
                        else:
                            # Sell order
                            order = MarketOrder('SELL', abs(trade_size))
                            logger.info(f"Placing SELL order: {abs(trade_size)} shares of {symbol}")
                        
                        # Place order
                        trade = self.ib.placeOrder(contract, order)
                        executed_trades += 1
                        
                        # Wait for order confirmation
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error executing trade for {symbol}: {e}")
                        continue
            
            logger.info(f"Executed {executed_trades} trades")
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    def run_trading_loop(self):
        """Main trading loop"""
        try:
            logger.info("Starting trading loop...")
            
            while True:
                start_time = time.time()
                
                # 1. Update portfolio status
                self._update_portfolio_status()
                
                # 2. Get market data
                market_data = self._get_market_data()
                
                if not market_data:
                    logger.warning("No market data available, skipping this iteration")
                    time.sleep(self.trading_frequency)
                    continue
                
                # 3. Calculate technical indicators
                indicators = self._calculate_technical_indicators(market_data)
                
                # 4. Get LLM sentiment and risk analysis
                sentiment_scores, risk_scores = self._get_llm_sentiment_and_risk(market_data, indicators)
                
                # 5. Construct observation
                observation = self._construct_observation(market_data, indicators, sentiment_scores, risk_scores)
                
                # 6. Predict actions
                actions = self._predict_actions(observation)
                
                # 7. Execute trades
                self._execute_trades(actions, market_data)
                
                # 8. Log performance
                elapsed_time = time.time() - start_time
                logger.info(f"Trading iteration completed in {elapsed_time:.2f} seconds")
                logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
                
                # Wait for next iteration
                time.sleep(max(0, self.trading_frequency - elapsed_time))
                
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.disconnect_from_ibkr()

    def _get_llm_sentiment_and_risk(self, market_data: Dict, indicators: Dict) -> Tuple[Dict, Dict]:
        """
        Get LLM-based sentiment and risk analysis for all stocks
        """
        sentiment_scores = {}
        risk_scores = {}
        
        if not self.enable_llm_features or not self.deepseek_client:
            # Return placeholder values if LLM features are disabled
            for symbol in self.stocks_list:
                sentiment_scores[symbol] = 0.0
                risk_scores[symbol] = 0.0
            return sentiment_scores, risk_scores
        
        try:
            for symbol in self.stocks_list:
                if symbol not in market_data:
                    sentiment_scores[symbol] = 0.0
                    risk_scores[symbol] = 0.0
                    continue
                
                # Get current price and calculate price change
                current_price = market_data[symbol]['price']
                
                # Calculate price change percentage (if we have enough data)
                close_prices = market_data[symbol]['close']
                if len(close_prices) >= 2:
                    price_change = ((current_price - close_prices.iloc[-2]) / close_prices.iloc[-2]) * 100
                else:
                    price_change = 0.0
                
                # Get sentiment analysis
                try:
                    sentiment = self.deepseek_client.get_sentiment_analysis(
                        symbol=symbol,
                        current_price=current_price,
                        price_change=price_change
                    )
                    sentiment_scores[symbol] = sentiment
                except Exception as e:
                    logger.error(f"Error getting sentiment for {symbol}: {e}")
                    sentiment_scores[symbol] = 0.0
                
                # Get risk assessment
                try:
                    volatility = indicators.get(symbol, {}).get('volatility', 0.02)
                    volume_ratio = indicators.get(symbol, {}).get('volume_ratio', 1.0)
                    
                    risk = self.deepseek_client.get_risk_assessment(
                        symbol=symbol,
                        volatility=volatility,
                        volume_ratio=volume_ratio
                    )
                    risk_scores[symbol] = risk
                except Exception as e:
                    logger.error(f"Error getting risk assessment for {symbol}: {e}")
                    risk_scores[symbol] = 0.0
            
            logger.info(f"Retrieved LLM sentiment and risk for {len(sentiment_scores)} stocks")
            
        except Exception as e:
            logger.error(f"Error getting LLM sentiment and risk: {e}")
            # Fallback to placeholder values
            for symbol in self.stocks_list:
                sentiment_scores[symbol] = 0.0
                risk_scores[symbol] = 0.0
        
        return sentiment_scores, risk_scores

def main():
    """Main function to run the trading agent"""
    try:
        # Initialize trading agent using configuration
        agent = TradingAgent()
        
        # Connect to IBKR
        agent.connect_to_ibkr()
        
        # Run trading loop
        agent.run_trading_loop()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

