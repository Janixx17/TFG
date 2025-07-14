# TFG
https://www.kaggle.com/datasets/everydaycodings/global-news-dataset
https://newsdata.io/datasets
https://www.kaggle.com/datasets/rmisra/news-category-dataset
https://github.com/Webhose/free-news-datasets

# IA's
## FINRL [https://github.com/AI4Finance-Foundation/FinRL](https://github.com/benstaf/FinRL_DeepSeek)

Estructura principal fitxers:<br>
├── data/<br>
│   ├── utils_data_download.py  # per descarregar dades històriques<br>
├── notebooks/<br>
│   ├── 01_train_agent.ipynb  # entrenament amb FinRL<br>
│   ├── 02_backtest_agent.ipynb  # backtesting<br>
├── models/<br>
│   ├── (s'hi desarà l'agent entrenat)<br>
├── trading_bot/<br>
│   ├── __init__.py<br>
│   ├── trading_env.py  # entorn personalitzat (basat en StockTradingEnv de FinRL)<br>
│   ├── rl_agent.py  # per carregar el model i fer prediccions<br>
│   ├── trader_ibkr.py  # integració amb IBKR utilitzant ib_insync<br>
│   ├── logger.py  # per al registre de les operacions (logging)<br>
├── requirements.txt<br>
├── README.md<br>
└── main.py  # script que executa el bucle de trading en viu amb IBKR<br>


"""
Input Structure for agent_cppo_deepseek_100_epochs_20k_steps_01.pth
The neural network model receives an observation vector of size 1009 that contains the following information:

Observation Space Breakdown:
The observation space is calculated as:

state_space_llm_risk = 1 + 2 * stock_dimension + (2 + len(INDICATORS)) * stock_dimension

Where:

stock_dimension = 84 (number of unique stocks, top 100 NASDAQ stocks)
INDICATORS = technical indicators from FinRL (typically around 9-10 indicators)
Final calculation: 1 + 2×84 + (2+10)×84 = 1 + 168 + 1008 = 1177
But the actual model uses 1009 dimensions, which corresponds to the CPPO-DeepSeek risk environment with 0.1% weight.

Input Components:
Cash balance (1 dimension)
Stock prices (84 dimensions - current prices of 84 stocks)
Stock holdings (84 dimensions - number of shares held for each stock)
Technical indicators (840 dimensions - 10 technical indicators × 84 stocks)
LLM sentiment scores (84 dimensions - DeepSeek sentiment for each stock)
LLM risk scores (84 dimensions - DeepSeek risk assessment for each stock)
Technical Indicators (from FinRL):

The model uses standard technical indicators such as:

Moving averages (MA)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Bollinger Bands
Commodity Channel Index (CCI)
And others from the INDICATORS list
LLM Enhancement:
This specific model (_01) incorporates:

DeepSeek sentiment analysis with 0.1% weight
DeepSeek risk assessment with 0.1% weight
Both are integrated into the observation space for each of the 84 stocks
The model architecture is a Multi-Layer Perceptron (MLP) with:

Input layer: 1009 dimensions
Hidden layers: 512 neurons each (2 hidden layers)
Output layer: 84 dimensions (actions for each stock)
Activation: ReLU

This creates a comprehensive trading environment where the agent can make informed decisions based on traditional financial indicators enhanced with modern LLM-based sentiment and risk analysis.

"""
