# TFG
https://www.kaggle.com/datasets/everydaycodings/global-news-dataset
https://newsdata.io/datasets
https://www.kaggle.com/datasets/rmisra/news-category-dataset
https://github.com/Webhose/free-news-datasets

# IA's
## FINRL https://github.com/AI4Finance-Foundation/FinRL

my-trading-bot/
├── data/
│   ├── utils_data_download.py  # per descarregar dades històriques
├── notebooks/
│   ├── 01_train_agent.ipynb  # entrenament amb FinRL
│   ├── 02_backtest_agent.ipynb  # backtesting
├── models/
│   ├── (s'hi desarà l'agent entrenat)
├── trading_bot/
│   ├── __init__.py
│   ├── trading_env.py  # entorn personalitzat (basat en StockTradingEnv de FinRL)
│   ├── rl_agent.py  # per carregar el model i fer prediccions
│   ├── trader_ibkr.py  # integració amb IBKR utilitzant ib_insync
│   ├── logger.py  # per al registre de les operacions (logging)
├── requirements.txt
├── README.md
└── main.py  # script que executa el bucle de trading en viu amb IBKR

