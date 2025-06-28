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

