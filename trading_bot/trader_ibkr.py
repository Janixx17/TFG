from ib_insync import *
import pandas as pd

symbols = pd.read_csv('../stock_csv/top100.csv').iloc[:,0]

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

print("Conectado:", ib.isConnected())

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='5 mins',
    whatToShow='MIDPOINT',
    useRTH=True,
    formatDate=1
)

for bar in bars:
    print(bar.date, bar.open, bar.high, bar.low, bar.close)

ib.disconnect()
