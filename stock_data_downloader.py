import yfinance as yf
import datetime as dt
import numpy as np

def get_data(Symbols, times): #TODO guardar-la
    for symbol in Symbols:
        for time in times:
            if time[1] == 8:
                for i in range(1, 5):  # El segundo valor (5) no se incluye, así que va de 1 a 4
                    day1, day2 = calculate_time(int(time[1]), i)
                    data = yf.download(symbol, end=day1, start=day2, interval=f"{time[0]}m")
                    data.to_csv(f"data/{symbol}_{time[0]}m_{day1}_{day2}.csv")

            print(symbol, time)
            day1, day2 = calculate_time(int(time[1]), 1) #time must be numpy array
            if time[0] != 1440:
                data = yf.download(symbol, end=day1.today(), start=day2, interval=f"{time[0]}m")
                data.to_csv(f"data/{symbol}_{time[0]}m_{day1}_{day2}.csv")
            else:
                data = yf.download(symbol, end=day1, start=day2, interval="1d")
                data.to_csv(f"data/{symbol}_1d_{day1}_{day2}.csv")


def get_data2(Symbols, times):
    for symbol in Symbols:
        for time in times:
            if time[1] == 8:
                for i in range(1, 5):  # El segon valor (5) no es inclòs, així que va de 1 a 4
                    day1, day2 = calculate_time(int(time[1]), i)
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=day2, end=day1, interval=f"{time[0]}m")

                    # Obtenir dades extra
                    info = ticker.info
                    bid = info.get("bid", np.nan)
                    ask = info.get("ask", np.nan)
                    market_cap = info.get("marketCap", np.nan)

                    # Afegir columnes extra
                    data["Bid"] = bid
                    data["Ask"] = ask
                    data["MarketCap"] = market_cap

                    data.to_csv(f"data/{symbol}_{time[0]}m_{day1}_{day2}2222.csv")

            print(symbol, time)
            day1, day2 = calculate_time(int(time[1]), 1)

            ticker = yf.Ticker(symbol)
            if time[0] != 1440:
                data = ticker.history(start=day2, end=day1, interval=f"{time[0]}m")
            else:
                data = ticker.history(start=day2, end=day1, interval="1d")

            # Obtenir dades extra
            info = ticker.info
            bid = info.get("bid", np.nan)
            ask = info.get("ask", np.nan)
            market_cap = info.get("marketCap", np.nan)

            # Afegir columnes extra
            data["Bid"] = bid
            data["Ask"] = ask
            data["MarketCap"] = market_cap

            data.to_csv(f"data/{symbol}_{time[0]}m_{day1}_{day2}22222222.csv")


def calculate_time(time, i):
    if i < 4:
        return dt.date.today() - dt.timedelta(days=time*(i-1)), dt.date.today() - dt.timedelta(days=time*i)
    else:
        return dt.date.today() - dt.timedelta(days=time * (i - 1)), dt.date.today() - dt.timedelta(days=time * i-3)


if __name__ == '__main__':

    symbols = np.loadtxt("stock_csv/top100.csv", delimiter=",", dtype=str, skiprows=1)
    symbols = symbols[:, 0]
    times = np.loadtxt("stock_csv/times.csv", delimiter=",", dtype=int, skiprows=1)

    #auxsymbols = symbols[0:2]
    #auxtimes = times[1:3]

    get_data2(symbols, times)

    #data = yf.download("NVDA", start="2025-02-24", end="2025-02-28", interval="1m")
    #print(data.head())





