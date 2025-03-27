from os import write
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np

def get_simbols():
    link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks'
    tables = pd.read_html(link)
    sp500_df = tables[0]
    return sp500_df["Symbol"].tolist()

def get_index_weight(symbol):
    try:
        yt = yf.Ticker(symbol).info['marketCap']
    except:
        return symbol, 0

    return symbol, yt

def get_top_100_index():
    import csv
    sp500_df = get_simbols()
    sptop100 = []

    with open("top100.csv", "w", newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Symbol", "Weight"])

        for symbol in sp500_df:
            sym = get_index_weight(symbol)
            sptop100.append(sym)

        sptop100.sort(key=lambda x: x[1], reverse=True)
        sptop100 = sptop100[:100]

        writer.writerows(sptop100)

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



def calculate_time(time, i):
    if i < 4:
        return dt.date.today() - dt.timedelta(days=time*(i-1)), dt.date.today() - dt.timedelta(days=time*i)
    else:
        return dt.date.today() - dt.timedelta(days=time * (i - 1)), dt.date.today() - dt.timedelta(days=time * i-3)


if __name__ == '__main__':

    #get_top_100_index()

    symbols = np.loadtxt("top100.csv", delimiter=",", dtype=str, skiprows=1)
    symbols = symbols[:, 0]
    times = np.loadtxt("times.csv", delimiter=",", dtype=int, skiprows=1)

    #auxsymbols = symbols[0:2]
    #auxtimes = times[1:3]

    get_data(symbols, times)


    #data = yf.download("NVDA", start="2025-02-24", end="2025-02-28", interval="1m")
    #print(data.head())





