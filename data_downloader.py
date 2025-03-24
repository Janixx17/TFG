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
            #if time[:, 1] == 8:#todo
            #    continue
            print(symbol, time)
            day1, day2 = calculate_time(int(time[1])) #time must be numpy array
            if time[0] != 7299:
                data = yf.download(symbol, end=day1.today(), start=day2, interval=f"{time[0]}m")
                data.to_csv(f"data/{symbol}_{day1}_{day2}_{time[0]}m.csv")
            else:
                data = yf.download(symbol, end=day1, start=day2, interval="1d")
                data.to_csv(f"data/{symbol}_{day1}_{day2}_1d.csv")



def calculate_time(time):
    return dt.date.today(), dt.date.today() - dt.timedelta(days=time)


if __name__ == '__main__':

    #get_top_100_index()

    symbols = np.loadtxt("top100.csv", delimiter=",", dtype=str, skiprows=1)
    symbols = symbols[:, 0]
    times = np.loadtxt("times.csv", delimiter=",", dtype=int, skiprows=1)

    auxsymbols = symbols[0:2]
    auxtimes = times[1:3]

    get_data(auxsymbols, auxtimes)


    #data = yf.download("NVDA", start="2002-01-01", end="2025-03-23", interval="1h")
    #print(data.head())





