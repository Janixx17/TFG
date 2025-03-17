from os import write
import yfinance as yf
import pandas as pd

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

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

if __name__ == '__main__':

    get_top_100_index()

    data = yf.download("NVDA", start="2023-01-01", end="2024-03-01", interval="1h")
    print(data.head())





