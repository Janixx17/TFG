import requests
import json
import time
from bs4 import BeautifulSoup

key = "t0HAgHWbJ6c0rcXtJlQzZfCwRKHtjmVI"

seccions = ["World", "Business Day", "Your Money", "Technology", "U.S.", "Science"]
tags_financers = [
    "Stock Market", "Stocks", "Investing", "Investments", "S&P 500",
    "NASDAQ", "Dow Jones", "Wall Street", "Bonds", "ETFs", "Mutual Funds",
    "Trading", "Equities", "Financial Markets", "Commodities",
    "Cryptocurrency", "Bitcoin", "Earnings", "IPO", "Initial Public Offering",
    "SEC", "Federal Reserve", "Interest Rates", "Inflation",
    "Recession", "Economic Growth", "Corporate Finance", "Mergers and Acquisitions",
    "Venture Capital", "Private Equity", "Business News", "Company News",
    "Profits", "Losses", "Revenue", "Quarterly Results", "Shareholders",
    "Dividends", "Financial Reports", "Economic Policy", "Fiscal Policy",
    "Monetary Policy", "Capital Markets", "Financial Services", "Banking",
    "Fintech", "Credit", "Debt", "Tax", "Spending", "Budget"
]
def en_seccio(article):
    seccio = article.get("section_name", "")
    return seccio in seccions

def te_tag_financer(article):
    tags = [t.get("value", "") for t in article.get("keywords", [])]
    return any(tag in tags_financers for tag in tags)

def download_news(link):
    url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q=election&api-key={key}"
    r = requests.get(link)
    return r

if __name__ == '__main__':

    new = download_news(link)
    print(new.status_code)
    print(new.text)


    for year in range(2025, 2010,-1):
        for month in range(12,0,-1):
            if not (year==2025 and month>4):
                print(f"Data:{year}-{month}")
                url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={key}"
                try:
                    resposta = requests.get(url)
                    dades = resposta.json()
                    articles = dades["response"]["docs"]

                    bons = [a for a in articles if en_seccio(a)]
                    #bons = [a for a in articles if en_seccio(a) and te_tag_financer(a)]

                    for art in bons:
                        titol = art.get("headline", {}).get("main", "Sense títol")
                        url_art = art.get("web_url", "")
                        data = art.get("pub_date", "")
                        seccio = art.get("section_name", "")
                        print(f"📰 [{data[:10]}] ({seccio}): {titol} - {url_art}")

                    time.sleep(6)  # per no sobrecarregar l'API
                except Exception as e:
                    print(f"❌ Error en processar {year}-{month:02d}: {e}")
