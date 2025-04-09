#import requests
#from bs4 import BeautifulSoup
#from newspaper import Article
#import time
#import csv
#import os
#
#headers = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
#}
#
#empresas = {
#    "Nvidia": "NVDA:NASDAQ",
#    "Apple": "AAPL:NASDAQ",
#    "Microsoft": "MSFT:NASDAQ"
#}
#
## Crear carpeta para guardar CSVs
#os.makedirs("noticias_finance", exist_ok=True)
#
#for nombre, ticker in empresas.items():
#    print(f"\n🔎 Buscando noticias para {nombre} ({ticker})")
#    url = f"https://www.google.com/finance/quote/{ticker}"
#    r = requests.get(url, headers=headers)
#    soup = BeautifulSoup(r.text, "html.parser")
#    with open("debug.html", "w", encoding="utf-8") as f:
#        f.write(soup.prettify())
#
#    noticias = []
#
#    for card in soup.select("div.yY3Lee")[:10]:
#        print("Buscando cards...")
#        cards = soup.select("div.yY3Lee")
#        print(f"{len(cards)} cards encontrados")
#
#        for card in cards[:10]:
#            print("Card raw HTML:", card.prettify()[:300])  # Muestra un resumen de cada card
#
#        try:
#            a_tag = card.find("a", class_="SxcTic")
#            link = "https://www.google.com" + a_tag.get("href")
#
#            fuente = card.find("div", class_="sfyJob").text.strip()
#            fecha = card.find("div", class_="Adak").text.strip()
#            titulo = a_tag.text.strip()
#
#            print(f"📰 {titulo} | {fuente} | {fecha}")
#
#            # Newspaper extrae el cuerpo completo
#            redirect = requests.get(link, headers=headers, allow_redirects=True)
#            article = Article(redirect.url, language='es')
#            article.download()
#            article.parse()
#            texto = article.text
#
#            noticias.append({
#                "titulo": titulo,
#                "fuente": fuente,
#                "fecha": fecha,
#                "enlace": redirect.url,
#                "texto": texto
#            })
#
#            print(f"✅ Texto extraído de: {redirect.url}")
#            time.sleep(1.5)
#
#        except Exception as e:
#            print(f"❌ Error con noticia: {e}")
#            continue
#
#    # Guardar en CSV
#    output_file = f"noticias_finance/{nombre}.csv"
#    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
#        writer = csv.DictWriter(f, fieldnames=["titulo", "fuente", "fecha", "enlace", "texto"])
#        writer.writeheader()
#        for noticia in noticias:
#            writer.writerow(noticia)
#
#    print(f"💾 Noticias de {nombre} guardadas en {output_file}")
import requests
from bs4 import BeautifulSoup
import csv
import os
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

empresas = {
    "Nvidia": "NVDA",
    "Apple": "AAPL",
    "Microsoft": "MSFT"
}

# Carpeta donde guardar CSVs
os.makedirs("noticias_finance", exist_ok=True)

for nombre, ticker in empresas.items():
    print(f"\n🔎 Buscando noticias para {nombre} ({ticker})")

    url = f"https://finance.yahoo.com/quote/{ticker}/news/"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    noticias = []

    news_cards = soup.select("li.stream-item.story-item")

    print(f"📄 Se encontraron {len(news_cards)} artículos")

    for card in news_cards[:10]:  # limitar a 10 por empresa
        try:
            a_tag = card.find("a", class_="subtle-link")
            if not a_tag:
                continue

            link = a_tag["href"]
            if not link.startswith("http"):
                link = "https://finance.yahoo.com" + link

            titulo = a_tag.get("title", "").strip()
            if not titulo:
                titulo = a_tag.text.strip()

            # Extraer fuente y fecha desde el div .publishing
            publishing = card.find("div", class_="publishing")
            fuente = "Desconocido"
            fecha = "Desconocida"
            if publishing:
                partes = publishing.text.strip().split("•")
                if len(partes) == 2:
                    fuente = partes[0].strip()
                    fecha = partes[1].strip()

            print(f"📰 {titulo} | {fuente} | {fecha}")

            noticias.append({
                "titulo": titulo,
                "fuente": fuente,
                "fecha": fecha,
                "enlace": link
            })

        except Exception as e:
            print(f"❌ Error extrayendo noticia: {e}")
            continue

    # Guardar en CSV
    output_file = f"noticias_finance/{nombre}.csv"
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["titulo", "fuente", "fecha", "enlace"])
        writer.writeheader()
        for noticia in noticias:
            writer.writerow(noticia)

    print(f"💾 Noticias de {nombre} guardadas en {output_file}")
    time.sleep(2)
