"""
from ib_insync import *
import pandas as pd

symbols = pd.read_csv('../stock_csv/top100.csv').iloc[:,0]

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

print("Conectado:", ib.isConnected())



contract = Stock('AAPL', 'SMART', 'USD')
market_data = ib.reqMktData(contract, '', True, False)
ib.sleep(2)

print(f"Precio actual de AAPL:")
print(f"Last: {market_data.last}")
print(f"Bid: {market_data.bid}")
print(f"Ask: {market_data.ask}")

ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 mins',
    whatToShow='MIDPOINT',
    useRTH=True,
    formatDate=1
)

for bar in bars:
    print(bar.date, bar.open, bar.high, bar.low, bar.close)

ib.disconnect()
"""
from ib_insync import IB, Stock
import datetime

# Conectar a TWS (puerto por defecto es 7497 para TWS, 4001 para IB Gateway)
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Lista de símbolos que quieres revisar
symbols = ['AAPL', 'GOOG', 'MSFT']

providers = ["DJ-RTG", "DJ-RTPRO", "BRFG", "BRFUPDN", "DJ-N"]

"""
# Obtener proveedores de noticias disponibles para tu cuenta
providers = ib.reqNewsProviders()
print("Proveedores disponibles:")
for p in providers:
    print(f" - {p.code}: {p.name}")

# Usaremos el primer proveedor disponible (puedes cambiarlo si lo deseas)
if not providers:
    print("No hay proveedores de noticias disponibles.")
    ib.disconnect()
    exit()
"""

# Recorrer los símbolos
for symbol in symbols:
    print(f"\n🔍 Buscando noticias para {symbol}...")

    # Crear contrato del símbolo
    contract = Stock(symbol, 'SMART', 'USD')

    # Obtener detalles del contrato para conseguir el conId
    details = ib.reqContractDetails(contract)
    if not details:
        print(f"❌ No se pudo obtener el contrato para {symbol}")
        continue

    conId = details[0].contract.conId
    news_found = False

    # Probar cada proveedor hasta encontrar noticias
    for provider_code in providers:
        print(f"   🔄 Probando proveedor: {provider_code}")
        
        try:
            # Obtener noticias históricas (1 noticia más reciente)
            news = ib.reqHistoricalNews(
                conId=conId,
                providerCodes=provider_code,
                startDateTime='',
                totalResults=1,
                endDateTime=''
            )

            if news:
                print(f"   ✅ Noticias encontradas con proveedor: {provider_code}")
                
                # Descargar cuerpo de la noticia
                article_id = news[0].articleId
                body = ib.reqNewsArticle(articleId=article_id, providerCode=provider_code)

                print(f"📰 Titular: {news[0].headline}")
                print(f"🕒 Fecha: {news[0].time}")
                print(f"📄 Cuerpo:\n{body.articleText[:1000]}")  # Puedes mostrar más si quieres
                
                news_found = True
                break  # Salir del bucle de proveedores ya que encontramos noticias
            else:
                print(f"   ⚠️ No hay noticias disponibles con proveedor: {provider_code}")
                
        except Exception as e:
            print(f"   ❌ Error con proveedor {provider_code}: {e}")
            continue
    
    if not news_found:
        print(f"❌ No se encontraron noticias para {symbol} con ningún proveedor.")

# Desconectar al final
ib.disconnect()
