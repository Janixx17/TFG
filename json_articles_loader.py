import requests
import time
import json

key = "t0HAgHWbJ6c0rcXtJlQzZfCwRKHtjmVI"

seccions = ["World", "Business Day", "Your Money", "Technology", "U.S.", "Science"]

def en_seccio(article):
    seccio = article.get("section_name", "")
    return seccio in seccions

if __name__ == '__main__':

    id = 1

    with open("json/articles.json", "w", encoding="utf-8") as f:
        f.write("[\n")

    for year in range(2025, 2010,-1):
        for month in range(12,0,-1):
            if not (year==2025 and month>5):
                print(f"Data:{year}-{month}")
                url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={key}"
                try:
                    resposta = requests.get(url)
                    dades = resposta.json()
                    articles = dades["response"]["docs"]

                    bons = [a for a in articles if en_seccio(a)]

                    for art in bons:

                        json_object = {
                            "id": id,
                            "title": art.get("headline", {}).get("main", "Null"),
                            "url": art.get("web_url", ""),
                            "date": art.get("pub_date", ""),
                            "file": ""
                        }
                        with open("json/articles.json", "a", encoding="utf-8") as f:
                            f.write(json.dumps(json_object, ensure_ascii=False, indent=4))
                            f.write(",\n")

                        id += 1
                        titol = art.get("headline", {}).get("main", "Sense títol")
                        url_art = art.get("web_url", "")
                        data = art.get("pub_date", "")
                        seccio = art.get("section_name", "")
                        print(f"📰 [{data[:10]}] ({seccio}): {titol} - {url_art}")

                    time.sleep(10)  #per no sobrecarregar l'API

                except Exception as e:
                    print(f"❌ Error en processar {year}-{month:02d}: {e}, response code:{resposta.status_code}, response body:{resposta.text}")
                    time.sleep(20)

    with open("json/articles.json", "a", encoding="utf-8") as f:
        f.write("]")