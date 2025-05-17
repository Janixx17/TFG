from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import os

VALID_SAMESITE_VALUES = ["Strict", "Lax", "None"]

def sanitize_cookie(cookie):
    # Asegurarse que 'sameSite' tenga un valor permitido o eliminarlo
    same_site = cookie.get("sameSite")
    if same_site is not None:
        # Normaliza el valor (por si viene en minúsculas)
        same_site = same_site.capitalize()
        if same_site not in VALID_SAMESITE_VALUES:
            del cookie["sameSite"]
        else:
            cookie["sameSite"] = same_site
    return cookie

def request_article(article, cookies):
    id = article['id']

    options = Options()
    options.add_argument("--headless")  # amagar finestra

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(article['url'])
    time.sleep(3)

    for cookie in cookies:
        safe_cookie = sanitize_cookie(cookie)
        driver.add_cookie(safe_cookie)

    # Recargar ya autenticado
    time.sleep(1)

    with open(f"article{id}.html", "w", encoding="utf-8") as f:
        f.write(BeautifulSoup(driver.page_source).prettify())

    # write in the file the article not the source code

    with open(f"article{id}.txt", "r", encoding="utf-8") as f:
        if not f.read():
            raise TypeError("Null file exception")

    driver.quit()


if __name__ == "__main__":

    with open("json/articles.json", "r+", encoding="utf-8") as f: #carreguem els articles
        articles = json.load(f)

    if not os.path.exists("json/cookies.csv"):
        raise FileNotFoundError(f"L'arxiu cookies.csv no existeix.")

    with open("json/cookies.json", "r", encoding="utf-8") as f:
        cookies = json.load(f)

        for article in articles['articles']:
            if article['file'] == "":
                article['file'] = f"article{article['id']}.txt"
                try:
                    request_article(article, cookies)
                    a = 3
                except:
                    print("Occured an exception")
                    time.sleep(60)

                f.seek(0)
                json.dump(articles, f, indent=4, ensure_ascii=False)
                f.truncate()


