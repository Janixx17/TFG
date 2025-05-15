from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

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


if __name__ == "__main__":

    with open("json/articles.json", "r", encoding="utf-8") as f: #carreguem els articles
        articles = json.load(f)



    with open("json/cookies.json", "r", encoding="utf-8") as f:
        cookies = json.load(f)

    options = Options()
    options.add_argument("--headless")  #amagar finestra

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get("https://www.nytimes.com/2025/04/30/world/middleeast/uk-us-yemen-houthis-strikes.html")
    time.sleep(3)

    for cookie in cookies:
        safe_cookie = sanitize_cookie(cookie)
        driver.add_cookie(safe_cookie)

    # Recargar ya autenticado
    time.sleep(1)

    with open("noticia.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    driver.quit()