"""
Debug script to test news crawler
"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Non-headless for debugging
opts = Options()
# opts.add_argument("--headless=new")  # Comment out for visual debugging
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-gpu")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=1920,1080")
opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36")

driver = webdriver.Chrome(options=opts)

stock_code = "066570"
LIST_URL = f"https://www.tossinvest.com/stocks/{stock_code}/news?menu=news&symbol-or-stock-code={stock_code}"

print(f"📍 Opening URL: {LIST_URL}")
driver.get(LIST_URL)

print("⏳ Waiting for page to load...")
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
except Exception as e:
    print(f"⚠️  Page load warning: {e}")

print("\n🔍 Analyzing page source...")
soup = BeautifulSoup(driver.page_source, "html.parser")

# Try different selectors
print("\n1️⃣ Trying original selector: a[href*='/stocks/'][href*='news']")
news_items_1 = soup.select("a[href*='/stocks/'][href*='news']")
print(f"   Found {len(news_items_1)} items")
for i, item in enumerate(news_items_1[:3]):
    print(f"   - {i+1}. {item.get('href')}")

print("\n2️⃣ Trying: a[href*='/news/']")
news_items_2 = soup.select("a[href*='/news/']")
print(f"   Found {len(news_items_2)} items")
for i, item in enumerate(news_items_2[:3]):
    print(f"   - {i+1}. {item.get('href')}")

print("\n3️⃣ Trying: all <a> tags")
all_links = soup.find_all("a", href=True)
print(f"   Found {len(all_links)} total links")
news_related = [link for link in all_links if 'news' in link['href'].lower()]
print(f"   Found {len(news_related)} news-related links")
for i, item in enumerate(news_related[:5]):
    print(f"   - {i+1}. {item.get('href')}")

print("\n4️⃣ Looking for article or list elements")
articles = soup.find_all(['article', 'li', 'div'], class_=lambda x: x and ('news' in x.lower() or 'article' in x.lower() or 'item' in x.lower()))
print(f"   Found {len(articles)} potential article containers")

print("\n✅ Debug complete. Browser window will stay open for 30 seconds for manual inspection...")
time.sleep(30)

driver.quit()
print("🔚 Browser closed")
