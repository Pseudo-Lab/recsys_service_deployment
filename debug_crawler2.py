"""
Debug script v2 - Wait for dynamic content
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

print("⏳ Waiting 10 seconds for JavaScript to load content...")
time.sleep(10)

# Try scrolling to trigger lazy loading
print("📜 Scrolling to trigger lazy loading...")
for i in range(3):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

print("\n🔍 Analyzing page with Selenium...")

# Try finding elements by common news list patterns
try:
    print("\n1️⃣ Looking for links by XPath (contains 'news')...")
    news_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'news')]")
    print(f"   Found {len(news_links)} links")
    for i, link in enumerate(news_links[:10]):
        href = link.get_attribute('href')
        text = link.text.strip()[:50]
        print(f"   - {i+1}. {text} | {href}")
except Exception as e:
    print(f"   Error: {e}")

try:
    print("\n2️⃣ Looking for all <a> tags...")
    all_links = driver.find_elements(By.TAG_NAME, "a")
    print(f"   Found {len(all_links)} total links")

    # Filter for potential news links
    news_like = []
    for link in all_links:
        href = link.get_attribute('href')
        text = link.text.strip()
        if href and ('news' in href or 'article' in href or len(text) > 20):
            news_like.append((href, text[:50]))

    print(f"   Found {len(news_like)} potential news links")
    for i, (href, text) in enumerate(news_like[:10]):
        print(f"   - {i+1}. {text} | {href}")
except Exception as e:
    print(f"   Error: {e}")

try:
    print("\n3️⃣ Looking for list items (li)...")
    list_items = driver.find_elements(By.TAG_NAME, "li")
    print(f"   Found {len(list_items)} list items")
    for i, item in enumerate(list_items[:5]):
        text = item.text.strip()[:80]
        if len(text) > 20:
            print(f"   - {i+1}. {text}")
            # Check if it contains a link
            try:
                link = item.find_element(By.TAG_NAME, "a")
                print(f"       Link: {link.get_attribute('href')}")
            except:
                pass
except Exception as e:
    print(f"   Error: {e}")

print("\n4️⃣ Saving page source to file for inspection...")
with open('/Users/kyeongchanlee/projects/recsys_service_deployment/page_source.html', 'w', encoding='utf-8') as f:
    f.write(driver.page_source)
print("   Saved to page_source.html")

print("\n✅ Debug complete. Browser window will stay open for 20 seconds for manual inspection...")
print("   💡 Please manually inspect the browser to see if news items are visible")
time.sleep(20)

driver.quit()
print("🔚 Browser closed")
