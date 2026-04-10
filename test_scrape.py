from playwright.sync_api import sync_playwright
import pandas as pd
import io

def test_scrape():
    url = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot?interval=ONE_FOURTH_HOUR&dp=SELECT_RANGE&showGraph=false&fromDate=01-02-2024&toDate=01-02-2024"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.wait_for_selector("table", timeout=10_000)
        
        html = page.content()
        dfs = pd.read_html(io.StringIO(html))
        
        for i, df in enumerate(dfs):
            print(f"Table {i} shape: {df.shape}")
            if df.shape[0] > 50:
                print("Columns:", df.columns.tolist())
                print(df.head(5))
                break
                
        browser.close()

if __name__ == "__main__":
    test_scrape()
