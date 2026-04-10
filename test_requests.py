import urllib.request
import pandas as pd
import io

def test_api():
    url = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot?interval=ONE_FOURTH_HOUR&dp=SELECT_RANGE&showGraph=false&fromDate=01-02-2024&toDate=01-02-2024"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        dfs = pd.read_html(io.StringIO(html))
        print("Tables shape:", [df.shape for df in dfs])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_api()
