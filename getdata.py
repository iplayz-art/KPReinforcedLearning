import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import pyupbit
from pybit.unified_trading import HTTP
import yfinance as yf

def get_bybit_symbols():
    endpoint = "/v5/market/instruments-info"
    url = f"{BYBIT_BASE_URL}{endpoint}"
    params = {"category": "linear"}
    response = requests.get(url, params=params)
    time.sleep(RATE_LIMIT_DELAY)
    if response.status_code == 200:
        data = response.json()
        if data.get("retCode") == 0:
            return [item['symbol'] for item in data['result']['list']]
        else:
            raise Exception(f"Bybit API Error: {data.get('retMsg')}")
    else:
        raise Exception(f"Bybit HTTP Error: {response.status_code}, {response.text}")

def get_upbit_markets():
    url = f"{UPBIT_BASE_URL}/market/all"
    response = requests.get(url)
    time.sleep(RATE_LIMIT_DELAY)
    if response.status_code == 200:
        data = response.json()
        return {item['market']: item['korean_name'] for item in data if item['market'].startswith("KRW-")}
    else:
        raise Exception(f"Upbit HTTP Error: {response.status_code}, {response.text}")

def get_common_symbols():
    bybit_symbols = get_bybit_symbols()
    upbit_markets = get_upbit_markets()
    common_symbols = []
    for symbol in bybit_symbols:
        if symbol.endswith("USDT"):
            base = symbol.replace("USDT", "")
            upbit_market = f"KRW-{base}"
            if upbit_market in upbit_markets:
                common_symbols.append(base)

    return common_symbols

def create_db_connection(db_name="symbols.db"):
    conn = sqlite3.connect(db_name)
    return conn

def create_table_if_not_exists(conn):
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS symbols (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT UNIQUE)''')
    conn.commit()

def save_symbols_to_db(symbols, db_name="symbols.db"):
    conn = create_db_connection(db_name)
    create_table_if_not_exists(conn)
    cursor = conn.cursor()
    for symbol in symbols:
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))

    conn.commit()
    conn.close()
    print(f"Symbols saved to {db_name}")

session = HTTP(testnet=False)

def calculate_time_period(day):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=day)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    return start_ms, end_ms , start_time, end_time

def fetch_krw_usd(interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    all_data = pd.DataFrame()
    start_time = start_time - timedelta(days=2)
    retries = 0
    while retries < max_retries:
        try:
            data = yf.download('KRW=X', interval=f"{interval}m", start=start_time, end=end_time)
            if not data.empty:
                data = data[['Close']].rename(columns={"Close": "KRW_USD"})
                data.index = pd.to_datetime(data.index)
                data.index = data.index.tz_localize('UTC', ambiguous='NaT')
                all_data = data
            break
        except Exception as e:
            print(f"Error fetching data for KRW/USD: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for KRW/USD. Skipping this request.")
                break

    return all_data


def fetch_krw_usdt(interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    all_data = pd.DataFrame()

    retries = 0
    while retries < max_retries:
        try:
            df = pyupbit.get_ohlcv(ticker=f"KRW-USDT", interval=f"minute{interval}", count=1440*day//interval, to=end_time, period=0.1)
            if df is None or df.empty:
                print(f"Upbit 데이터 없음: USDT")
                break

            df.index = df.index.tz_localize("Asia/Seoul").tz_convert("UTC")
            df = df[["close"]].rename(columns={"close": f"KRW-USDT"})
            df.index.name = 'timestamp'

            all_data = df
            break
        except Exception as e:
            print(f"Error fetching data for USDT: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for USDT. Skipping this request.")
                break

    if all_data.empty:
        print(f"Upbit USDT 데이터가 반환되지 않았습니다: USDT")

    all_data = all_data[~all_data.index.duplicated(keep='last')]
    all_data = all_data.sort_index()
    all_data = all_data.ffill()

    return all_data

def fetch_upbit(symbol, interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    all_data = pd.DataFrame()

    retries = 0
    while retries < max_retries:
        try:
            df = pyupbit.get_ohlcv(ticker=f"KRW-{symbol}", interval=f"minute{interval}", count=1440*day //interval, to=end_time, period=0.1)
            if df is None or df.empty:
                print(f"Upbit 데이터 없음: {symbol}")
                break

            df.index = df.index.tz_localize("Asia/Seoul").tz_convert("UTC")
            df = df[["close"]].rename(columns={"close": f"KRW-{symbol}"})
            df.index.name = 'timestamp'

            all_data = df
            break
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for {symbol}. Skipping this request.")
                break

    if all_data.empty:
        print(f"Upbit 데이터가 반환되지 않았습니다: {symbol}")

    all_data = all_data[~all_data.index.duplicated(keep='last')]
    all_data = all_data.sort_index()
    all_data = all_data.ffill()

    return all_data

def fetch_bybit(symbol, interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    all_data = pd.DataFrame()
    all_closes = []

    retries = 0
    while retries < max_retries:
        try:
            response = session.get_kline(
                category="linear",
                symbol=f"{symbol}USDT",
                interval=str(interval),
                start=start_ms,
                end=end_ms,
                limit=1000
            )

            result_list = response['result'].get('list', [])
            if not result_list:
                print(f"데이터가 없습니다: {symbol}")
                break

            for data in result_list:
                all_closes.append([data[0], data[4]])

            all_data = pd.DataFrame(all_closes, columns=["timestamp", symbol])
            all_data["timestamp"] = pd.to_datetime(all_data["timestamp"].astype(int), unit='ms', utc=True)
            all_data.set_index("timestamp", inplace=True)
            all_data = all_data.rename(columns={symbol: f"{symbol[:-4]}"})
            all_data.index.name = 'timestamp'

            all_data = all_data[~all_data.index.duplicated(keep='last')]
            all_data = all_data.sort_index(ascending=True)
            all_data = all_data.ffill()
            time.sleep(0.1)

            break

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for {symbol}. Skipping this request.")
                break

    if all_data.empty:
        print(f"데이터가 없습니다: {symbol}")

    return all_data

def get_symbols():
    conn = sqlite3.connect('symbols.db')
    cursor = conn.cursor()

    cursor.execute('SELECT symbol FROM symbols')
    symbols = [row[0] for row in cursor.fetchall()]

    conn.close()
    return symbols


def save_upbit_bybit(symbol, bybit_data, upbit_data):
    conn = sqlite3.connect('symbols.db')
    cursor = conn.cursor()

    table_name = f"'{symbol}'"
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        timestamp DATETIME PRIMARY KEY,
        upbit_close REAL,
        bybit_close REAL
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ex_rate (
        timestamp DATETIME PRIMARY KEY,
        krw_usd REAL
    )''')

    if not upbit_data.empty and not bybit_data.empty:
        merged_data = pd.merge(upbit_data, bybit_data, left_index=True, right_index=True, how='outer', suffixes=('_upbit', '_bybit'))
        merged_data = merged_data.ffill()
        for timestamp, row in merged_data.iterrows():
            timestamp = timestamp.to_pydatetime()
            cursor.execute(f'''
            REPLACE INTO {table_name} (timestamp, upbit_close, bybit_close) VALUES (?, ?, ?)
            ''', (timestamp, row['KRW-' + symbol] if 'KRW-' + symbol in row else None, row[bybit_data.columns[0]] if bybit_data.columns[0] in row else None))
    conn.commit()
    conn.close()

def save_krw_usd(krw_usd_data, db_path="symbols.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ex_rate (
        timestamp DATETIME PRIMARY KEY,
        krw_usd REAL
    )
    ''')

    if not krw_usd_data.empty:
        krw_usd_data = krw_usd_data.ffill()
        for timestamp, row in krw_usd_data.iterrows():
            timestamp = timestamp.to_pydatetime()
            cursor.execute('''
            REPLACE INTO ex_rate (timestamp, krw_usd) VALUES (?, ?)
            ''', (timestamp, row['KRW_USD'].values[0]))
        conn.commit()
    conn.close()

def save_krw_usdt(krw_usdt_data, db_path="symbols.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ex_usdt_rate (
        timestamp DATETIME PRIMARY KEY,
        krw_usdt REAL
    )
    ''')

    if not krw_usdt_data.empty:
        krw_usdt_data = krw_usdt_data.ffill()
        for timestamp, row in krw_usdt_data.iterrows():
            timestamp = timestamp.to_pydatetime()
            cursor.execute('''
            REPLACE INTO ex_usdt_rate (timestamp, krw_usdt) VALUES (?, ?)
            ''', (timestamp, row['KRW-USDT']))
        conn.commit()
    conn.close()

def fetch_alldata(interval, day):
    symbols = get_symbols()

    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")

        upbit_data = fetch_upbit(symbol, interval, day)
        print(f"Upbit {symbol} 데이터:")
        print(upbit_data.head())
        print(upbit_data.tail())

        bybit_data = fetch_bybit(symbol, interval, day)
        print(f"Bybit {symbol} 데이터:")
        print(bybit_data.head())
        print(bybit_data.tail())

        save_upbit_bybit(symbol, bybit_data, upbit_data)
        print(f"Data saved to DB for {symbol}")

symbol = 'BTC'
interval = 60
day = 5

BYBIT_BASE_URL = "https://api.bybit.com"
UPBIT_BASE_URL = "https://api.upbit.com/v1"
RATE_LIMIT_DELAY = 0.1

# fetch_krw_usd(interval, day)
# fetch_krw_usdt(interval, day)
# fetch_upbit(symbol, interval, day)
# fetch_bybit(symbol, interval, day)


if __name__ == "__main__":
    common_symbols = get_common_symbols()
    print(f"Common Symbols: {common_symbols}")
    save_symbols_to_db(common_symbols)


krw_usd_data = fetch_krw_usd(interval, day)
krw_usdt_data = fetch_krw_usdt(interval, day)

print(f"KRW-USD 데이터:")
print(krw_usd_data.head())
print(f"KRW-USD 데이터:")
print(krw_usdt_data.head())

save_krw_usd(krw_usd_data)
save_krw_usdt(krw_usdt_data)
fetch_alldata(interval, day)
