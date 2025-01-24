import sqlite3
import requests
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import pyupbit
from pybit.unified_trading import HTTP
import yfinance as yf


BYBIT_BASE_URL = "https://api.bybit.com"
UPBIT_BASE_URL = "https://api.upbit.com/v1"

# Rate Limit 대기 시간
RATE_LIMIT_DELAY = 0.1  

# Bybit 지원 심볼 조회
def get_bybit_symbols():
    endpoint = "/v5/market/instruments-info"
    url = f"{BYBIT_BASE_URL}{endpoint}"
    params = {"category": "linear"}
    response = requests.get(url, params=params)
    time.sleep(RATE_LIMIT_DELAY)  # 요청 후 대기
    if response.status_code == 200:
        data = response.json()
        if data.get("retCode") == 0:
            return [item['symbol'] for item in data['result']['list']]
        else:
            raise Exception(f"Bybit API Error: {data.get('retMsg')}")
    else:
        raise Exception(f"Bybit HTTP Error: {response.status_code}, {response.text}")

# Upbit 지원 마켓 조회
def get_upbit_markets():
    url = f"{UPBIT_BASE_URL}/market/all"
    response = requests.get(url)
    time.sleep(RATE_LIMIT_DELAY) 
    if response.status_code == 200:
        data = response.json()
        return {item['market']: item['korean_name'] for item in data if item['market'].startswith("KRW-")}
    else:
        raise Exception(f"Upbit HTTP Error: {response.status_code}, {response.text}")

# Bybit와 Upbit 공통 심볼 가져오기
def get_common_symbols():
    bybit_symbols = get_bybit_symbols()
    upbit_markets = get_upbit_markets()
    
    common_symbols = []
    for symbol in bybit_symbols:
        if symbol.endswith("USDT"):
            base = symbol.replace("USDT", "")
            upbit_market = f"KRW-{base}"
            if upbit_market in upbit_markets:
                common_symbols.append(base)  # "USDT" 접미사를 제거한 심볼만 저장

    return common_symbols

# DB 연결
def create_db_connection(db_name="symbols.db"):
    conn = sqlite3.connect(db_name)
    return conn

# 심볼 테이블 생성 함수
def create_table_if_not_exists(conn):
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS symbols (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT UNIQUE)''')
    conn.commit()

# 공통 심볼을 DB에 저장
def save_symbols_to_db(symbols, db_name="symbols.db"):
    conn = create_db_connection(db_name)

    # 테이블 생성
    create_table_if_not_exists(conn)

    cursor = conn.cursor()

    # 심볼을 DB에 저장
    for symbol in symbols:
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))

    conn.commit()
    conn.close()
    print(f"Symbols saved to {db_name}")

if __name__ == "__main__":
    common_symbols = get_common_symbols()
    print(f"Common Symbols: {common_symbols}")
    save_symbols_to_db(common_symbols)

# Bybit API 설정
session = HTTP(testnet=False)

# 공통 기간 계산
def calculate_time_period(day):
    end_time = datetime.now(timezone.utc)  # 현재 UTC 시간
    start_time = end_time - timedelta(days=day)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    return start_ms, end_ms , start_time, end_time

# USD/KRW 데이터 가져오기 함수
def fetch_krw_usd_multiple_requests(interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    start_time = end_time - timedelta(days=1)
    all_data = pd.DataFrame()

    # 5일씩 데이터를 요청
    for i in range(day):
        retries = 0
        while retries < max_retries:
            try:
                data = yf.download('KRW=X', interval=f"{interval}m", start=start_time, end=end_time)
                if not data.empty:
                    data = data[['Close']].rename(columns={"Close": "KRW_USD"})
                    all_data = pd.concat([all_data, data])
                break  # 데이터 가져오기가 성공하면 재시도 종료

            except Exception as e:
                print(f"Error fetching data for KRW/USD: {e}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached for KRW/USD. Skipping this request.")
                    break  # 재시도 횟수 초과 시 중지

        # 5일씩 이전으로 이동
        end_time = start_time
        start_time = end_time - timedelta(days=1)

    # 중복된 데이터 제거 후 리셋
    all_data = all_data.sort_index()
    return all_data

def get_usdt(interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    end_time_upbit = end_time
    start_time_upbit = end_time - timedelta(days=1) 
    all_data = pd.DataFrame()

    for i in range(day):
        retries = 0
        while retries < max_retries:
            try:
                df = pyupbit.get_ohlcv(ticker=f"KRW-USDT", interval=f"minute{interval}", count=1600, to=end_time_upbit, period=0.1)

                if df is None or df.empty:
                    print(f"Upbit 데이터 없음: {symbol}")
                    break

                print(f"{end_time_upbit}")
                print(f"{start_time_upbit}")

                # 5일씩 이전으로 이동
                end_time_upbit = start_time_upbit
                start_time_upbit = end_time_upbit - timedelta(days=1)

                df.index = df.index.tz_localize("Asia/Seoul").tz_convert("UTC")
                df = df[["close"]]  # 'close' 열만 남기기
                df = df.rename(columns={"close": f"KRW-USDT"})
                df.index.name = 'timestamp'
                all_data = pd.concat([all_data, df])
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
        print(f"Upbit USDT 데이터가 반환되지 않았습니다: {symbol}")

    all_data = all_data[~all_data.index.duplicated(keep='last')]
    all_data = all_data.sort_index(
    all_data = all_data.ffill()

    return all_data


def get_upbit_close_data(symbol, interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    end_time_upbit = end_time
    start_time_upbit = end_time - timedelta(days=1)
    all_data = pd.DataFrame()

    for i in range(day): 
        retries = 0
        while retries < max_retries:
            try:
                df = pyupbit.get_ohlcv(ticker=f"KRW-{symbol}", interval=f"minute{interval}", count=1600, to=end_time_upbit, period=0.1)

                if df is None or df.empty:
                    print(f"Upbit 데이터 없음: {symbol}")
                    break

                print(f"{end_time_upbit}")
                print(f"{start_time_upbit}")

                end_time_upbit = start_time_upbit
                start_time_upbit = end_time_upbit - timedelta(days=1)

                df.index = df.index.tz_localize("Asia/Seoul").tz_convert("UTC") 
                df = df[["close"]]  # 'close' 열만 남기기
                df = df.rename(columns={"close": f"KRW-{symbol}"}))
                df.index.name = 'timestamp'
                all_data = pd.concat([all_data, df])
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


def get_bybit_close_data(symbol, interval, day, max_retries=5, retry_delay=5):
    start_ms, end_ms, start_time, end_time = calculate_time_period(day)
    start_time = end_ms - int(timedelta(hours=16).total_seconds() * 1000)
    end_time = end_ms
    all_data = pd.DataFrame()
    all_closes = []

    for i in range((24*day)//16 + 1):
        retries = 0
        while retries < max_retries:
            try:
                response = session.get_kline(
                    category="linear",
                    symbol=f"{symbol}USDT",
                    interval=str(interval),
                    start=start_time,
                    end=end_time,
                    limit=1000
                )

                result_list = response['result'].get('list', [])
                if not result_list:
                    print(f"데이터가 없습니다: {symbol}")
                    break

                for data in result_list:
                    all_closes.append([data[0], data[4]])  # timestamp와 close 값만 추출

                print(f"End Time (ms): {end_time}")
                print(f"Start Time (ms): {start_time}")

                end_time = start_time
                start_time = end_time - int(timedelta(hours=16).total_seconds() * 1000)
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

    if all_closes:
        df = pd.DataFrame(all_closes, columns=["timestamp", symbol])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms', utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.rename(columns={symbol: f"{symbol[:-4]}"})  # '1INCHUSDT' -> '1INCH'
        df.index.name = 'timestamp'
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index(ascending=True)
        df = df.ffill()

    else:
        print(f"데이터가 없습니다: {symbol}")
        df = pd.DataFrame()

    return df



def get_all_symbols_from_db():
    conn = sqlite3.connect('symbols.db')
    cursor = conn.cursor()

    cursor.execute('SELECT symbol FROM symbols')
    symbols = [row[0] for row in cursor.fetchall()]

    conn.close()
    return symbols


def save_to_db(symbol, bybit_data, upbit_data):
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
        # 두 DataFrame을 timestamp 기준으로 병합 (left join)
        merged_data = pd.merge(upbit_data, bybit_data, left_index=True, right_index=True, how='outer', suffixes=('_upbit', '_bybit'))
        merged_data = merged_data.ffill()
        for timestamp, row in merged_data.iterrows():
            timestamp = timestamp.to_pydatetime()

            cursor.execute(f'''
            REPLACE INTO {table_name} (timestamp, upbit_close, bybit_close) VALUES (?, ?, ?)
            ''', (timestamp, row['KRW-' + symbol] if 'KRW-' + symbol in row else None, row[bybit_data.columns[0]] if bybit_data.columns[0] in row else None))

    conn.commit()
    conn.close()

def save_krw_usd_to_db(krw_usd_data, db_path="symbols.db"):
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

def save_krw_usdt_to_db(krw_usdt_data, db_path="symbols.db"):
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
            ''', (timestamp, row['KRW-USDT'])  # Series에서 값을 추출하여 저장

        conn.commit()

    conn.close()

def fetch_data_for_all_symbols(interval, day):
    symbols = get_all_symbols_from_db()

    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")

        upbit_data = get_upbit_close_data(symbol, interval, day)
        print(f"Upbit {symbol} 데이터:")
        print(upbit_data.head())
        print(upbit_data.tail()) 
        
        bybit_data = get_bybit_close_data(symbol, interval, day)
        print(f"Bybit {symbol} 데이터:")
        print(bybit_data.head())
        print(bybit_data.tail()) 

        save_to_db(symbol, bybit_data, upbit_data)
        print(f"Data saved to DB for {symbol}")



day = 5 
interval = 1


krw_usd_data = fetch_krw_usd_multiple_requests(interval, day)
krw_usdt_data = get_usdt(interval, day)
print(f"KRW-USD 데이터:")
print(krw_usd_data.head())
print(krw_usdt_data.head()) 

save_krw_usd_to_db(krw_usd_data)
save_krw_usdt_to_db(krw_usdt_data)
fetch_data_for_all_symbols(interval, day)

conn = sqlite3.connect('symbols.db')
cursor = conn.cursor()

cursor.execute('SELECT symbol FROM symbols')
symbols = cursor.fetchall()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
all_tables = [table[0] for table in cursor.fetchall()]

for symbol in symbols:
    symbol_name = symbol[0] 

    if symbol_name in all_tables:
        cursor.execute(f"PRAGMA table_info('{symbol_name}')")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'bybit_close_krw_usd' not in columns:
            cursor.execute(f'ALTER TABLE "{symbol_name}" ADD COLUMN bybit_close_krw_usd REAL')

        if 'bybit_close_krw_usdt' not in columns:
            cursor.execute(f'ALTER TABLE "{symbol_name}" ADD COLUMN bybit_close_krw_usdt REAL')

        if 'kimchi_premium_usd' not in columns:
            cursor.execute(f'ALTER TABLE "{symbol_name}" ADD COLUMN kimchi_premium_usd REAL')

        if 'kimchi_premium_usdt' not in columns:
            cursor.execute(f'ALTER TABLE "{symbol_name}" ADD COLUMN kimchi_premium_usdt REAL')

        cursor.execute(f'''
            UPDATE "{symbol_name}"
            SET bybit_close_krw_usd = (
                SELECT bybit_close * KRW_USD
                FROM "ex_rate"
                WHERE "ex_rate".timestamp = "{symbol_name}".timestamp
            )
        ''')

        cursor.execute(f'''
            UPDATE "{symbol_name}"
            SET bybit_close_krw_usdt = (
                SELECT bybit_close * KRW_USDT
                FROM "ex_usdt_rate"
                WHERE "ex_usdt_rate".timestamp = "{symbol_name}".timestamp
            )
        ''')

        cursor.execute(f'''
            UPDATE "{symbol_name}"
            SET kimchi_premium_usd = (
                SELECT (upbit_close / bybit_close_krw_usd - 1) * 100
                FROM "ex_rate"
                WHERE "ex_rate".timestamp = "{symbol_name}".timestamp
            )
        ''')

        cursor.execute(f'''
            UPDATE "{symbol_name}"
            SET kimchi_premium_usdt = (
                SELECT (upbit_close / bybit_close_krw_usdt - 1) * 100
                FROM "ex_usdt_rate"
                WHERE "ex_usdt_rate".timestamp = "{symbol_name}".timestamp
            )
        ''')

conn.commit()
conn.close()




