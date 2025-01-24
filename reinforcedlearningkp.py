import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import sqlite3
import keras
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

def get_symbol_data(symbol, timestamp):
    conn = sqlite3.connect('symbols.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT timestamp, kimchi_premium_usdt, bybit_close_krw_usdt, upbit_close FROM {symbol} WHERE timestamp = ?", (timestamp,))
    data = cursor.fetchone()
    conn.close()
    if data:
        kimchi_premium, bybit_price, upbit_price, timestamp = data
        if any(x is None or pd.isna(x) for x in [kimchi_premium, bybit_price, upbit_price, timestamp]):
            return None
        return kimchi_premium, bybit_price, upbit_price, timestamp
    else:
        return None

def clean_symbol_data(symbol):
    conn = sqlite3.connect('symbols.db')
    query = f"SELECT * FROM {symbol}"
    df = pd.read_sql(query, conn)
    df_cleaned = df.dropna(subset=['kimchi_premium_usdt', 'bybit_close_krw_usdt', 'upbit_close','timestamp'])
    df_cleaned.to_sql(symbol, conn, if_exists='replace', index=False)
    conn.close()

clean_symbol_data('AAVE')

conn = sqlite3.connect('symbols.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM AAVE")
data = cursor.fetchall()
upbit_prices = [row[1] for row in data]
bybit_prices = [row[4] for row in data]
kimchi_premium = [row[6] for row in data]
dataset1 = np.array(list(zip(upbit_prices, bybit_prices, kimchi_premium)))
conn.close()

columns_to_select = [0, 1, 2]
dataset = pd.DataFrame(dataset1, columns=["col1", "col2", "col3"])
selected_data = dataset[["col3"]]

X = selected_data.values.flatten()
X = [float(x) for x in X]

dataset = dataset.fillna(method='ffill')
selected_data = dataset.iloc[:, [2]]
X = selected_data.values.flatten()
X = [float(x) for x in X]

validation_size = 0.2
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = tf.keras.models.load_model(model_name) if is_eval else self._model()

    def _model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(tf.keras.layers.Dense(units=32, activation="relu"))
        model.add(tf.keras.layers.Dense(units=8, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label = 'Selling signal', markevery = states_sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    plt.show()

window_size = 1
agent = Agent(window_size)
data = X_train
l = len(data) - 1
batch_size = 32
episode_count = 10

for e in range(episode_count + 1):
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    
    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        
        if action == 1:
            upbit_buy_price = dataset.iloc[t, 0]
            bybit_sell_price = dataset.iloc[t, 1]
            agent.inventory.append(upbit_buy_price)
            agent.inventory.append(bybit_sell_price)
            states_buy.append(t)
            
        elif action == 2 and len(agent.inventory) > 0:
            bought_upbit_price = agent.inventory.pop(0)
            bought_bybit_price = agent.inventory.pop(0)
            upbit_close = dataset.iloc[t, 0]
            bybit_close_krw_usdt = dataset.iloc[t, 0]
            upbit_fee = 0.0005
            bybit_fee = 0.00055
            upbit_sell_fee = upbit_close * upbit_fee
            bybit_close_fee = bybit_close_krw_usdt * bybit_fee
            upbit_profit = upbit_close - bought_upbit_price - upbit_sell_fee
            bybit_profit = bought_bybit_price - bybit_close_krw_usdt - bybit_close_fee
            reward = upbit_profit + bybit_profit
            total_profit += reward
            states_sell.append(t)
            
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        
        if done:
            plot_behavior(data, states_buy, states_sell, total_profit)
            
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
            
    if e % 2 == 0:
        agent.model.save(working_dir + "model_ep" + str(e))
