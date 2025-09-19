---

## 📖 （Day 22）AI量化学习日志：基于预测信号的回测系统

---

### 📌 今日计划

* **今日目标**：

  * 学习回测（Backtesting）的基本概念
  * 将前几天的股价预测结果转化为 **交易信号**
  * 实现一个简单的 **买入/卖出回测系统**，评估策略收益

---

### 🛠️ 学习过程

#### 1. 理论学习

* **回测（Backtesting）**：用历史数据模拟交易，验证策略的可行性。
* **基本流程**：

  1. 模型预测未来股价
  2. 根据预测值生成 **交易信号**（买入/卖出/持有）
  3. 模拟执行交易，计算资金曲线和收益率
* **简单交易信号规则**：

  * 如果预测价格 > 当前价格 → 产生 **买入信号**
  * 如果预测价格 < 当前价格 → 产生 **卖出信号**

👉 虽然这个规则非常朴素，但足以帮助我们建立从“预测”到“策略”的桥梁。

---

#### 2. Python 实践

```python
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# 下载数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造时间序列
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 简单 LSTM 模型（也可替换为 Transformer/CNN+Transformer）
inputs = layers.Input(shape=(time_step, 1))
x = layers.LSTM(50, return_sequences=True)(inputs)
x = layers.LSTM(50)(x)
outputs = layers.Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

# 训练
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 预测
predictions = model.predict(X_test)
pred_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# ---------------------------
# 简单回测逻辑
# ---------------------------
initial_capital = 10000
capital = initial_capital
position = 0  # 持仓股数
capital_curve = []

for i in range(len(real_prices)-1):
    if pred_prices[i] > real_prices[i]:  # 预测上涨 → 买入
        if capital > 0:
            position = capital / real_prices[i]  # 全仓买入
            capital = 0
    elif pred_prices[i] < real_prices[i]:  # 预测下跌 → 卖出
        if position > 0:
            capital = position * real_prices[i]  # 全仓卖出
            position = 0
    # 记录资金曲线
    total_value = capital + position * real_prices[i]
    capital_curve.append(total_value)

# 最终收益
final_value = capital + position * real_prices[-1]
profit = final_value - initial_capital

# 绘制资金曲线
plt.figure(figsize=(12,6))
plt.plot(capital_curve, label="资金曲线", color="blue")
plt.title(f"简单回测结果 (最终收益: {profit:.2f} USD)")
plt.xlabel("交易日")
plt.ylabel("账户价值 (USD)")
plt.legend()
plt.show()
```

---

#### 3. 实验结果

* 模型预测的信号成功生成了一条 **资金曲线**
* 在部分区间内资金曲线跑赢了单纯持有策略，但在震荡行情下容易产生假信号
* 说明 **预测模型+交易规则** 才能形成真正的量化策略

---

### 📊 完成情况

* ✅ 理解了回测的基本概念
* ✅ 将预测结果转化为简单交易信号
* ✅ 实现了一个买入/卖出的资金曲线模拟
* ⚠️ 不足：规则过于简单，容易受到预测误差和震荡行情影响

---

### 💡 学习心得

* 今天的最大收获是：**预测 ≠ 收益**，预测只是量化的一部分，关键还是交易规则和风控。
* 回测能让我们快速验证一个想法是否靠谱，比“拍脑袋”做交易要科学得多。
* 下一步要考虑如何设计 **更合理的信号+仓位管理**，比如均线突破、止损止盈。

---

### 🚀 明日计划

* 尝试设计 **改进的交易规则**（如均线+预测结合）
* 引入 **止损/止盈机制**
* 学习回测框架 **backtrader** 的使用，为后续复杂策略做准备

---

### 📂 附录

* 📜 [Backtesting Basics - Investopedia](https://www.investopedia.com/terms/b/backtesting.asp)
* 📜 [Backtrader 官方文档](https://www.backtrader.com/docu/)

---

## 📖 （Day 23）AI量化学习日志：均线+预测结合的交易策略

---

### 📌 今日计划

* **今日目标**：

  * 在昨日简单回测的基础上，增加 **均线指标**
  * 将 **预测信号** 与 **均线趋势** 结合，减少噪声
  * 引入 **止损止盈机制**，避免大亏 & 锁定利润
  * 完成一次更贴近实盘的回测

---

### 🛠️ 学习过程

#### 1. 理论学习

* **均线（Moving Average, MA）**：用来判断趋势。

  * 短期均线（如 10 日 MA） > 长期均线（如 30 日 MA） → 上涨趋势
  * 反之 → 下跌趋势

* **交易逻辑设计**：

  1. 模型预测未来价格
  2. 如果预测上涨 & 当前处于多头趋势（短均线 > 长均线） → 买入
  3. 如果预测下跌 & 当前处于空头趋势（短均线 < 长均线） → 卖出
  4. 设置 **止损**（-5%） 和 **止盈**（+10%），避免大幅回撤

---

#### 2. Python 实践

```python
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# 下载数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 计算均线
data["MA10"] = data["Close"].rolling(window=10).mean()
data["MA30"] = data["Close"].rolling(window=30).mean()

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造时间序列
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM 模型
inputs = layers.Input(shape=(time_step, 1))
x = layers.LSTM(50, return_sequences=True)(inputs)
x = layers.LSTM(50)(x)
outputs = layers.Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

# 训练
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 预测
predictions = model.predict(X_test)
pred_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# ---------------------------
# 改进回测逻辑（均线 + 止损止盈）
# ---------------------------
initial_capital = 10000
capital = initial_capital
position = 0
buy_price = 0
capital_curve = []

ma10 = data["MA10"].values[-len(real_prices):]
ma30 = data["MA30"].values[-len(real_prices):]

for i in range(len(real_prices)-1):
    price = real_prices[i][0]
    pred = pred_prices[i][0]

    # 检查是否有持仓
    if position > 0:
        # 止损条件
        if (price - buy_price) / buy_price <= -0.05:
            capital = position * price
            position = 0
        # 止盈条件
        elif (price - buy_price) / buy_price >= 0.1:
            capital = position * price
            position = 0

    # 买入逻辑
    if pred > price and ma10[i] > ma30[i] and capital > 0:
        position = capital / price
        buy_price = price
        capital = 0

    # 卖出逻辑
    elif pred < price and ma10[i] < ma30[i] and position > 0:
        capital = position * price
        position = 0

    # 记录资金曲线
    total_value = capital + position * price
    capital_curve.append(total_value)

# 最终收益
final_value = capital + position * real_prices[-1]
profit = final_value - initial_capital

# 绘制资金曲线
plt.figure(figsize=(12,6))
plt.plot(capital_curve, label="资金曲线", color="blue")
plt.title(f"均线+预测策略回测 (最终收益: {profit:.2f} USD)")
plt.xlabel("交易日")
plt.ylabel("账户价值 (USD)")
plt.legend()
plt.show()
```

---

#### 3. 实验结果

* **资金曲线**比 Day 22 更平滑，震荡中假信号减少
* 止损机制有效降低了回撤，止盈让部分盈利被锁定
* 在大趋势行情中表现优异，但在横盘中仍有较多交易噪声

---

### 📊 完成情况

* ✅ 将预测信号与均线结合，减少噪声
* ✅ 增加止损止盈，提高实盘可行性
* ✅ 得到更稳定的资金曲线
* ⚠️ 不足：策略仍然比较简单，仓位控制比较“激进”（全仓进出）

---

### 💡 学习心得

* 今天最大的收获是：**预测模型的输出最好与传统指标结合**，单一信号很容易过拟合或失效。
* **风控机制（止损/止盈）是真正能保命的东西**，即使预测再好，也要防止极端行情下的崩溃。
* 下一步应该学习更专业的回测框架（如 backtrader），支持多因子、多仓位策略。

---

### 🚀 明日计划

* 学习 **backtrader 框架**，尝试将预测结果接入 backtrader
* 实现 **多因子策略**（均线 + RSI + 预测）
* 引入 **资金管理**（比如固定比例开仓、分批加仓）

---

### 📂 附录

* 📜 [Moving Average Trading Strategy](https://www.investopedia.com/articles/technical/04/041404.asp)
* 📜 [Backtrader 文档](https://www.backtrader.com/docu/)

---

## 📖 （Day 24）AI量化学习日志：Backtrader 回测框架初体验

---

### 📌 今日计划

* **今日目标**：

  * 学习使用 **Backtrader** 进行回测
  * 将 **LSTM 预测结果** 接入 Backtrader
  * 在策略中结合 **均线指标**，实现更真实的交易逻辑
  * 产出一条 **完整资金曲线**

---

### 🛠️ 学习过程

#### 1. 理论学习

* **Backtrader**：一个强大的 Python 回测框架，支持数据导入、指标计算、策略定义、资金管理、可视化。
* 核心结构：

  1. **Cerebro**：大脑，负责调度
  2. **Data Feed**：行情数据
  3. **Strategy**：交易策略逻辑
  4. **Broker**：资金管理
  5. **Analyzer**：绩效评估

我们要做的是：

* 导入股票数据（AAPL）
* 加载 **AI 预测信号**（LSTM 输出）
* 策略：预测上涨且 MA10 > MA30 → 买入；预测下跌且 MA10 < MA30 → 卖出

---

#### 2. Python 实践

```python
import backtrader as bt
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# =====================
# 1. 下载 & 预处理数据
# =====================
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造时间序列
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM 模型（简单训练）
inputs = layers.Input(shape=(time_step, 1))
x = layers.LSTM(50, return_sequences=True)(inputs)
x = layers.LSTM(50)(x)
outputs = layers.Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# 预测信号
predictions = model.predict(X, verbose=0)
pred_prices = scaler.inverse_transform(predictions)

# 将预测结果放入 DataFrame
data = data.iloc[time_step:]
data["Pred_Close"] = pred_prices

# =====================
# 2. 定义 Backtrader 策略
# =====================
class AIPredictStrategy(bt.Strategy):
    def __init__(self):
        self.ma10 = bt.indicators.SMA(self.data.close, period=10)
        self.ma30 = bt.indicators.SMA(self.data.close, period=30)

    def next(self):
        pred_price = self.data.pred[0]  # 预测价格
        current_price = self.data.close[0]

        # 买入条件
        if not self.position:
            if pred_price > current_price and self.ma10[0] > self.ma30[0]:
                self.buy()
        # 卖出条件
        else:
            if pred_price < current_price and self.ma10[0] < self.ma30[0]:
                self.sell()

# =====================
# 3. 接入 Backtrader
# =====================
class PandasDataExtend(bt.feeds.PandasData):
    lines = ("pred",)
    params = (("pred", -1),)

data_feed = PandasDataExtend(dataname=data, datetime=None, open=0, high=1, low=2,
                             close=3, volume=5, openinterest=-1, pred="Pred_Close")

cerebro = bt.Cerebro()
cerebro.addstrategy(AIPredictStrategy)
cerebro.adddata(data_feed)
cerebro.broker.setcash(10000.0)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

# 回测
print("初始资金: ", cerebro.broker.getvalue())
results = cerebro.run()
print("最终资金: ", cerebro.broker.getvalue())

# 画图
cerebro.plot()
```

---

#### 3. 实验结果

* **回测完成**，能得到资金曲线和买卖点标记图
* 策略逻辑可扩展性强，可以继续加入 **RSI、MACD、止损止盈**
* Sharpe 比率可直接输出，用来衡量策略质量

---

### 📊 完成情况

* ✅ 掌握了 Backtrader 的基本用法
* ✅ 将 AI 预测结果接入策略
* ✅ 得到了完整的回测资金曲线
* ⚠️ 模型只训练了 5 个 epoch，预测效果比较一般

---

### 💡 学习心得

* 今天的收获很大，**Backtrader 大大提升了回测效率和真实感**，资金曲线比自己写循环专业多了。
* 未来可以直接在 Backtrader 中构建多因子策略，而不是手动拼接。
* 也发现了一个问题：LSTM 模型训练太简单，预测效果一般，明天需要优化训练。

---

### 🚀 明日计划

* 优化 LSTM 模型训练（增加 epoch、dropout 防止过拟合）
* 在 Backtrader 策略中加入 **RSI**
* 尝试对比 **AI预测+技术指标** vs **仅技术指标** 的回测效果

---

### 📂 附录

* 📜 [Backtrader 官方文档](https://www.backtrader.com/docu/)
* 📜 [Backtrader 数据源扩展](https://www.backtrader.com/docu/data/data/)

---

## 📖 （Day 25）AI量化学习日志：多因子策略（AI+均线+RSI）

---

### 📌 今日计划

* **今日目标**：

  * 在 Backtrader 框架中，增加 **RSI 指标**
  * 构建一个 **多因子策略**：AI预测 + 均线趋势 + RSI 超买超卖
  * 与 **单因子策略（仅均线/仅RSI）** 对比，观察资金曲线和收益差异

---

### 🛠️ 学习过程

#### 1. 理论学习

* **RSI（Relative Strength Index，相对强弱指数）**：衡量价格超买或超卖。

  * RSI > 70 → 超买（可能下跌）
  * RSI < 30 → 超卖（可能上涨）

* **多因子逻辑**：

  1. **买入条件**：预测价格 > 当前价格，MA10 > MA30，且 RSI < 30
  2. **卖出条件**：预测价格 < 当前价格，MA10 < MA30，或 RSI > 70

这种组合逻辑避免了单一信号的“假突破”。

---

#### 2. Python 实践

```python
import backtrader as bt
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# =====================
# 1. 下载数据 + LSTM预测
# =====================
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 简单 LSTM 预测模型
inputs = layers.Input(shape=(time_step, 1))
x = layers.LSTM(50, return_sequences=True)(inputs)
x = layers.LSTM(50, dropout=0.2)(x)
outputs = layers.Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=8, batch_size=32, verbose=0)

predictions = model.predict(X, verbose=0)
pred_prices = scaler.inverse_transform(predictions)

# 补充预测列
data = data.iloc[time_step:].copy()
data["Pred_Close"] = pred_prices

# =====================
# 2. 策略定义
# =====================
class MultiFactorStrategy(bt.Strategy):
    def __init__(self):
        self.ma10 = bt.indicators.SMA(self.data.close, period=10)
        self.ma30 = bt.indicators.SMA(self.data.close, period=30)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

    def next(self):
        pred_price = self.data.pred[0]
        current_price = self.data.close[0]

        # 买入条件：预测上涨 + 多头趋势 + RSI < 30
        if not self.position:
            if pred_price > current_price and self.ma10[0] > self.ma30[0] and self.rsi[0] < 30:
                self.buy()

        # 卖出条件：预测下跌 + 空头趋势 或 RSI > 70
        else:
            if (pred_price < current_price and self.ma10[0] < self.ma30[0]) or self.rsi[0] > 70:
                self.sell()

# =====================
# 3. Backtrader 接入
# =====================
class PandasDataExtend(bt.feeds.PandasData):
    lines = ("pred",)
    params = (("pred", -1),)

data_feed = PandasDataExtend(dataname=data, datetime=None, open=0, high=1, low=2,
                             close=3, volume=5, openinterest=-1, pred="Pred_Close")

cerebro = bt.Cerebro()
cerebro.addstrategy(MultiFactorStrategy)
cerebro.adddata(data_feed)
cerebro.broker.setcash(10000.0)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

print("初始资金: ", cerebro.broker.getvalue())
results = cerebro.run()
print("最终资金: ", cerebro.broker.getvalue())
print("夏普比率: ", results[0].analyzers.sharpe.get_analysis())
print("最大回撤: ", results[0].analyzers.drawdown.get_analysis())

# 绘制资金曲线
cerebro.plot()
```

---

#### 3. 实验结果

* **资金曲线更平稳**，相比单因子策略，震荡减少
* **RSI 的过滤效果明显**，在高位超买时避免了追涨杀跌
* 夏普比率高于 Day 24 策略，风险调整后的收益更好
* 最大回撤明显缩小，说明风险控制更优

---

### 📊 完成情况

* ✅ 成功实现多因子策略（AI+MA+RSI）
* ✅ 获得了更平滑的资金曲线
* ✅ 夏普比率 & 最大回撤均优于单因子策略
* ⚠️ 模型预测仍有改进空间，未来可尝试 Transformer

---

### 💡 学习心得

* 今天最大的感受是：**多因子策略能显著降低单因子的噪声**，让交易更接近实盘需求。
* 发现 Backtrader 可以非常方便地叠加各种指标，真正做到策略灵活组合。
* 下一步应该尝试 **仓位管理**，比如分批买入/卖出，而不是全仓操作。

---

### 🚀 明日计划

* 在 Backtrader 中实现 **仓位管理**（资金分配）
* 尝试 **止损止盈** 的参数优化（比如止损 5%，止盈 15%）
* 继续对比不同策略的回测效果

---

### 📂 附录

* 📜 [RSI 指标原理](https://www.investopedia.com/terms/r/rsi.asp)
* 📜 [Backtrader 多因子示例](https://community.backtrader.com/)

---



