---

# 第一周

## 📖 （Day 1）AI量化学习日志：量化交易入门与收益率计算

---

### 📌 今日计划

* **今日目标**：了解量化交易的基本概念，并用 Python 实现股票的日收益率计算。
* **学习内容**：

  1. 量化交易的定义与优势
  2. 收益率的基本概念（简单收益率、对数收益率）
  3. 用 Python 对股票历史价格计算日收益率

---

### 🛠️ 学习过程

#### 1. 理论学习

* **量化交易（Quantitative Trading）**：通过数学模型和计算机程序进行投资决策。核心优势是避免情绪化决策、能快速处理大量数据。
* **收益率**：衡量投资收益的关键指标。

  * 简单收益率公式：

    $$
    R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
    $$
  * 对数收益率公式：

    $$
    r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
    $$

    对数收益率更适合连续复利计算，金融领域常用。

---

#### 2. Python 实践

我选用了 `yfinance` 获取股票数据（比如苹果 AAPL），并计算了日收益率：

```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 下载苹果公司近 1 年的历史数据
data = yf.download("AAPL", period="1y")

# 计算简单收益率和对数收益率
data['Simple Return'] = data['Adj Close'].pct_change()
data['Log Return'] = (data['Adj Close'] / data['Adj Close'].shift(1)).apply(lambda x: pd.np.log(x))

# 绘制收益率曲线
plt.figure(figsize=(10,5))
data['Log Return'].plot(title="AAPL 日对数收益率")
plt.show()
```

运行结果：

* 成功获取了苹果过去 1 年的股价数据。
* 绘制出了日对数收益率的曲线，可以直观看到波动情况。

---

### 📊 完成情况

* ✅ 学习了量化交易的基本概念
* ✅ 掌握了收益率计算公式
* ✅ 用 Python 成功获取股票数据并绘制了收益率图表
* ⚠️ 小问题：`pd.np.log` 已经弃用，后续需要改成 `import numpy as np` 再用 `np.log`

---

### 💡 学习心得

* 今天算是打开了 AI 量化的第一扇门，之前只是听说“量化交易”，但没想过其实可以很快动手写出一个收益率分析。
* 最大收获是理解了“对数收益率”的重要性，它在后续建模中会更常用。
* 遇到的坑是 pandas 的旧写法报了警告，不过这也让我意识到要时刻注意库的更新。

---

### 🚀 明日计划

* 学习如何获取更多股票的历史数据
* 初步绘制 **K 线图** 和 **移动平均线（MA）**
* 为后续策略开发打基础

---

### 📂 附录

* 📜 [量化投资基础知识 - Investopedia](https://www.investopedia.com/terms/q/quantitativeanalysis.asp)
* 📜 [对数收益率与简单收益率区别](https://quant.stackexchange.com/questions/179/why-use-log-returns)


---

## 📖（Day 2） AI量化学习日志：获取股票历史数据与可视化

---

### 📌 今日计划

* **今日目标**：学会获取多只股票的历史数据，并用可视化方法进行初步分析。
* **学习内容**：

  1. 使用 `yfinance` 或 `akshare` 获取股票历史数据
  2. 了解股票常见指标（开盘价、收盘价、最高价、最低价、成交量）
  3. 绘制股票价格曲线

---

### 🛠️ 学习过程

#### 1. 理论学习

* 股票历史数据包含以下主要字段：

  * **Open**（开盘价）
  * **Close**（收盘价）
  * **High**（最高价）
  * **Low**（最低价）
  * **Volume**（成交量）
* 这些数据是构建量化模型的基础，后续指标（如均线、波动率、夏普比率等）都依赖这些原始数据。

---

#### 2. Python 实践

今天继续用 `yfinance`，获取苹果（AAPL）和微软（MSFT）两只股票过去一年的历史数据，并画出收盘价曲线对比。

```python
import yfinance as yf
import matplotlib.pyplot as plt

# 下载苹果和微软的历史数据（1年）
stocks = ["AAPL", "MSFT"]
data = yf.download(stocks, period="1y")['Adj Close']

# 绘制对比收盘价曲线
plt.figure(figsize=(12,6))
for stock in stocks:
    data[stock].plot(label=stock)

plt.title("AAPL vs MSFT 收盘价对比 (过去1年)")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.legend()
plt.show()
```

运行结果：

* 成功获取到苹果和微软的收盘价历史数据。
* 可视化结果显示，两只股票走势有一定的相关性，但波动幅度不同。

---

### 📊 完成情况

* ✅ 学会了获取多只股票的历史数据
* ✅ 掌握了股票数据的主要字段
* ✅ 绘制了收盘价曲线，直观对比了不同股票的走势

---

### 💡 学习心得

* 今天第一次画多只股票对比图，直观感受到 **数据可视化的重要性**。
* 之前只是看K线图，但用 Python 可以灵活地对比不同股票，更方便做策略分析。
* 小坑：`yfinance` 获取的数据有时延迟，需要注意数据时区问题。

---

### 🚀 明日计划

* 学习绘制 **K线图（Candlestick Chart）**
* 添加 **移动平均线（MA）** 指标，初步感受技术分析

---

### 📂 附录

* 📜 [yfinance 文档](https://pypi.org/project/yfinance/)
* 📜 [Matplotlib 绘图基础](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)

---


## 📖 （Day 3）AI量化学习日志：K线图与均线初探

---

### 📌 今日计划

* **今日目标**：掌握股票 K 线图的绘制方法，并在图中叠加移动平均线（MA），初步体会趋势分析。
* **学习内容**：

  1. K 线图（Candlestick Chart）的结构与意义
  2. 移动平均线（MA）的计算方法
  3. 用 Python 绘制 K 线图并叠加 MA

---

### 🛠️ 学习过程

#### 1. 理论学习

* **K 线图**

  * 又叫蜡烛图，由 **开盘价（Open）、收盘价（Close）、最高价（High）、最低价（Low）** 四个数据构成。
  * 红色蜡烛代表上涨（收盘 > 开盘），绿色蜡烛代表下跌（收盘 < 开盘）。
* **移动平均线（MA）**

  * **简单移动平均线（SMA）**：

    $$
    MA_n = \frac{P_{t} + P_{t-1} + \dots + P_{t-n+1}}{n}
    $$
  * 常见用法：MA5（5日均线）、MA20（20日均线）用于判断趋势。

---

#### 2. Python 实践

今天用 `mplfinance` 库绘制苹果公司（AAPL）过去3个月的K线图，并叠加 5 日、20 日均线。

```python
import yfinance as yf
import mplfinance as mpf

# 下载苹果公司近3个月的股票数据
data = yf.download("AAPL", period="3mo")

# 添加均线
ma_days = [5, 20]

# 绘制K线图 + 均线
mpf.plot(data, type='candle', mav=ma_days, volume=True, 
         title="AAPL K线图 + MA5 & MA20",
         style='yahoo')
```

运行结果：

* 绘制出标准的K线图，每根蜡烛代表一天走势。
* 图中有 **MA5（短期均线）** 和 **MA20（中期均线）**，交叉点可以作为趋势信号参考。

---

### 📊 完成情况

* ✅ 掌握了K线图的基本原理
* ✅ 成功绘制了股票K线图
* ✅ 在图中叠加MA5、MA20，并观察趋势变化

---

### 💡 学习心得

* K线图比单纯的收盘价曲线更直观，可以清楚地看到每天的波动范围。
* 短期均线与中长期均线的交叉点，常被用来判断买入/卖出信号（比如“黄金交叉”、“死亡交叉”）。
* 今天第一次用 `mplfinance`，感觉比 `matplotlib` 更适合金融可视化，后续会继续用它绘制更多图表。

---

### 🚀 明日计划

* 学习计算 **技术指标**（如 RSI、MACD）
* 尝试用 Python 绘制 RSI 指标，并分析超买/超卖信号

---

### 📂 附录

* 📜 [K线图基础 - Investopedia](https://www.investopedia.com/terms/c/candlestick.asp)
* 📜 [mplfinance 官方文档](https://github.com/matplotlib/mplfinance)

---

## 📖（Day 4）AI量化学习日志：RSI 指标计算与可视化

---

### 📌 今日计划

* **今日目标**：掌握 RSI（相对强弱指数）的计算方法，并用 Python 实现可视化。
* **学习内容**：

  1. RSI 指标的原理与公式
  2. Python 实现 RSI 计算
  3. 可视化 RSI，并观察超买/超卖信号

---

### 🛠️ 学习过程

#### 1. 理论学习

* **RSI（Relative Strength Index）**

  * 由 Welles Wilder 提出，用来衡量股票价格的超买和超卖情况。
  * 计算步骤：

    1. 计算每一天的涨跌幅
    2. 分别计算平均上涨（Avg Gain）和平均下跌（Avg Loss）
    3. 计算相对强度 RS = Avg Gain / Avg Loss
    4. 计算 RSI：

       $$
       RSI = 100 - \frac{100}{1 + RS}
       $$
* **解读方式**：

  * RSI > 70 → 可能超买（价格偏高，可能回调）
  * RSI < 30 → 可能超卖（价格偏低，可能反弹）

---

#### 2. Python 实践

今天继续用苹果公司（AAPL）的数据，计算并绘制 RSI 指标：

```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 下载苹果公司近6个月数据
data = yf.download("AAPL", period="6mo")

# 计算每日涨跌
delta = data['Adj Close'].diff()

# 分别计算涨跌
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# 设置RSI周期（常用14天）
window = 14
avg_gain = gain.rolling(window=window, min_periods=window).mean()
avg_loss = loss.rolling(window=window, min_periods=window).mean()

# 计算RSI
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
data['RSI'] = rsi

# 绘制RSI曲线
plt.figure(figsize=(12,6))
plt.plot(data.index, data['RSI'], label="RSI (14)", color="purple")
plt.axhline(70, color="red", linestyle="--")  # 超买线
plt.axhline(30, color="green", linestyle="--")  # 超卖线
plt.title("AAPL RSI 指标 (14日)")
plt.legend()
plt.show()
```

运行结果：

* 成功绘制 RSI 曲线，明显可以看到当 RSI 高于 70 时，股价往往处于局部高点；当 RSI 低于 30 时，股价往往处于低点。

---

### 📊 完成情况

* ✅ 掌握了 RSI 的计算公式与意义
* ✅ 用 Python 成功实现了 RSI 指标
* ✅ 可视化了 RSI 曲线，并标注了超买/超卖区间

---

### 💡 学习心得

* RSI 是第一个接触的 **量化技术指标**，它不像均线那样滞后，能够提供相对直观的买卖信号。
* 但单独使用 RSI 会有很多“假信号”，比如强势上涨时 RSI 长时间维持在 70 以上，依然可能继续涨。
* 感觉 RSI 更适合与其他指标结合，比如 **均线、MACD**。

---

### 🚀 明日计划

* 学习 **MACD 指标** 的计算与绘制
* 尝试将 MACD 与 RSI 结合，分析买卖信号

---

### 📂 附录

* 📜 [RSI 指标解释 - Investopedia](https://www.investopedia.com/terms/r/rsi.asp)
* 📜 [Python 金融时间序列分析](https://pandas.pydata.org/)


---

## 📖（Day 5）AI量化学习日志

**主题：量化投资常见策略初识**

---

### ✅ 今日目标

* 了解常见的量化投资策略类型。
* 学习这些策略的基本逻辑与优缺点。
* 为后续实践找到适合自己实现的切入点。

---

### 📌 学习内容

今天主要学习了几类经典量化投资策略，每一类都有对应的投资逻辑和应用场景：

1. **趋势跟随策略（Trend Following）**

   * 核心思想：顺势而为，买涨不买跌。
   * 常用方法：均线交叉（MA Cross）、布林带突破等。
   * 优点：容易理解，适合波动较大的市场。
   * 缺点：容易出现假突破，信号滞后。

2. **均值回归策略（Mean Reversion）**

   * 核心思想：价格长期会回归均值。
   * 常用方法：布林带回归、RSI超买超卖。
   * 优点：适合震荡行情，胜率较高。
   * 缺点：在单边趋势中容易连续亏损。

3. **多因子选股（Multi-Factor Models）**

   * 核心思想：结合价值因子、动量因子、成长因子等指标挑选股票。
   * 常用因子：PE、PB、ROE、动量指标。
   * 优点：理论基础扎实，能长期跑赢市场。
   * 缺点：实现复杂，因子有效性随市场变化。

4. **套利策略（Arbitrage）**

   * 核心思想：利用市场错误定价赚取无风险收益。
   * 常见形式：ETF套利、期现套利、跨市场套利。
   * 优点：理论上低风险。
   * 缺点：执行门槛高，对资金和交易速度要求大。

---

### 💻 实战小练习：均线交叉策略（Python示例）

```python
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 下载苹果公司股票数据
data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")

# 计算短期均线和长期均线
data["MA10"] = data["Close"].rolling(window=10).mean()
data["MA30"] = data["Close"].rolling(window=30).mean()

# 生成交易信号
data["Signal"] = 0
data.loc[data["MA10"] > data["MA30"], "Signal"] = 1  # 买入
data.loc[data["MA10"] < data["MA30"], "Signal"] = -1 # 卖出

# 可视化
plt.figure(figsize=(12,6))
plt.plot(data["Close"], label="Close Price")
plt.plot(data["MA10"], label="MA10")
plt.plot(data["MA30"], label="MA30")
plt.legend()
plt.title("均线交叉策略示例")
plt.show()
```

🔎 逻辑很简单：

* 当 **短期均线突破长期均线** → 买入信号
* 当 **短期均线跌破长期均线** → 卖出信号

这就是一个最常见的趋势跟随策略。

---

### 📝 今日总结

通过学习不同的策略，我发现：

* **趋势跟随** 和 **均值回归** 更适合作为入门的实战项目（逻辑直观，代码实现也比较容易）。
* **多因子选股** 和 **套利策略** 更适合后期深入，需要更多数据和交易工具支持。

👉 我的下一步计划是：先选择 **股价预测 + 趋势跟随策略** 作为实战项目的方向。

---

### 🎯 明日计划

* 学习股价预测常用的 **机器学习方法（回归模型）**。
* 尝试用 `scikit-learn` 训练一个简单的线性回归预测股价。


---

## 📖 （Day 6）AI量化学习日志：线性回归预测股价

---

### 📌 今日计划

* **今日目标**：学习股价预测中的线性回归方法，并用 `scikit-learn` 实现一个简单模型。
* **学习内容**：

  1. 线性回归的基本原理
  2. 特征工程：用前几天的价格预测下一天价格
  3. Python 实现股价预测

---

### 🛠️ 学习过程

#### 1. 理论学习

* **线性回归（Linear Regression）**

  * 数学形式：

    $$
    y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon
    $$
  * 在股价预测中，\$x\_i\$ 可以是过去几天的收盘价，\$y\$ 则是下一天的价格。

* **优点**：简单直观、计算速度快、容易上手。

* **缺点**：无法捕捉股价的非线性特征，预测效果有限。

---

#### 2. Python 实践

我用 `yfinance` 下载苹果公司（AAPL）的股价数据，并用过去 5 天的收盘价预测第 6 天的收盘价：

```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 下载苹果公司近 1 年的历史数据
data = yf.download("AAPL", period="1y")

# 构造特征：用前 5 天价格预测下一天
window = 5
X, y = [], []
for i in range(len(data) - window):
    X.append(data["Close"].iloc[i:i+window].values)
    y.append(data["Close"].iloc[i+window])
X, y = np.array(X), np.array(y)

# 划分训练集和测试集
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
print("均方误差 MSE:", mse)

# 可视化预测结果
plt.figure(figsize=(12,6))
plt.plot(y_test, label="真实价格")
plt.plot(y_pred, label="预测价格")
plt.legend()
plt.title("线性回归预测 AAPL 股价")
plt.show()
```

运行结果：

* 成功训练了一个线性回归模型。
* 均方误差（MSE）在几美元的范围，说明模型能大致拟合趋势。
* 预测曲线和真实曲线有一定重合，但在波动较大时效果不佳。

---

### 📊 完成情况

* ✅ 学习了线性回归的基本原理
* ✅ 构造了基于历史股价的特征数据
* ✅ 用 `scikit-learn` 成功训练并预测股价
* ⚠️ 问题：线性模型过于简单，在剧烈波动的行情下预测效果不理想

---

### 💡 学习心得

* 今天第一次真正“预测”股价，虽然效果一般，但感觉很有成就感。
* 线性回归很适合作为入门，关键是理解“用历史数据预测未来”的建模思路。
* 我意识到未来需要引入更多 **特征**（如成交量、技术指标）以及 **非线性模型**（如随机森林、神经网络）来提升预测效果。

---

### 🚀 明日计划

* 学习 **移动平均（MA）与特征工程结合**，增强预测模型。
* 尝试在模型中引入 **技术指标**（如 RSI、MACD）。
* 为后续机器学习模型打下基础。

---

### 📂 附录

* 📜 [Linear Regression in Machine Learning - Scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html)
* 📜 [股价预测与回归模型基础](https://towardsdatascience.com/predicting-stock-prices-with-linear-regression-machine-learning-44f0f0c04a1a)

---

## 📖 （Day 7）AI量化学习日志：移动平均线与特征工程

---

### 📌 今日计划

* **今日目标**：在股价预测中引入 **移动平均线（MA）** 作为特征，提升模型表现。
* **学习内容**：

  1. 移动平均线的概念与作用
  2. 如何用 MA 作为特征输入回归模型
  3. Python 实现「收盘价 + MA 特征」的预测

---

### 🛠️ 学习过程

#### 1. 理论学习

* **移动平均线（Moving Average, MA）**

  * 定义：一定周期内收盘价的平均值。
  * 常见类型：

    * **MA5 / MA10**：短期趋势
    * **MA20 / MA30**：中期趋势
    * **MA60**：长期趋势
  * 作用：平滑价格曲线，帮助捕捉趋势。

* 在预测建模中，MA 能作为一种“趋势特征”，帮助模型识别股价走势。

---

#### 2. Python 实践

在昨天的线性回归模型基础上，我加入了 **5 日均线（MA5）和 10 日均线（MA10）** 作为特征：

```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 下载苹果公司近 1 年的历史数据
data = yf.download("AAPL", period="1y")

# 构造移动平均线
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()

# 丢掉 NaN
data = data.dropna()

# 特征 = 收盘价 + MA5 + MA10
X = data[["Close", "MA5", "MA10"]].values
y = data["Close"].shift(-1).dropna().values  # 预测下一天收盘价
X = X[:-1]  # 对齐 y

# 划分训练集和测试集
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 建模
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
print("均方误差 MSE:", mse)

# 可视化
plt.figure(figsize=(12,6))
plt.plot(y_test, label="真实价格")
plt.plot(y_pred, label="预测价格")
plt.legend()
plt.title("线性回归 + 移动平均线预测 AAPL 股价")
plt.show()
```

运行结果：

* 模型加入 MA 特征后，预测曲线比 Day6 更贴近真实走势。
* 均方误差（MSE）下降，说明预测效果有所提升。

---

### 📊 完成情况

* ✅ 学习了移动平均线的基本原理
* ✅ 将 MA 作为新特征加入回归模型
* ✅ 模型预测效果有所改善
* ⚠️ 问题：仍然存在对剧烈波动拟合不足的情况，说明线性回归依旧偏简单

---

### 💡 学习心得

* 今天感受到 **特征工程的重要性**：

  * 同样的模型，换不同特征，效果差别很大。
* MA 作为技术指标很经典，加入模型后预测曲线确实更加稳定。
* 也意识到单一的线性回归模型存在瓶颈，后续需要尝试更复杂的机器学习方法（如随机森林、XGBoost、LSTM）。

---

### 🚀 明日计划

* 学习 **更多技术指标（RSI、MACD）** 的计算方法。
* 将这些指标加入特征，尝试进一步提升预测效果。

---

### 📂 附录

* 📜 [移动平均线（MA）详解](https://www.investopedia.com/terms/m/movingaverage.asp)
* 📜 [sklearn Linear Regression 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

---

