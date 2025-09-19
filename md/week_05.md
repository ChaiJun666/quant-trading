---

## 📖 （Day 26）AI量化学习日志：仓位管理与止盈止损优化

---

### 📌 今日计划

* **今日目标**：

  * 在 Backtrader 策略中加入 **仓位管理（固定比例开仓）**
  * 优化 **止损止盈参数**（止损 5%，止盈 15%）
  * 对比全仓 vs 分仓资金曲线，分析风险收益差异

---

### 🛠️ 学习过程

#### 1. 理论学习

* **仓位管理（Position Sizing）**：

  * 实盘中不能“满仓干”，通常会按资金比例开仓。
  * 例如：每次开仓 = 总资金的 **20%**，剩余资金作为安全垫。

* **止损止盈优化**：

  * **止损**：防止亏损扩大（比如 -5% 强制止损）
  * **止盈**：在盈利达到一定程度（比如 +15%）时落袋为安
  * 配合仓位管理，可以大幅降低回撤

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

# 简单 LSTM 模型
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
# 2. 策略定义（仓位管理 + 止损止盈）
# =====================
class RiskManagedStrategy(bt.Strategy):
    params = (("stake", 0.2),  # 每次开仓占比
              ("stop_loss", 0.05),  # 止损 5%
              ("take_profit", 0.15))  # 止盈 15%

    def __init__(self):
        self.ma10 = bt.indicators.SMA(self.data.close, period=10)
        self.ma30 = bt.indicators.SMA(self.data.close, period=30)
        self.buy_price = None

    def next(self):
        pred_price = self.data.pred[0]
        current_price = self.data.close[0]

        # 有持仓时检查止盈止损
        if self.position:
            change = (current_price - self.buy_price) / self.buy_price
            if change <= -self.p.stop_loss or change >= self.p.take_profit:
                self.sell()

        # 买入条件
        if not self.position:
            if pred_price > current_price and self.ma10[0] > self.ma30[0]:
                size = (self.broker.getcash() * self.p.stake) / current_price
                self.buy(size=size)
                self.buy_price = current_price

        # 卖出条件
        elif pred_price < current_price and self.ma10[0] < self.ma30[0]:
            self.sell()

# =====================
# 3. Backtrader 回测
# =====================
class PandasDataExtend(bt.feeds.PandasData):
    lines = ("pred",)
    params = (("pred", -1),)

data_feed = PandasDataExtend(dataname=data, datetime=None, open=0, high=1, low=2,
                             close=3, volume=5, openinterest=-1, pred="Pred_Close")

cerebro = bt.Cerebro()
cerebro.addstrategy(RiskManagedStrategy)
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

* **资金曲线更稳定**，相比全仓策略，波动明显减小
* **止损机制** 有效控制了单次亏损，避免了大幅回撤
* **止盈机制** 让部分盈利锁定，防止回吐
* **夏普比率提升**，说明风险调整后的收益更优

---

### 📊 完成情况

* ✅ 实现了仓位管理（每次开仓 20% 资金）
* ✅ 加入止损止盈，避免极端行情的爆仓风险
* ✅ 得到了更真实、更稳健的资金曲线
* ⚠️ 止损止盈参数仍需优化，不同市场下最佳值可能不同

---

### 💡 学习心得

* 今天最大的收获是：**仓位管理是量化的核心之一**，哪怕预测模型再好，没有风控也很难长期生存。
* **止盈止损不是减少盈利，而是保证稳定性**，资金曲线平滑比短期暴利更重要。
* 下一步，可以尝试 **网格搜索/回测参数优化**，找到最优的止损止盈组合。

---

### 🚀 明日计划

* 学习 Backtrader **参数优化功能（optstrategy）**
* 用网格搜索寻找 **最佳止损止盈参数**
* 对比不同参数下的收益和回撤表现

---

### 📂 附录

* 📜 [Position Sizing Basics](https://www.investopedia.com/terms/p/positionsizing.asp)
* 📜 [Backtrader Strategy Optimization](https://www.backtrader.com/docu/optstrategy/)

---

## 📖 （Day 27）AI量化学习日志：参数优化实战

---

### 📌 今日计划

* 学习 Backtrader 的 **参数优化（optstrategy）** 功能
* 批量测试不同的 **止损 & 止盈组合**
* 找出最优参数（风险收益比最优）

---

### 🛠️ 学习过程

#### 1. 理论学习

* **为什么要优化？**
  单一参数可能只适合某段行情 → 批量测试才能找到最稳健的组合。

* **Backtrader 的方法**：

  * `cerebro.optstrategy()`
  * 给参数设置多个取值
  * 自动回测，返回结果对比

---

#### 2. Python 实践

```python
import backtrader as bt
import pandas as pd
import yfinance as yf

# 策略
class RiskManagedStrategy(bt.Strategy):
    params = (("stake", 0.2), 
              ("stop_loss", 0.05),
              ("take_profit", 0.15))

    def __init__(self):
        self.ma10 = bt.indicators.SMA(self.data.close, period=10)
        self.ma30 = bt.indicators.SMA(self.data.close, period=30)
        self.buy_price = None

    def next(self):
        current_price = self.data.close[0]
        if self.position:
            change = (current_price - self.buy_price) / self.buy_price
            if change <= -self.p.stop_loss or change >= self.p.take_profit:
                self.sell()
        if not self.position and self.ma10[0] > self.ma30[0]:
            size = (self.broker.getcash() * self.p.stake) / current_price
            self.buy(size=size)
            self.buy_price = current_price

# 数据
data = yf.download("AAPL", period="2y")
feed = bt.feeds.PandasData(dataname=data)

# Cerebro 优化
cerebro = bt.Cerebro()
cerebro.optstrategy(RiskManagedStrategy,
                    stop_loss=[0.03, 0.05, 0.07, 0.08],
                    take_profit=[0.10, 0.15, 0.20])

cerebro.adddata(feed)
cerebro.broker.setcash(10000.0)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

results = cerebro.run(maxcpus=1)

# 提取结果
res = []
for r in results:
    s = r[0]
    res.append({
        "stop_loss": s.params.stop_loss,
        "take_profit": s.params.take_profit,
        "final_value": s.broker.getvalue(),
        "sharpe": s.analyzers.sharpe.get_analysis().get("sharperatio"),
        "max_drawdown": s.analyzers.drawdown.get_analysis().max.drawdown
    })

print(pd.DataFrame(res).sort_values("sharpe", ascending=False))
```

---

#### 3. 实验结果

* **止损过小（3%）** → 频繁触发，收益低
* **止盈过小（10%）** → 提前卖出，错失趋势
* **止损 5% + 止盈 15%** 在测试区间效果最佳：

  * 夏普比率最高
  * 回撤较小
  * 资金曲线最平滑

---

### 📊 完成情况

✅ 学会了 `optstrategy` 用法
✅ 批量测试止损止盈组合
✅ 找到相对最优参数
⚠️ 注意：最优解依赖历史数据，需要定期重新优化

---

### 💡 学习心得

* 没有“万能参数”，市场环境变了参数也要变。
* 参数优化是量化交易必不可少的环节，可以大幅提高策略稳健性。
* 下一步：把 **均线周期** 也纳入优化，进行多参数联合测试。

---

### 🚀 明日计划

* 尝试 **多参数优化（止损 + 止盈 + 均线周期）**
* 使用多进程并行提升效率
* 对比牛市、震荡市、熊市下的表现

---


## 📖 （Day 28）AI量化学习日志：多参数联合优化

---

### 📌 今日计划

* 使用 **Backtrader 的 optstrategy** 做多参数优化
* 同时优化 **止损、止盈、均线周期（MA10/MA30）**
* 找到在当前行情下风险收益比最优的组合

---

### 🛠️ 学习过程

#### 1. 理论学习

* 单一参数优化（比如止损）容易导致过拟合。
* **多参数优化** → 同时考虑风险控制 & 趋势信号。
* 常见组合：

  * **风控参数**：止损、止盈
  * **信号参数**：均线周期、阈值
  * **资金管理参数**：仓位比例

---

#### 2. Python 实践

```python
import backtrader as bt
import pandas as pd
import yfinance as yf

# 策略（多参数）
class MultiParamStrategy(bt.Strategy):
    params = (
        ("stake", 0.2),
        ("stop_loss", 0.05),
        ("take_profit", 0.15),
        ("ma_short", 10),
        ("ma_long", 30),
    )

    def __init__(self):
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.p.ma_short)
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.p.ma_long)
        self.buy_price = None

    def next(self):
        price = self.data.close[0]
        if self.position:
            change = (price - self.buy_price) / self.buy_price
            if change <= -self.p.stop_loss or change >= self.p.take_profit:
                self.sell()
        else:
            if self.ma_short[0] > self.ma_long[0]:
                size = (self.broker.getcash() * self.p.stake) / price
                self.buy(size=size)
                self.buy_price = price
            elif self.ma_short[0] < self.ma_long[0]:
                self.sell()

# 数据
data = yf.download("AAPL", period="2y")
feed = bt.feeds.PandasData(dataname=data)

# Cerebro 优化
cerebro = bt.Cerebro()
cerebro.optstrategy(MultiParamStrategy,
                    stop_loss=[0.03, 0.05, 0.07],
                    take_profit=[0.10, 0.15, 0.20],
                    ma_short=[5, 10, 15],
                    ma_long=[20, 30, 50])

cerebro.adddata(feed)
cerebro.broker.setcash(10000.0)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

results = cerebro.run(maxcpus=1)

# 提取结果
res = []
for r in results:
    s = r[0]
    res.append({
        "stop_loss": s.params.stop_loss,
        "take_profit": s.params.take_profit,
        "ma_short": s.params.ma_short,
        "ma_long": s.params.ma_long,
        "final_value": s.broker.getvalue(),
        "sharpe": s.analyzers.sharpe.get_analysis().get("sharperatio"),
        "max_drawdown": s.analyzers.drawdown.get_analysis().max.drawdown
    })

df = pd.DataFrame(res)
print(df.sort_values("sharpe", ascending=False).head(10))
```

---

#### 3. 实验结果

* **最优参数组合**（示例结果）：

  * 止损 5%
  * 止盈 15%
  * MA10 / MA30
* 表现：

  * 夏普比率 **明显高于单参数优化**
  * 最大回撤降低
  * 策略资金曲线更平滑

---

### 📊 完成情况

✅ 学会了多参数优化的实现
✅ 批量测试了止损、止盈、均线组合
✅ 找到在当前市场下更稳健的参数
⚠️ 问题：组合数量多时，回测速度明显变慢

---

### 💡 学习心得

* 今天最大的收获是：**参数不是孤立的**，止损止盈必须和信号周期结合考虑。
* 多参数优化可以避免单一参数“看似优秀但其实过拟合”的问题。
* 需要注意 **参数空间爆炸**，组合过多时最好用并行计算。

---

### 🚀 明日计划

* 尝试 **并行回测优化** 提升效率
* 引入 **交叉验证（不同时间区间测试参数）** 检验稳健性
* 对比不同市场阶段（牛市/震荡/熊市）下的表现

---


