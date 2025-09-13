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


## day02
## day03
## day04
## day05
## day06
## day07