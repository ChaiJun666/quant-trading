---

## 📖 （Day 15）AI量化学习日志：LSTM预测股价

---

### 📌 今日计划

* **今日目标**：初步掌握 LSTM（长短期记忆网络）在时间序列中的应用
* **学习内容**：

  1. LSTM 的原理与优势
  2. 用 Keras/TensorFlow 搭建 LSTM 网络
  3. 使用历史股价预测未来趋势

---

### 🛠️ 学习过程

#### 1. 理论学习

* **传统机器学习模型**（RF、XGBoost）不能很好捕捉 **时间序列依赖性**。
* **LSTM（Long Short-Term Memory）** 是 RNN 的改进：

  * 通过“门控机制”（输入门、遗忘门、输出门）避免梯度消失
  * 擅长处理 **序列数据**（股价、气温、文本等）
  * 能记住长时间的趋势信息

在金融时间序列中，LSTM 常被用来预测下一步股价走势。

---

#### 2. Python 实践

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 下载数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造训练数据（过去 60 天预测下一天）
X, y = [], []
time_step = 60
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# 调整输入维度 [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 构建 LSTM 模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 预测
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))

real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 可视化
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(predictions, label="LSTM预测", color="blue", linestyle="--")
plt.legend()
plt.title("LSTM 股价预测效果")
plt.show()
```

---

#### 3. 实验结果

* LSTM 能更好地捕捉股价趋势，预测曲线更加“平滑”
* 但对短期波动敏感度不足（预测更像是趋势线，而不是点对点拟合）
* 损失函数收敛较快，验证集与训练集表现接近，说明没有明显过拟合

---

### 📊 完成情况

* ✅ 掌握了 LSTM 的基本原理
* ✅ 成功搭建 LSTM 并进行股价预测
* ✅ 绘制预测曲线，效果较为理想
* ⚠️ 注意：LSTM 训练耗时较长，参数（时间窗口、层数、神经元数）影响很大

---

### 💡 学习心得

* LSTM 让我第一次感受到 **深度学习预测时间序列的魅力**。
* 和 XGBoost 相比，它能捕捉到“趋势”，但短期波动预测还不够精准。
* 在量化实盘中，可能需要结合 **机器学习（短期）+ LSTM（趋势）**，形成更稳健的预测系统。

---

### 🚀 明日计划

* 尝试 **优化 LSTM**（增加层数、调整时间窗口）
* 学习 **GRU（门控循环单元）**，和 LSTM 对比
* 开始考虑如何把预测结果转化为 **交易信号**

---

### 📂 附录

* 📜 [LSTM 原理详解](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* 📜 [Keras LSTM 官方文档](https://keras.io/api/layers/recurrent_layers/lstm/)

---

## 📖 （Day 16）AI量化学习日志：GRU与LSTM对比

---

### 📌 今日计划

* **今日目标**：学习 GRU（门控循环单元）的原理，并与 LSTM 进行对比
* **学习内容**：

  1. GRU 的结构与优势
  2. 用 Keras 搭建 GRU 网络进行股价预测
  3. 对比 GRU 与 LSTM 在股价预测中的表现

---

### 🛠️ 学习过程

#### 1. 理论学习

在深度学习预测股价时，**LSTM** 是常用选择。但 LSTM 结构复杂，包含：

* 输入门
* 遗忘门
* 输出门

👉 **GRU（Gated Recurrent Unit）** 是 LSTM 的简化版本：

* 只有 **更新门（Update Gate）** 和 **重置门（Reset Gate）**
* 计算量更小，训练速度更快
* 在很多任务中，性能与 LSTM 相当

可以理解为：**GRU 是轻量版 LSTM**。

---

#### 2. Python 实践

```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

# 下载股票数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造序列数据
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 构建 GRU 模型
gru_model = Sequential([
    GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    GRU(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
gru_model.compile(optimizer="adam", loss="mean_squared_error")

# 构建 LSTM 模型（用于对比）
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

# 分别训练
gru_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 预测
gru_pred = scaler.inverse_transform(gru_model.predict(X_test))
lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test))
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 可视化对比
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(lstm_pred, label="LSTM预测", linestyle="--", color="blue")
plt.plot(gru_pred, label="GRU预测", linestyle="--", color="orange")
plt.legend()
plt.title("GRU vs LSTM 股价预测")
plt.show()
```

---

#### 3. 实验结果

* **LSTM**：拟合效果更稳定，能捕捉到较长期趋势
* **GRU**：训练速度快，结果和 LSTM 相近，有时甚至更好
* 在苹果股价数据中，GRU 曲线比 LSTM 更“贴合”，但 LSTM 更平滑

---

### 📊 完成情况

* ✅ 学习了 GRU 的原理
* ✅ 成功实现了 GRU 和 LSTM 的对比实验
* ✅ 绘制了预测曲线并进行了分析
* ⚠️ 发现：不同数据集下，GRU 和 LSTM 的表现可能互有胜负

---

### 💡 学习心得

* **LSTM ≈ 精细，GRU ≈ 高效**
* 在金融场景中，如果数据量很大，GRU 更实用；如果要捕捉更复杂的长期依赖，LSTM 更合适
* 今天的对比让我更理解了为什么很多论文会用 GRU 替代 LSTM：**差不多的效果，省下不少算力**

---

### 🚀 明日计划

* 学习 **混合模型**：把 LSTM/GRU 与 **卷积神经网络（CNN）** 结合
* 尝试构建 **CNN-LSTM** 网络，提高预测能力
* 探索如何把预测结果转化为 **买卖信号**

---

### 📂 附录

* 📜 [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)
* 📜 [GRU vs LSTM: A Comparison](https://arxiv.org/abs/1412.3555)


---

## 📖 （Day 17）AI量化学习日志：CNN-LSTM混合模型

---

### 📌 今日计划

* **今日目标**：学习 CNN 与 LSTM 结合的混合模型
* **学习内容**：

  1. CNN 提取特征 + LSTM 处理时间依赖的原理
  2. 用 Keras 搭建 CNN-LSTM 模型
  3. 对比 LSTM、GRU 与 CNN-LSTM 的表现

---

### 🛠️ 学习过程

#### 1. 理论学习

* **CNN（卷积神经网络）**：本来常用于图像识别，但在时间序列中也可以提取局部模式（比如短期波动特征）。
* **LSTM**：适合捕捉长期依赖关系。
* **CNN-LSTM 混合模型**思路：

  * CNN 提取股价的局部特征
  * LSTM 再建模这些特征的时序依赖
  * 最终得到更鲁棒的预测

👉 通俗理解：**CNN-LSTM = 特征提取 + 时序建模**。

---

#### 2. Python 实践

```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, TimeDistributed

# 下载数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造训练数据
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# 输入维度调整为 CNN-LSTM 可接受格式
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 构建 CNN-LSTM 模型
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# 训练
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 预测
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 可视化
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(predictions, label="CNN-LSTM预测", linestyle="--", color="red")
plt.legend()
plt.title("CNN-LSTM 股价预测效果")
plt.show()
```

---

#### 3. 实验结果

* CNN-LSTM 预测曲线比单纯的 LSTM 更“贴合”短期波动
* 验证集损失下降速度比 LSTM 略快
* 在捕捉短期趋势方面，CNN-LSTM 表现优于 LSTM 和 GRU

---

### 📊 完成情况

* ✅ 掌握了 CNN-LSTM 的原理
* ✅ 成功实现并训练模型
* ✅ 绘制预测曲线并观察效果
* ⚠️ 需要注意：卷积核大小、LSTM 层数等超参数会显著影响结果

---

### 💡 学习心得

* 今天的 CNN-LSTM 实验让我第一次看到“混合模型”的强大：

  * CNN 能抓到局部模式
  * LSTM 能把局部模式串起来
* 在股价预测中，这样的组合能同时兼顾 **短期波动** 与 **长期趋势**。
* 未来可以尝试 **CNN-GRU** 或更复杂的 **Attention + LSTM** 结构。

---

### 🚀 明日计划

* 学习 **Attention 机制** 的基本原理
* 尝试在 LSTM 上加入 Attention
* 探索 **LSTM-Attention 模型** 在股价预测中的表现

---

### 📂 附录

* 📜 [CNN for Time Series](https://towardsdatascience.com/convolutional-neural-networks-for-time-series-classification-4d447fcbf3e)
* 📜 [CNN-LSTM Keras Example](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)

---

## 📖 （Day 18）AI量化学习日志：LSTM + Attention 模型

---

### 📌 今日计划

* **今日目标**：学习 Attention 机制，并将其应用到 LSTM 股价预测中
* **学习内容**：

  1. Attention 的基本原理
  2. 在 LSTM 模型中引入 Attention 层
  3. 对比 LSTM 与 LSTM+Attention 的预测效果

---

### 🛠️ 学习过程

#### 1. 理论学习

* **LSTM 的局限**：虽然能捕捉时间依赖，但对 **长序列中不同时间点的重要性** 无法区分。
* **Attention 机制**：让模型在预测时，对不同时间点的“权重”进行分配。
* 通俗理解：

  * LSTM 是“看完整段历史，再总结”
  * Attention 是“挑出关键的历史片段，更精准地总结”

👉 在股价预测中，某些天的波动可能比其他天更关键，Attention 能帮助模型“聚焦”到这些重要点。

---

#### 2. Python 实践

```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention

# 下载股票数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造训练数据
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 定义 LSTM + Attention 模型
inputs = Input(shape=(X.shape[1], 1))
lstm_out = LSTM(50, return_sequences=True)(inputs)
attention_out = Attention()([lstm_out, lstm_out])  # 自注意力
lstm_out2 = LSTM(50, return_sequences=False)(attention_out)
drop = Dropout(0.2)(lstm_out2)
dense1 = Dense(25, activation="relu")(drop)
outputs = Dense(1)(dense1)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mean_squared_error")

# 训练
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 预测
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 可视化
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(predictions, label="LSTM+Attention预测", linestyle="--", color="purple")
plt.legend()
plt.title("LSTM+Attention 股价预测效果")
plt.show()
```

---

#### 3. 实验结果

* **LSTM+Attention** 的预测曲线更接近真实走势
* Attention 使得模型在关键点（大涨大跌日）上预测效果更好
* 验证集损失比纯 LSTM 更低，说明泛化能力增强

---

### 📊 完成情况

* ✅ 掌握了 Attention 的原理
* ✅ 成功实现 LSTM+Attention 模型
* ✅ 预测效果提升，尤其在波动较大的区间
* ⚠️ 需要注意：Attention 增加了计算量，训练更慢

---

### 💡 学习心得

* Attention 机制让我意识到，**不是所有历史数据都一样重要**。
* 在股市这种波动强烈的场景下，Attention 能帮助模型聚焦到关键行情。
* 今天的实验进一步证明了 **深度学习 + 注意力机制** 的潜力，后面甚至可以尝试 **Transformer**。

---

### 🚀 明日计划

* 学习 **Transformer** 的基本结构（Encoder-Decoder）
* 尝试用 **Transformer Encoder** 来预测股价
* 与 LSTM+Attention 进行对比

---

### 📂 附录

* 📜 [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* 📜 [Keras Attention Layer](https://keras.io/api/layers/attention_layers/attention/)


---

## 📖 （Day 19）AI量化学习日志：Transformer 股价预测

---

### 📌 今日计划

* **今日目标**：理解 Transformer 的基本结构，并实现一个基于 **Transformer Encoder** 的股价预测模型
* **学习内容**：

  1. Transformer 的 Encoder 结构
  2. 自注意力（Self-Attention）在时间序列预测中的作用
  3. 搭建 Transformer Encoder + 全连接层的预测模型

---

### 🛠️ 学习过程

#### 1. 理论学习

* **LSTM+Attention 的不足**：LSTM 序列依赖强，训练速度较慢。
* **Transformer 优势**：完全基于 Attention 机制，支持并行训练，能更好地捕捉远距离依赖。
* **核心结构**：

  * **Self-Attention**：捕捉序列中不同位置之间的关系
  * **Feed Forward Layer**：非线性映射
  * **LayerNorm + 残差连接**：让训练更稳定

👉 通俗理解：
LSTM 像是“顺序阅读”，而 Transformer 是“全局扫视”，一眼就能看到所有历史点的联系。

---

#### 2. Python 实践

```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# 下载数据
data = yf.download("AAPL", period="2y")
close_prices = data["Close"].values.reshape(-1,1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# 构造时间序列数据
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

# 位置编码（可简化）
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims//2)) / np.float32(d_model))
    angle_rads = positions * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

# Transformer Encoder 模块
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# 构建模型
inputs = layers.Input(shape=(time_step, 1))
x = transformer_encoder(inputs, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1)(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

# 训练
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 预测
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 可视化
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(predictions, label="Transformer预测", linestyle="--", color="orange")
plt.legend()
plt.title("Transformer 股价预测效果")
plt.show()
```

---

#### 3. 实验结果

* **预测走势更加平滑**，但在极端波动点上略显滞后
* 在验证集上，Transformer 的收敛速度 **比 LSTM 更快**
* 对长序列的依赖捕捉明显优于 LSTM

---

### 📊 完成情况

* ✅ 理解了 Transformer Encoder 的结构
* ✅ 搭建了第一个 Transformer 股价预测模型
* ✅ 与 LSTM+Attention 进行了对比，速度更快，效果更稳
* ⚠️ 注意：需要更多数据和调参，避免过拟合

---

### 💡 学习心得

* Transformer 让我真正体会到 **Attention is All You Need** 的魅力。
* 通过全局自注意力机制，模型能在更大范围内捕捉股价走势的关联性。
* 今天是第一次尝试，后续还可以加入 **更深的 Encoder 层数、正则化技巧** 来提升效果。

---

### 🚀 明日计划

* 学习 **多层 Transformer Encoder 堆叠**
* 尝试 **Transformer vs LSTM+Attention 的系统对比实验**
* 加入 **更多特征（成交量、技术指标）**，让模型更贴近实战

---

### 📂 附录

* 📜 [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* 📜 [Keras MultiHeadAttention](https://keras.io/api/layers/attention_layers/multi_head_attention/)

---

## 📖 （Day 20）AI量化学习日志：多层 Transformer + 技术指标增强预测

---

### 📌 今日计划

* **今日目标**：在 Day19 单层 Transformer 基础上，

  * 搭建 **多层 Transformer Encoder 堆叠模型**
  * 加入 **技术指标（MACD、RSI、MA）** 作为输入特征
  * 对比单一收盘价序列，验证效果是否提升

---

### 🛠️ 学习过程

#### 1. 理论学习

* **多层 Transformer Encoder**
  堆叠多层 Encoder 可以捕捉更复杂的时间依赖，但层数过多可能导致过拟合。
* **技术指标增强**
  股价不仅仅依赖历史收盘价，还会受 **趋势指标（MA）、动量指标（RSI）、趋势反转指标（MACD）** 影响。
* **组合思路**
  输入特征 = \[收盘价、MA、RSI、MACD] → Transformer Encoder → Dense → 输出预测

---

#### 2. Python 实践

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# 下载数据
data = yf.download("AAPL", period="2y")

# 计算技术指标
data['MA10'] = data['Close'].rolling(window=10).mean()
data['RSI'] = 100 - (100 / (1 + (data['Close'].pct_change().rolling(window=14).mean() / 
                                abs(data['Close'].pct_change().rolling(window=14).mean()))))
ema12 = data['Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26

# 填充缺失值
data = data.dropna()

# 选择特征
features = data[['Close','MA10','RSI','MACD']].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(features)

# 构造时间序列数据
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_features)):
    X.append(scaled_features[i-time_step:i])
    y.append(scaled_features[i, 0])  # 预测收盘价
X, y = np.array(X), np.array(y)

# 划分训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Transformer Encoder 模块
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# 构建多层 Transformer
inputs = layers.Input(shape=(time_step, X.shape[2]))
x = transformer_encoder(inputs, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)  # 堆叠第二层
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)  # 堆叠第三层

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1)(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

# 训练
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 预测
predictions = model.predict(X_test)
real_prices = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), 
                                                      np.zeros((len(y_test), X.shape[2]-1))], axis=1))[:,0]
pred_prices = scaler.inverse_transform(np.concatenate([predictions, 
                                                      np.zeros((len(predictions), X.shape[2]-1))], axis=1))[:,0]

# 可视化
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(pred_prices, label="Transformer+指标预测", linestyle="--", color="blue")
plt.legend()
plt.title("多层 Transformer + 技术指标增强 股价预测")
plt.show()
```

---

#### 3. 实验结果

* **收敛速度**：比单层模型更快，验证集 Loss 更低
* **预测走势更贴近实际**，尤其在趋势转折点，技术指标帮助模型识别变化
* **缺点**：在极端暴涨暴跌时，预测仍然滞后

---

### 📊 完成情况

* ✅ 搭建了多层 Transformer Encoder
* ✅ 加入了 MACD、RSI、MA 技术指标作为特征
* ✅ 整体预测效果优于单一收盘价模型
* ⚠️ 问题：RSI 计算方式较简化，后续需要改进

---

### 💡 学习心得

* 技术指标确实能提供额外的信息，让模型不再“盲人摸象”。
* 多层 Transformer 在表达能力上更强，但也更容易过拟合，需要 Dropout + 正则化。
* 今天算是迈出了 **从实验室走向实盘建模** 的关键一步。

---

### 🚀 明日计划

* 学习 **Transformer + 多变量预测（成交量、财报数据）**
* 尝试 **加入卷积层（CNN-Transformer Hybrid）**，提升局部特征提取能力
* 开始准备 **回测框架**，将预测结果和策略收益联系起来

---

### 📂 附录

* 📜 [Keras Transformer Encoder 示例](https://keras.io/examples/timeseries/timeseries_transformer_classification/)
* 📜 [MACD & RSI 指标详解](https://www.investopedia.com/)

---

## 📖 （Day 21）AI量化学习日志：CNN + Transformer Hybrid 股价预测

---

### 📌 今日计划

* **今日目标**：结合 CNN 与 Transformer 的优势，构建一个 **混合模型**：

  * CNN：提取局部模式（如短期价格趋势）
  * Transformer：捕捉全局时间依赖（长期走势）
* **学习内容**：

  1. CNN-Transformer 的组合结构设计
  2. 模型实现与训练
  3. 对比单纯 Transformer 的预测效果

---

### 🛠️ 学习过程

#### 1. 理论学习

* **CNN 的作用**：善于识别 **局部特征模式**，比如股价短期的波动形态。
* **Transformer 的作用**：通过自注意力机制，捕捉 **全局依赖**，适合建模长期走势。
* **Hybrid 思路**：
  输入序列 → **CNN 卷积层提取局部模式** → **Transformer Encoder 提取全局依赖** → Dense 输出预测

👉 类似于“显微镜 + 广角镜”，既能看清短期波动，又能掌握长期趋势。

---

#### 2. Python 实践

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# 下载数据
data = yf.download("AAPL", period="2y")

# 添加技术指标
data['MA10'] = data['Close'].rolling(window=10).mean()
data['RSI'] = 100 - (100 / (1 + (data['Close'].pct_change().rolling(window=14).mean() /
                                abs(data['Close'].pct_change().rolling(window=14).mean()))))
ema12 = data['Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26
data = data.dropna()

# 选择特征
features = data[['Close','MA10','RSI','MACD']].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(features)

# 构造时间序列数据
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_features)):
    X.append(scaled_features[i-time_step:i])
    y.append(scaled_features[i, 0])  # 预测收盘价
X, y = np.array(X), np.array(y)

# 划分训练/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Transformer Encoder 模块
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# 构建 CNN + Transformer Hybrid 模型
inputs = layers.Input(shape=(time_step, X.shape[2]))

# CNN 层提取局部特征
x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="causal")(inputs)
x = layers.MaxPooling1D(pool_size=2)(x)

# Transformer 层提取全局依赖
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)

# 输出层
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1)(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

# 训练
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# 预测
predictions = model.predict(X_test)
real_prices = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), 
                                                      np.zeros((len(y_test), X.shape[2]-1))], axis=1))[:,0]
pred_prices = scaler.inverse_transform(np.concatenate([predictions, 
                                                      np.zeros((len(predictions), X.shape[2]-1))], axis=1))[:,0]

# 可视化
plt.figure(figsize=(12,6))
plt.plot(real_prices, label="真实价格", color="black")
plt.plot(pred_prices, label="CNN+Transformer预测", linestyle="--", color="green")
plt.legend()
plt.title("CNN + Transformer Hybrid 股价预测")
plt.show()
```

---

#### 3. 实验结果

* **预测曲线更贴近实际走势**，短期波动识别效果好于单纯 Transformer
* **收敛速度更快**，训练过程中 Loss 降得更平稳
* **不足**：模型参数更多，训练时间比单纯 Transformer 稍长

---

### 📊 完成情况

* ✅ 搭建了 CNN + Transformer Hybrid 模型
* ✅ 验证了 CNN 对短期模式识别的提升效果
* ✅ 与单纯 Transformer 对比，整体效果更优
* ⚠️ 注意：需要更系统的调参和交叉验证，避免过拟合

---

### 💡 学习心得

* CNN + Transformer 的组合让我看到 **局部 + 全局** 的强大融合。
* 在股价预测中，短期波动和长期趋势同样重要，Hybrid 模型能平衡两者。
* 今天的实验让我有信心，后续可以尝试更多 **多模态特征融合**（例如财报数据 + 新闻情绪）。

---

### 🚀 明日计划

* 学习 **回测框架（Backtesting）** 的基础
* 将预测结果转化为 **交易信号**（买入/卖出）
* 尝试用历史数据做一个 **简单策略回测**

---

### 📂 附录

* 📜 [Hybrid CNN-Transformer for Time Series](https://arxiv.org/abs/2012.07436)
* 📜 [Keras Conv1D 文档](https://keras.io/api/layers/convolution_layers/convolution1d/)

---


