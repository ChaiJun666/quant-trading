
---

## 🏗 阶段 1：量化交易基础

在用 AI 做量化之前，要先打牢 **金融 & 量化基础**。

1. **金融市场基础**

   * 股票、期货、外汇、加密货币市场的基本规则
   * 常见指标：收益率、夏普比率、最大回撤、波动率
   * 技术分析 & 基本面分析（K线、均线、PE、财报等）
2. **量化交易基础知识**

   * 策略类型：趋势跟随、均值回归、套利、因子模型
   * 风险控制：仓位管理、止盈止损、资金曲线回测
   * 回测框架学习：`backtrader` / `zipline` / `quantstats`
3. **数据获取**

   * Tushare、Akshare、yfinance（免费数据源）
   * Wind、聚宽（付费/半付费数据源）

---

## 🤖 阶段 2：AI/机器学习基础

把 AI 技术和量化结合前，需掌握机器学习基础。

1. **机器学习**

   * 监督学习：线性回归、逻辑回归、决策树、随机森林
   * 无监督学习：聚类、降维（PCA）
   * 时间序列预测：ARIMA、LSTM、Transformer
   * 库：`scikit-learn`、`xgboost`
2. **深度学习**

   * PyTorch/TensorFlow 基础
   * LSTM、GRU、Attention 在金融时间序列的应用
   * 强化学习（DQN、PPO 等）在交易中的应用
3. **特征工程**

   * 金融特征：价格、成交量、动量、波动率、因子指标
   * 技术指标库：`TA-Lib` / `finta`

---

## 📊 阶段 3：AI量化实战

把 AI 方法应用到量化交易策略上。

1. **经典项目**

   * 股票价格预测（LSTM/GRU/Transformer）
   * 因子选股（ML 分类/回归模型）
   * 强化学习训练交易 Agent
2. **回测 & 风控**

   * 使用 `backtrader` 搭建策略 + AI预测信号
   * 用 `quantstats` 评估策略（收益率、夏普比率等）
   * 多因子模型结合 ML 优化投资组合
3. **实盘交易**

   * 接入券商 API（如 IB、掘金、聚宽、交易宝）
   * 模拟盘测试 → 小资金实盘
   * 自动化交易系统架构（行情获取 → 策略预测 → 下单执行）

---

## ⚙ 阶段 4：进阶与优化

1. **量化研究进阶**

   * 多因子模型、风险模型（Barra）
   * 高频交易（HFT）基础（需考虑延迟、撮合速度）
2. **AI前沿方法**

   * 生成模型（GAN）预测时间序列
   * 大语言模型（LLM）结合新闻/研报做情绪分析
   * 图神经网络（GNN）做资产相关性建模
3. **系统工程**

   * Docker/Kubernetes 部署量化系统
   * 分布式计算（Spark、Ray）
   * 数据库（ClickHouse、TDengine）处理海量行情

---

## 📚 推荐学习资源

* **书籍**

  * 《量化投资：以Python为工具》– Ernest Chan
  * 《机器学习与量化投资》– Lopez de Prado
  * 《Advances in Financial Machine Learning》– Marcos López de Prado
* **课程**

  * Coursera：Machine Learning for Trading
  * Udemy：Python for Financial Analysis and Algorithmic Trading
  * 国内平台：聚宽量化、优矿量化社区

---

## 🚀 建议学习顺序（路径图）

1. **量化基础**（金融指标 + 策略原理 + 数据获取）
2. **机器学习/深度学习**（掌握 ML、DL、时序模型）
3. **量化实战项目**（股票预测、因子模型、强化学习交易）
4. **回测 & 风控体系**（性能指标 + 风险管理）
5. **实盘交易系统搭建**（API + 自动化下单）
6. **进阶研究**（多因子模型、前沿 AI 方法）

---


