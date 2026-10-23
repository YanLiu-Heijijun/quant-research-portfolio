# Black-Litterman 组合构建与 ESG 行业情景分析

这是已完成 CQF Final 研究的**公开代码整理版**，用于展示组合建模、研究验证和负面结果分析。完整原始提交报告、题目、执行输出、市场数据及审计快照均未打包。Notebook 只保留研究代码与新写的简短导航说明。

**当前公开包不是一键复现包。** 完整研究依赖未公开的本地授权数据及其审计文件；仓库另附完全离线的合成数据检查，验证数值实现，不生成课程成绩或真实投资收益。

## 研究范围

- 13 个 ETF：XLK、XLF、XLE、XLI、XLP、XLY、XLV、XLB、XLRE、XLC、XLU、TLT、GLD；ESGU、MTUM、SPY 作为因子或市场代理。
- 静态分析以 ETF AUM 权重反向优化均衡收益，结合 XLK 绝对观点和行业篮子的相对观点，计算置信度缩放的 Omega、BL 后验均值及预测协方差。
- 实现全额投资、仅做多、单资产上限 20% 的均值–方差、最大夏普与 ERC 配置；使用观点消融、协方差敏感性、HAC/Newey–West、区块 Bootstrap 和 Holm 校正检查结论。
- 滚动评估每次使用 756 个交易日训练，按月更新，延迟一个交易日实施，基本情景计入 10 bp 单边交易成本。该滚动部分采用等权先验与等权相对篮子，**没有把后期 AUM 快照当作历史逐时点权重**。

## 结果与解释边界

以下是原研究保存输出中的结果摘录，**本次公开整理没有重新执行完整历史实验**。回顾性滚动评估覆盖 2024-08-01 至 2026-07-23，共 495 个持有日、24 次调仓。

- 双观点组合费后 CAGR 为 **13.21%**、夏普 **0.768**、最大回撤 **−11.99%**。
- 匹配的无观点中性 MVO 基准费后 CAGR 为 **14.25%**、夏普 **0.837**、最大回撤 **−12.38%**。
- 双观点相对无观点的年化平均主动收益为 **−0.93%**，HAC 95% 区间 **[−2.23%, 0.36%]**；原始 HAC p 值 **0.1589**，匹配消融族内 Holm 校正后 **0.3178**。主动收益差与 CAGR 差是不同指标，不混用。

这组结果没有证明观点带来正的增量收益，更没有验证 ESG Alpha。更合适的表述是：**构建并检验一组观点，发现其在该历史样本中未优于匹配的无观点配置。**

这里使用 **pseudo-OOS（回顾性滚动样本外）** 一词：每轮估计窗口与持有窗口有时间隔离，但资产池、行业篮子和研究情景是回顾性选择的，存在研究选择偏差，不能等同于预注册实验、独立未来留出样本或实盘。行业篮子并非逐公司碳排放测量；ETF AUM 只是静态均衡权重代理。对角 Omega、参数设定、协方差估计、交易成本假设、有限样本 HAC 近似和 Bootstrap 的弱平稳假设均限制外推。反向观点属于探索性压力情景，不能事后挑选最佳情景宣称发现策略。

## 先运行离线合成检查

使用 Python 3.10 或更高版本，在本项目目录执行：

```console
python -m pip install -r demo/requirements.txt
python demo/smoke_demo.py
```

脚本从公开 notebook **只加载** `BlackLittermanModel` 与 `PortfolioOptimizer` 类定义，不执行数据读取或其它单元。它用固定随机种子的合成数据检查：单观点后验与独立闭式计算一致、无观点时后验均值等于先验、预测协方差正定、权重投影和 SciPy SLSQP 解满足预算及上下界约束。它不读取真实数据、不联网、不输出收益率。

仅安装 NumPy 与 pandas 时，可运行 `python demo/smoke_demo.py --core-only`；此模式明确跳过优化求解器，只检查后验与权重投影。通过这些检查不代表完整研究、统计推断或交易实现均已验证。

## 完整研究需要的本地输入

`notebooks/analysis_clean.ipynb` 默认且仅从本地审计快照加载数据。历史数据供应商辅助代码保留用于阅读，但公开执行路径禁用在线下载，也不需要提交 API 密钥。

所需目录如下；`inputs/` 和运行生成的 `outputs/` 已在项目内忽略，不应上传：

```text
black-litterman/
  inputs/snapshots/2026-07-24/
    snapshot_manifest.json
    snapshot_metadata.json
    fred_DGS3MO_raw.csv
    main/
      asset_prices.csv
      factor_prices.csv
      asset_returns.csv
      asset_excess.csv
      factor_returns.csv
      factor_excess.csv
      rf_dgs3mo_daily.csv
      rf_dgs3mo_annual_percent.csv
      rf_metadata.json
      market_weights.csv
      market_aum.csv
    robustness_5y/
      [与 main 相同的一组文件，覆盖五年窗口]
```

价格、收益与超额收益表以日期为索引，资产表以 ETF 代码为列；因子表使用研究管线的因子列名。收益表是日简单收益，超额收益逐日减去 DGS3MO / 100 / 252。无风险序列使用向前填充、不回填；截止日、日期对齐、首行处理与原始 FRED 序列需满足 notebook 内的校验。`market_weights.csv` 以代码为索引，`market_aum.csv` 保留 AUM 及来源日期等元数据。

审计清单必须含每个文件的相对路径、字节数和 SHA-256。加载器还检查 `status=complete`、分析截止日 2026-07-24、完整市场收盘日 2026-07-23、FRED 最后观测日 2026-07-22，以及原始无风险文件和元数据的一致性。**只有相同文件名不够，也不应伪造清单或关闭断言来使代码通过。** 仓库未分发这些文件；需要先在本地取得可合法使用的完整输入，其他样本需要同步调整数据处理、日期口径和校验，而不会复现上述历史数值。

完整依赖安装与启动方式：

```console
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/analysis_clean.ipynb
```

检查授权输入后，在 notebook 中按顺序执行。输出写入本项目 `outputs/runs/<运行标识>/`。CVXPY 为可选依赖，缺失时使用 SciPy SLSQP。依赖文件给出兼容范围，不是原始环境锁；不同求解器或版本可能产生数值差异。公开整理完成了语法与合成核心检查，未宣称完整历史复现。
