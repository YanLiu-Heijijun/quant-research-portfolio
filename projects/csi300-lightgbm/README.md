# 沪深 300 指数上涨事件预测与特征筛选

基于日频价格与成交量特征，使用 **LightGBM** 预测沪深 300 指数下一交易日的对数收益是否超过 **0.5%**。这是单一指数的时间序列分类研究；类别 0 也包括不足阈值的小幅上涨。

公开版保留了数据处理、46 个候选特征、特征筛选、参数搜索和分类评价的研究计算，并重写了说明。未包含考试题目、原始提交报告、第三方行情、登录代码、原始模型文件或已执行的 Notebook 输出。本项目没有实现 XGBoost 模型或成分股截面选股。

## 历史研究记录

下列数字来自此前本地研究的已保存结果，**不是本公开版本新运行得到的结果**：

- 原始数据实际覆盖 2021-05-24 至 2026-05-20，共 1,209 条日频记录；滚动特征清理后保留 1,088 条。
- 46 个候选特征经互信息、相关性过滤、RFECV 和模型重要性筛选后保留 3 个：20 日收益、OBV 5 日动量、20 日收益峰度。
- 按时间划分训练与测试段，并删除训练段末尾一条跨越标签边界的样本；训练 869 条，测试 218 条。测试样本日期为 2025-06-24 至 2026-05-19。
- LightGBM 基线测试 ROC-AUC 为 0.4891；参数搜索后测试 ROC-AUC 为 **0.5302**，训练 ROC-AUC 为 0.6350。该区分能力较弱，不代表统计显著的 Alpha。
- 以训练集 F1 选择阈值后，测试正类 precision 30.64%、recall 77.94%、F1 43.98%，整体准确率 38.07%。高召回伴随较多误报，不能单独作为有效预测的证据。

保留固定随机种子有助于重复实验，但原始依赖环境没有完整锁定；新数据、不同版本或快速模式的结果不会保证等于以上数字。

## 环境与运行

建议使用 Python 3.11 或 3.12，新建虚拟环境，在本项目目录运行：

```shell
python -m venv .venv
```

激活虚拟环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`；macOS/Linux 使用 `source .venv/bin/activate`。然后：

```shell
python -m pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python -m jupyterlab notebooks/analysis_clean.ipynb
```

生成脚本只创建 **人工模拟行情** `data/private/synthetic_daily.csv`，不联网、不调用终端 API，也不覆盖已有文件。它用于检查整个流程是否可以执行，不用于金融结论。Notebook 默认读取这个文件，`DATA_SOURCE_KIND='synthetic'`、`QUICK_MODE=True`，只搜索 2 组模型参数；逐格运行或选择 Run All。

如安装 TA-Lib 时找不到适合平台的二进制 wheel，请按 [TA-Lib Python 安装说明](https://github.com/TA-Lib/ta-lib-python#installation) 准备相应环境。此项目依赖 TA-Lib 计算技术指标，没有使用近似替代实现。

## 使用自己的本地数据

将有权使用的单一指数行情 CSV 放到本项目的 `data/private/`。在 Notebook 配置中将 `DATA_PATH` 指向该文件，并将 `DATA_SOURCE_KIND` 改为 `'licensed-local'`。该标记用于标识本次运行；程序不能代替你验证授权或来源。

需要以下列，额外列会被忽略：

- `time`：交易日期，推荐 `YYYY-MM-DD`，每个交易日一行，不允许重复日期。
- `open_price`：当日开盘价。
- `pre_close_price`：上一交易日收盘价，必须与前一行的收盘价对应。
- `high_price`、`low_price`：当日最高价与最低价。
- `close_price_nvnf`：当日收盘价；名称沿用研究代码，所有价格必须采用一致口径，不能混用复权与未复权数据。
- `volume`：当日成交量，使用同一种单位，不允许负值、零或缺失值。

CSV 首行应为：

```csv
time,open_price,pre_close_price,high_price,low_price,close_price_nvnf,volume
```

提供至少 750 行连续交易日数据，并确保训练、测试和各时间验证折都包含两类标签。Notebook 会检查日期重复、有限正值、OHLC 关系及前收盘对齐；它不会自动补齐缺失交易日或修复数据。首行的前收盘需要来自此前一个交易日。模型将最后一行排除，因为无法构造它的次日标签。

若希望使用历史完整参数网格，将 `QUICK_MODE=False`：486 组参数 × 5 个时间验证折，另有 RFECV 的模型训练，可能耗时较长。特征筛选结果随输入数据变化，不强制保留 3 个特征。

## 输出与检查

Notebook 展示本次运行的特征数量、切分日期、ROC-AUC、混淆矩阵和分类报告，并明确标注数据类型及 quick/full 模式。设置 `WRITE_RESULTS=True` 会把当前指标、特征清单、分类报告及图保存在被忽略的 `outputs/private/`；默认不导出。

检查 Notebook 结构、代码语法及已清除的输出：

```shell
python scripts/validate_notebook.py
```

验证脚本只解析代码，不训练模型。完整运行需要上述依赖及本地 CSV。运行后不要把带有本地数据输出的 Notebook 直接提交；可用下面的命令清除输出，再执行检查：

```shell
jupyter nbconvert --clear-output --inplace notebooks/analysis_clean.ipynb
python scripts/validate_notebook.py
```

2026-10-23 已通过 `nbformat` 格式校验、8 个代码单元的 Python 语法检查和空输出检查；使用人工模拟 CSV 完整执行了快速模式，包括 46 个特征构建、特征筛选、两组参数搜索及分类评价，且运行前后 Notebook 文件保持不变。测试环境为 Python 3.13、LightGBM 4.5.0、TA-Lib 0.6.8；运行中有 scikit-learn API 弃用提示。上述检查不等于原始市场数据结果复现，模拟指标不作为金融结论。

## 方法边界

- 外层测试段不参与特征拟合或参数搜索；但标准化和特征筛选先在整个外层训练段完成，再进行内部时间序列交叉验证。因此当前实现不是每一折内独立筛选的嵌套验证，内部 CV 分数可能偏乐观。
- 内部 `TimeSeriesSplit` 未设置 gap；外层删去一条边界样本的处理未同步到各内部折。分类阈值在最终模型的训练集预测上选择，也可能过拟合。
- 标准化后的方差阈值主要筛除恒定特征；LightGBM 重要性使用默认的 split 次数，不是 gain。当前选择这些做法是为明确保留历史研究设计，并不表示它们是最佳设计。
- 部分技术指标和 OBV 使用当日完整收盘信息。它们可用于收盘后的预测，但不能据此假设在同一收盘价无延迟成交。
- 原始探索性回测未包含成本，且存在对数收益复利口径、基准日期对齐和回撤计算的问题。公开 Notebook 已排除该部分，不把其中的收益、夏普或回撤作为验证结果。

后续方向是将全部变换和特征筛选放进每个训练折，使用适当 gap 的嵌套或滚动验证，通过折外预测选择阈值并估计指标不确定性。若进一步研究交易价值，还需确定可交易标的、成交时点、成本及一致的收益和净值核算。这些是待完成的改进，不是本项目已实现的能力。
