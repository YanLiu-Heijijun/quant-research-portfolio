# Quant Research Portfolio | 刘言

量化研究与金融工程个人作品集，整理自个人 CQF 学习项目，重点呈现金融问题的实现、实验验证和结果解释。

Independent implementations and research summaries covering portfolio construction, financial machine learning, derivatives pricing and market-risk validation. This is a personal portfolio, not an official CQF resource.

## 项目

- [Black-Litterman 组合构建与 ESG 行业情景分析](projects/black-litterman/README.md)
- [沪深 300 指数上涨事件预测与特征筛选](projects/csi300-lightgbm/README.md)
- [期权 Monte Carlo 定价与方差缩减](projects/monte-carlo-pricing/README.md)
- [投资组合构建与市场风险模型验证](projects/risk-model-validation/README.md)

[公开版个人简介](resume/resume_public.md)

## 如何运行

建议使用 Python 3.12，按各项目 README 安装依赖、准备有权使用的数据。模拟示例用于验证代码运行，不代表真实市场收益。

```text
python -m venv .venv
# 激活虚拟环境后，按所选项目 README 安装依赖并运行。
```

仓库未打包授权行情或私有快照，因此不承诺一键复现全部历史结果。公开整理版如修正了原实验，项目 README 会区分历史记录与当前实现。

## 说明

- 原始 CQF 考试题目、提交版报告、声明页和评分材料未上传。
- iFinD 及其他受许可限制的数据未随仓库分发；请按项目 README 获取或准备本地数据。
- 项目结果用于教育和研究展示，不构成投资建议。
- 课程项目中的模型结果应结合 README 中的假设和限制理解。
- 各 Notebook 已清空执行输出，并以新写的项目说明代替原始报告正文。
- MIT 许可仅适用于作者有权授权的原创代码，不授予第三方数据、课程材料或外部文献的使用权。

作者：刘言 · [GitHub](https://github.com/YanLiu-Heijijun)
