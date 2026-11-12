<!-- 由 @Classmateliu 创建、编写及维护 -->

<!-- 这里的内容是标题及logo部分的内容 -->
<div align="center">
<img src="image/logo_image.png" width="120"/>
</div>
<h1 align="center">仇恨言论检测</h1>

<!-- 这里对应中英文切换部分 -->
<h4 align="center">
  简体中文 | <a href="README_en.md">English</a>
</h4>
<div align="center">

<!-- 许可证与版本徽章 -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ICT-classmateliu/Hate-Speech-Detection?label=Release&color=blue)](https://github.com/ICT-classmateliu/Hate-Speech-Detection/releases)
<!-- MIT License
别人：可以自由使用、修改、再发布、商用。
要求：保留原作者版权声明和许可证文本。
责任：作者不对使用后造成的问题负责。 -->

<!-- 小标题部分 -->
<strong>@Classmateliu 采用 Python 开发，基于若干工具包实现</strong>
</div>

## 项目概述 ##

本项目是一项围绕文本情感分类相关任务开展的小型研究，以仇恨言论检测为具体研究场景，探索文本预处理、特征工程及不同分类模型在该任务中的应用与表现。
<br/>__研究内容：__ 仇恨言论通常被理解为任何基于个人或群体身份的口头、书面或行为表达。换句话说，是指基于他人的宗教、民族、国籍、种族、肤色、血统、性别或其他身份特征，对其进行攻击、贬低或煽动仇视的言论
<br/>__关键技术：__ 将仇恨言论检测建模为文本分类任务，结合 TF-IDF、n-gram、依存句法及情感词典等特征，在标注数据上训练和比较多种分类模型，探索情感相关特征对仇恨言论识别的作用。
<br/>__评价标准：__ 准确率、精确率（macro）、召回率（macro）、F1（weighted / macro）及 ROC/AUC（部分模型）

## 相关功能 ##

运行入口位于 `main/`：

- `train_final_model.py`：训练带类别权重的 PyTorch MLP，并将模型权重、标准化器、特征列和 TF-IDF 向量器保存至 `main/artifacts/`。
- `app_gradio.py`：加载模型文件并提供 Gradio 交互演示，包含相似样本匹配与特征可视化。
- `hate_speech_detection_gpu.py`：比较基准模型、完整特征模型及集成模型。PyTorch 根据 CUDA 可用性选择 GPU 或 CPU，XGBoost 也包含 GPU / CPU 配置；训练耗时取决于硬件和环境。

当前训练与演示流程使用英文数据及预计算特征；中文数据集作为研究资料提供。演示包含相似样本检索和启发式规则，其输出用于探索分析，不能直接视为独立测试集上的模型评估结果。

模型设置如下：

| 模型 | 输入特征 | 框架/库 | 使用GPU | 用途 | 训练方式 | 评价指标 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 基准模型 (Baseline MLP) | 加权 TF-IDF | PyTorch | CUDA 可用时 | 作为最简单的参考模型 | 5 折交叉验证 | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| Gradient Boosting (GB) | 加权 TF-IDF | scikit-learn | 否，CPU | 参考模型 | 3 折交叉验证 | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| Random Forest (RF) | 完整特征 | scikit-learn | 否，CPU | 参考模型 | 3 折交叉验证 | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| XGBoost | 完整特征 | XGBoost | CUDA 可用时 | 集成 | GPU 5 折 / CPU 3 折交叉验证 | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| PyTorch MLP | 完整特征 | PyTorch | CUDA 可用时 | 主模型 | 5 折交叉验证 + 留出集评估 | F1 (weighted), Accuracy, Precision (macro), Recall (macro), ROC/AUC |
| Voting 集成 | 完整特征 | scikit-learn | MLP/RF 使用 CPU；XGBoost 视配置 | 集成学习 | sklearn MLP + RF + XGBoost（soft voting） | F1 (weighted / macro), Accuracy, Precision (macro), Recall (macro) |
| Stacking 集成 | 完整特征 | mlxtend | MLP/RF 使用 CPU；XGBoost 视配置 | 集成学习 | sklearn MLP + RF + XGBoost，2 折内部交叉验证，LogisticRegression 作为元分类器 | F1 (weighted / macro), Accuracy, Precision (macro), Recall (macro) |

## 使用说明 ##

主程序使用的第三方依赖包括 `torch`、`numpy`、`pandas`、`scikit-learn`、`xgboost`、`mlxtend`、`matplotlib`、`gradio`、`plotly` 和 `joblib`。`re`、`string`、`json` 等属于 Python 标准库，无需单独安装。仓库尚未提供锁定版本的依赖文件，模型文件与依赖版本的兼容性需在本地验证。

在仓库根目录执行（终端需能调用 Python）：

```powershell
python -m pip install torch numpy pandas scikit-learn xgboost mlxtend matplotlib gradio plotly joblib
python main/app_gradio.py
```

仓库已包含 `main/artifacts/` 和 `test_feature_dataset/`，可先尝试启动演示。如需重新训练或比较模型，分别执行：

```powershell
python main/train_final_model.py
python main/hate_speech_detection_gpu.py
```

重新训练会更新 `main/artifacts/` 中的模型文件。使用 GPU 时，需要本地 PyTorch、CUDA 与显卡环境兼容。

重新生成特征还需要 NLTK、`stanfordcorenlp`、Java 和本地 Stanford CoreNLP。特征脚本保留了基于当前工作目录的输入路径及旧版 API（如 `get_feature_names()`、`sklearn.cross_validation`）；运行前需调整路径、词典文件名、CoreNLP 安装位置及相关 API。现有脚本尚不能直接按顺序一键生成训练所需的全部文件。

## 词典和词汇表 ##

- 仇恨言论词典 `dictionary/hatebase_dict.csv`
<br/>用于对推文进行采样，来自 https://www.hatebase.org/ 的原始词典。虽然该词典可以实现较高的召回率，但由于其中包含许多通常不会以冒犯或仇恨方式使用的词语（例如 yellow、oreo、bird），因此其误报率也较高
- 精简的n元语法-仇恨言论词典 `dictionary/refined_ngram_dict.csv`
<br/>包含一个精简的n元语法词典。从标注数据中提取了长度为1到4的n元语法集合，并计算了每个n元语法在被人工编码员判定为仇恨言论的推文中所占的比例,删除了不相关的术语。
- 积极词汇表 `dictionary/Postive-words.csv`
<br/>与积极观点/情感相关的词汇表，来自 https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English
- 消极词汇表 `dictionary/negative-word.csv`
<br/>与负面观点/情绪相关的词汇表。来自 https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English

## 数据集 ##

英文数据集存储在 `initial_dataset_en` 文件夹中，数据集来自发表的论文：《自动化仇恨言论检测与冒犯性语言问题》（“Automated Hate Speech Detection and the Problem of Offensive Language”）收录于 ICWSM
 - 内部包含一个开源的数据集，数据集包含 24,783 条推文，推文由 CrowdFlower 用户手动标记为 `hate_speech`、`offensive_language` 或 `neither`
 - __数据集定义__
<br/> `index` 推文的唯一标识符
<br/> `count` 给这条推文贴标签的 CrowdFlower 用户总数
<br/> `hate_speech` 在 CrowdFlower 上将该推文标记为包含或构成仇恨言论的用户数量
<br/> `offensive_language` 在 CrowdFlower 上将该推文标记为包含或构成冒犯性语言的用户数量
<br/> `neither` CrowdFlower 用户中，认为该推文既非仇恨言论也非冒犯性语言的人数
<br/> `class` CrowdFlower 用户给出的多数标签（0 代表仇恨言论，1 代表冒犯性语言，2 代表两者都不是）
<br/> `tweet` 推文（文本形式）
<br/> `clean_tweet` 转为小写并移除 URL、@提及及非字母数字字符，再合并多余空白后的推文文本

中文数据集存储在 `initial_dataset_cn` 文件夹中，数据集来自发表的论文：基于RoBERTa的中文仇恨言论侦测方法研究，收录于CCL 2023
 - 数据集中包含了17,430 条标注好的句子（train.csv：13,588 条，test.csv：3,842 条），覆盖种族，性别，地域等主题。其中，label 0 代表安全，label 1 代表仇恨言论

## 特征数据集生成脚本 ##

本项目按照“文本预处理 → 句法分析 → 特征抽取”的流程构建模型训练所需的多类特征
 -  `clean_tweets.py` 对原始标注数据 labeled_data.csv 中的推文进行清洗与规范化处理，生成包含 clean_tweet 的基础数据集 cleaned_tweets.csv
 -  `stanford_nlp.py` 调用 Stanford CoreNLP 对每条推文执行依存句法解析，并将解析结果按推文 index 保存为 dependency_dict.json
 -  `dependency_features.py` 在此基础上统计各类依存关系的出现次数，生成依存句法特征表 dependency_features.csv
 - `ngram_features.py` 基于清洗后的文本提取词级与字符级 n-gram 特征以及词级 TF-IDF 特征，分别输出 word_bigram_features.csv、char_bigram_features.csv 和 tfidf_features.csv
 - `sentiment_scores.py` 利用仇恨词典与情感词典计算每条推文中仇恨词、消极词和积极词的命中次数及其归一化比例，生成情感相关数值特征 sentiment_scores.csv
 - `tf-idf.py` 进一步基于仇恨词典计算仇恨词 TF-IDF 累加得分，用于衡量推文的仇恨强度。当前脚本将得分写入 try.csv，尚需整理为训练读取的 tfidf_scores.csv。训练还需要 labels.csv，运行模型时可使用仓库已提供的 test_feature_dataset

## 引用来源 ##

- 实现方案参考及README编写 https://github.com/aman-saha/hate-speech-detection/tree/master
- 词典 https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons
- 英文数据集 https://github.com/t-davidson/hate-speech-and-offensive-language
- 中文数据集 https://github.com/RXJ588/CHSD/tree/main
