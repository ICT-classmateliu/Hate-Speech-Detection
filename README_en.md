<!-- Created, written and maintained by @Classmateliu -->

<!-- Title and logo section -->
<div align="center">
<img src="image/logo_image.png" width="120"/>
</div>
<h1 align="center">Hate Speech Detection</h1>

<!-- Language toggle -->
<h4 align="center">
  English | <a href="README.md">简体中文</a>
</h4>
<div align="center">

<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ICT-classmateliu/Hate-Speech-Detection?label=Release&color=blue)](https://github.com/ICT-classmateliu/Hate-Speech-Detection/releases)
<!-- MIT License
Permissions: free to use, modify, redistribute, and commercial use.
Requirements: preserve original author copyright and license text.
Liability: author is not responsible for damages caused by use.
-->

<strong>Developed by @Classmateliu in Python using several libraries</strong>
</div>

## Project Overview ##

This project is a small-scale study of tasks related to text sentiment classification, using hate speech detection as a case study to explore text preprocessing, feature engineering, and the performance of different classification models.
<br/>__Research focus:__ Hate speech is commonly understood as any verbal, written, or behavioral expression targeting an individual or group based on identity. In other words, it refers to expressions that attack, demean, or incite hostility toward others based on religion, ethnicity, nationality, race, color, ancestry, gender, or other identity attributes.
<br/>__Key techniques:__ Hate speech detection is formulated as a text classification task. TF-IDF, n-gram, dependency syntax, and sentiment lexicon features are combined to train and compare multiple classifiers on labeled data, exploring the contribution of sentiment-related features to hate speech detection.
<br/>__Evaluation metrics:__ Accuracy, Precision (macro), Recall (macro), F1 (weighted / macro), and ROC/AUC (selected models)

## Features ##

The entry points are located in `main/`:

- `train_final_model.py`: trains a PyTorch MLP with class weights and saves model weights, scalers, feature columns, and a TF-IDF vectorizer to `main/artifacts/`.
- `app_gradio.py`: loads the saved artifacts and provides a Gradio demo with similar-sample matching and feature visualization.
- `hate_speech_detection_gpu.py`: compares baseline, full-feature, and ensemble models. PyTorch selects GPU or CPU based on CUDA availability, and XGBoost includes GPU / CPU configurations. Training time depends on hardware and environment.

The current training and demo workflow uses English data and precomputed features; the Chinese dataset is included as research material. The demo includes similar-sample retrieval and heuristic rules, so its output is intended for exploration and should not be treated as model evaluation on an independent test set.

The model configurations are summarized below:

| Model | Input features | Framework/Library | GPU Acceleration | Purpose | Training | Evaluation metrics |
| --- | --- | ---: | ---: | --- | --- | --- |
| Baseline MLP | Weighted TF-IDF | PyTorch | When CUDA is available | Simple reference model | 5-fold cross-validation | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| Gradient Boosting (GB) | Weighted TF-IDF | scikit-learn | No, CPU | Reference model | 3-fold cross-validation | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| Random Forest (RF) | Full features | scikit-learn | No, CPU | Reference model | 3-fold cross-validation | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| XGBoost | Full features | XGBoost | When CUDA is available | Ensemble | 5-fold on GPU / 3-fold on CPU | F1 (weighted), Accuracy, Precision (macro), Recall (macro) |
| PyTorch MLP | Full features | PyTorch | When CUDA is available | Main model | 5-fold cross-validation + held-out evaluation | F1 (weighted), Accuracy, Precision (macro), Recall (macro), ROC/AUC |
| Voting ensemble | Full features | scikit-learn | CPU for MLP/RF; XGBoost depends on configuration | Ensemble learning | sklearn MLP + RF + XGBoost (soft voting) | F1 (weighted / macro), Accuracy, Precision (macro), Recall (macro) |
| Stacking ensemble | Full features | mlxtend | CPU for MLP/RF; XGBoost depends on configuration | Ensemble learning | sklearn MLP + RF + XGBoost, 2-fold internal CV, LogisticRegression as meta-classifier | F1 (weighted / macro), Accuracy, Precision (macro), Recall (macro) |

## Usage ##

The main programs use `torch`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `mlxtend`, `matplotlib`, `gradio`, `plotly`, and `joblib`. Modules such as `re`, `string`, and `json` are part of the Python standard library and require no separate installation. The repository does not yet provide pinned dependencies; compatibility between saved artifacts and library versions needs local verification.

Run from the repository root with Python available in your terminal:

```powershell
python -m pip install torch numpy pandas scikit-learn xgboost mlxtend matplotlib gradio plotly joblib
python main/app_gradio.py
```

The repository includes `main/artifacts/` and `test_feature_dataset/`, so you can try the demo first. To retrain the model or compare models, run the corresponding command:

```powershell
python main/train_final_model.py
python main/hate_speech_detection_gpu.py
```

Retraining updates the model artifacts in `main/artifacts/`. GPU use requires a compatible local PyTorch, CUDA, and GPU setup.

Regenerating features also requires NLTK, `stanfordcorenlp`, Java, and a local Stanford CoreNLP installation. The feature scripts retain working-directory-relative input paths and legacy APIs such as `get_feature_names()` and `sklearn.cross_validation`. Before running them, update paths, dictionary filenames, the CoreNLP installation location, and the relevant APIs. The existing scripts do not yet generate every required training file through a single sequential workflow.

## Dictionaries and Lexicons ##

- Hate speech dictionary `dictionary/hatebase_dict.csv`
<br/>Used for sampling tweets, originating from https://www.hatebase.org/. Although this dictionary yields high recall, it contains many terms that are not typically used offensively or as hate speech (e.g., "yellow", "oreo", "bird"), so it also produces many false positives.
- Refined n-gram hate dictionary `dictionary/refined_ngram_dict.csv`
<br/>Contains a refined n-gram dictionary. It extracts n-grams of length 1 to 4 from annotated data and computes the proportion of each n-gram appearing in tweets labeled as hate speech by human annotators, removing irrelevant terms.
- Positive lexicon `dictionary/Postive-words.csv`
<br/>A lexicon associated with positive opinions/sentiments, from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English
- Negative lexicon `dictionary/negative-word.csv`
<br/>A lexicon associated with negative opinions/sentiments, from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English

## Datasets ##

English datasets are stored in the `initial_dataset_en` folder. The dataset originates from the paper: "Automated Hate Speech Detection and the Problem of Offensive Language" published in ICWSM.
 - The repository includes an open dataset with 24,783 tweets manually labeled by CrowdFlower users as `hate_speech`, `offensive_language`, or `neither`.
 - __Dataset schema__
<br/> `index` unique identifier for the tweet
<br/> `count` total number of CrowdFlower users who labeled this tweet
<br/> `hate_speech` number of CrowdFlower users who labeled the tweet as containing or constituting hate speech
<br/> `offensive_language` number of CrowdFlower users who labeled the tweet as containing offensive language
<br/> `neither` number of CrowdFlower users who labeled the tweet as neither hate speech nor offensive language
<br/> `class` majority label assigned by CrowdFlower users (0 = hate_speech, 1 = offensive_language, 2 = neither)
<br/> `tweet` the tweet text
<br/> `clean_tweet` tweet text after lowercasing, removing URLs, @mentions, and non-alphanumeric characters, and normalizing whitespace

Chinese datasets are stored in the `initial_dataset_cn` folder. The dataset originates from the paper: "Research on Chinese Hate Speech Detection based on RoBERTa", published in CCL 2023.
 - The dataset contains 17,430 labeled sentences (13,588 in train.csv and 3,842 in test.csv) covering topics such as race, gender, and region. Label 0 indicates safe, label 1 indicates hate speech.

## Feature dataset generation scripts ##

The project follows a "text preprocessing → syntactic analysis → feature extraction" process to construct multiple types of features required for model training.
- `clean_tweets.py` cleans and normalizes tweets from the original labeled data labeled_data.csv, generating a base dataset cleaned_tweets.csv containing clean_tweet
- `stanford_nlp.py` calls Stanford CoreNLP to perform dependency parsing on each tweet and saves the parsing results by tweet index as dependency_dict.json
- `dependency_features.py` further counts the occurrences of various dependency relations, generating a dependency syntactic feature table dependency_features.csv
- `ngram_features.py` extracts word-level and character-level n-gram features as well as word-level TF-IDF features based on cleaned text, outputting word_bigram_features.csv, char_bigram_features.csv, and tfidf_features.csv respectively
- `sentiment_scores.py` uses hate dictionaries and sentiment dictionaries to calculate hit counts and normalized ratios of hate words, negative words, and positive words in each tweet, generating sentiment-related numerical features sentiment_scores.csv
- `tf-idf.py` further calculates cumulative TF-IDF scores of hate words based on hate dictionaries to measure tweet hate intensity. The script currently writes scores to try.csv, which still needs to be adapted to the training input tfidf_scores.csv. Training also requires labels.csv; the supplied test_feature_dataset can be used to run the models

## References ##

- Implementation and README reference: https://github.com/aman-saha/hate-speech-detection/tree/master
- Lexicons: https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons
- English dataset: https://github.com/t-davidson/hate-speech-and-offensive-language
- Chinese dataset: https://github.com/RXJ588/CHSD/tree/main
