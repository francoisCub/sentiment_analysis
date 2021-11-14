# sentiment_analysis

Sentiment analysis project for Web and text analytics course

## Report

The results are presented in the [src/report.ipynb](https://github.com/francoisCub/sentiment_analysis/blob/main/src/report.ipynb) notebook.

## Environment

Install conda environment:
conda env create -f environment.yml

## Goals

Implement an RNN with attention for sentiment analysis. As inspiration, refer to the study of Letarte et al. (https://www.aclweb.org/anthology/W18-5429/) or that of Ambartsoumian & Popowich (https://www.aclweb.org/anthology/W18-6219/).
Experiments:

- Investigate the performance using LSTM vs. GRU
- Investigate the performance using various word embeddings: Word2Vec, GloVe, FastText.
- Investigate the performance using document level embeddings. See Wu et al. (https://www.aclweb.org/anthology/D18-1482/)
- Investigate the performance without attention
- (Report & interpret all scores)

## Taks

- Report
- Clean code
