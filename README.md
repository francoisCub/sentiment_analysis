# Sentiment Analysis

Sentiment analysis project for the 2021-2022 Web and text analytics course at University of Liège.

## Report

The methods and results are presented in the [src/report.ipynb](https://github.com/francoisCub/sentiment_analysis/blob/main/src/report.ipynb) notebook.

Raw results are stored in [src/results](https://github.com/francoisCub/sentiment_analysis/blob/main/src/results)

## Environment

Install conda environment:
`conda env create -f environment.yml`
`conda activate sentiment_analysis`

## Overview

We investigated different type of models for sentiment analysis.

- RNN, LSTM and GRU.
- Model with or without attention.
- Word level embeddings: Word2Vec, GloVe, FastText.
- Document level embeddings: word embedding average from scratch, GloVe word embedding average, sentence BERT and Word Mover Embedding.
- 2 datasets: IMDB and Yelp reviews.

## Usage

The experiments can directly be run via the corresponding notebooks in the src folder. Note that document level embedding have been precomputed but they can be re-computed via the encode.ipynb notebook in src/models.

Required data will be automatically downloaded and can take several gigabytes of storage.

## Authors

- [Francois Cubelier](https://github.com/francoisCub)
- [Lisa Bueres](https://github.com/Lisa-Byd)
- [Romain Charles](https://github.com/romaincharles3001)
