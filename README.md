# Sentiment Analysis

Sentiment analysis project for Web and text analytics course

## Report

The methods and results are presented in the [src/report.ipynb](https://github.com/francoisCub/sentiment_analysis/blob/main/src/report.ipynb) notebook.

## Environment

Install conda environment:
`conda env create -f environment.yml`

## Overview

We investigated different type of models for sentiment analysis.

- RNN, LSTM and GRU.
- Model with or without attention.
- Word level embeddings: Word2Vec, GloVe, FastText.
- Document level embeddings: word embedding average from scratch, GloVe word embedding average, sentence BERT and Word Mover Embedding.
- 2 datasets: IMDB and Yelp reviews.
