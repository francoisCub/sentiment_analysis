from collections import OrderedDict
from os.path import exists
from typing import Tuple

import gensim.downloader as api
import torchtext
from gensim.models import KeyedVectors
from torch import FloatTensor


def get_pretrained_embeddings(embedding="Glove", max_vectors=10000, dim=300, unk_token="<unk>", unk_index=0, cache=".vector_cache") -> Tuple[torchtext.vocab.Vocab, torchtext.vocab.Vectors]:
    """Returns pretrained embeddings

    Args:
        embedding (str, optional): Desired embedding in ["Word2Vec", "GloVe", "FastText"]. Defaults to "Glove".
        max_vectors (int, optinal): maximum size of the vocabulary. Defaults to 10000.
        dim (int, optinal): size of a word embedding. Defaults to 300.
        unk_token (str, optional): token for unknown tokens. Defaults to "<unk>".
        unk_index (int, optional): index for unknown tokens. Defaults to 0.

    Returns:
        Tuple[Vocab, Tensor]: vocabulary and corresponding embedding vectors
    """

    if embedding == "Glove":
        vectors = torchtext.vocab.GloVe(max_vectors=max_vectors, cache=cache, dim=dim)
        vocab = torchtext.vocab.vocab(vectors.stoi)
        vocab.insert_token(unk_token, unk_index)
        vocab.set_default_index(vocab[unk_token])
        vectors = vectors.vectors
    elif embedding == "Word2Vec":
        if not exists(cache+'/word2vec-google-news-300'):
            vectors_first = (api.load('word2vec-google-news-300'))
            vectors_first.save(cache+'/word2vec-google-news-300')
        model = KeyedVectors.load(cache+'/word2vec-google-news-300')
        vectors = FloatTensor(model.vectors)
        vectors = vectors[0:max_vectors]
        ordered_dict = OrderedDict(model.key_to_index)
        while len(ordered_dict) > max_vectors:
            ordered_dict.popitem()
        vocab = torchtext.vocab.vocab(ordered_dict)
        vocab.insert_token(unk_token, unk_index)
        vocab.set_default_index(vocab[unk_token])
    elif embedding == "FastText":
        vectors = torchtext.vocab.FastText(max_vectors=max_vectors, cache=cache)
        vocab = torchtext.vocab.vocab(vectors.stoi)
        vocab.insert_token(unk_token, unk_index)
        vocab.set_default_index(vocab[unk_token])
        vectors = vectors.vectors

    return vocab, vectors
