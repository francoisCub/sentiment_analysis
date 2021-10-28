import torchtext
from typing import Tuple

def get_pretrained_embeddings(embedding="Glove", max_vectors=10000, dim=300, unk_token="<unk>", unk_index=0, cache=".vector_cache") -> Tuple[torchtext.vocab.Vocab, torchtext.vocab.Vectors]:
    """Returns pretrianed embeddings

    Args:
        embedding (str, optional): Desired embedding in ["Word2Vec", "GloVe", "FastText"]. Defaults to "Glove".
        max_vectors (int, optinal): maximum size of the vocabulary. Defaults to 10000.
        dim (int, optinal): size of a word embedding. Defaults to 300.
        unk_token (str, optional): token for unknown tokens. Defaults to "<unk>".
        unk_index (int, optional): index for unknown tokens. Defaults to 0.

    Returns:
        Tuple[Vocab, Vectors]: vocabulary and corresponding embedding vectors
    """
    # TODO: implement Word2Vec and FastText. https://pytorch.org/text/stable/vocab.html 
    if embedding == "Glove":
        vectors = torchtext.vocab.GloVe(max_vectors=max_vectors, cache=cache, dim=dim)
        vocab = torchtext.vocab.vocab(vectors.stoi)
        vocab.insert_token(unk_token, unk_index)
        vocab.set_default_index(vocab[unk_token])
    elif embedding == "Word2Vec":
        # TODO (search... https://clay-atlas.com/us/blog/2021/08/06/pytorch-en-use-nn-embedding-load-gensim-pre-trained-weights/ ?)
        # At least the vectors
        pass
    elif embedding == "FastText":
        # TODO (like glove)
        pass

    return vocab, vectors
