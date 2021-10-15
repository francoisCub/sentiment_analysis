import torchtext

def get_pretrained_embeddings(embedding="Glove", max_vectors=10000, unk_token="<unk>", unk_index=0, cache="../.vector_cache") -> Tuple[Vocab, Vectors]:
    """Returns pretrianed embeddings

    Args:
        embedding (str, optional): Desired embedding in ["Word2Vec", "GloVe", "FastText"]. Defaults to "Glove".
        max_vectors (int, optinal): maximum size of the vocabulary. Defaults to 10000.
        unk_token (str, optional): token for unknown tokens. Defaults to "<unk>".
        unk_index (int, optional): index for unknown tokens. Defaults to 0.

    Returns:
        Tuple[Vocab, Vectors]: vocabulary and corresponding embedding vectors
    """
    # TODO: implement Word2Vec and FastText
    glove_vec = torchtext.vocab.GloVe(max_vectors=max_vectors, cache=cache)
    glove_vocab = torchtext.vocab.vocab(glove_vec.stoi)
    glove_vocab.insert_token(unk_token, unk_index)
    glove_vocab.set_default_index(glove_vocab[unk_token])

    return glove_vocab, glove_vec
