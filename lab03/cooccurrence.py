import re
from collections import Counter
from heapq import nsmallest

import numpy as np
from scipy import sparse

class CooccurrenceModel:

    WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*", re.IGNORECASE)
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


    def __init__(self, window_size=1):
        if isinstance(window_size, (bool, np.bool_)) or not isinstance(
            window_size, (int, np.integer)
        ) or window_size < 1:
            raise ValueError("window_size phải là số nguyên dương")
        self.window_size = int(window_size)
        self.vocabulary = []
        self.word_to_index = {}
        self.cooccurrence_matrix = None
        self.vector_norms = None


    def tokenize_sentence(self, text):
        return self.WORD_PATTERN.findall(text.lower())


    def document_to_sentences(self, document):
        if not isinstance(document, str):
            raise TypeError("Mỗi document phải là một chuỗi")
        parts = self.SENTENCE_PATTERN.split(document.replace("\r", "\n"))
        sentences = []
        for part in parts:
            tokens = self.tokenize_sentence(part)
            if tokens:
                sentences.append(tokens)
        return sentences
    

    def _iter_sentences(self, corpus):
        documents = [corpus] if isinstance(corpus, str) else corpus

        for document in documents:
            yield from self.document_to_sentences(document)


    def _get_word_index(self, word):
        if not isinstance(word, str):
            raise TypeError("Từ cần truy vấn phải là một chuỗi")
        word = word.lower()
        if word not in self.word_to_index:
            raise KeyError(f"Từ không có trong vocabulary: {word}")
        return self.word_to_index[word]


    def build_vocabulary(self, corpus, min_count=1, max_size=None):
        if isinstance(min_count, (bool, np.bool_)) or not isinstance(
            min_count, (int, np.integer)
        ) or min_count < 1:
            raise ValueError("min_count phải là số nguyên dương")
        if max_size is not None and (
            isinstance(max_size, (bool, np.bool_))
            or not isinstance(max_size, (int, np.integer))
            or max_size < 1
        ):
            raise ValueError("max_size phải là số nguyên dương hoặc None")
        token_counts = Counter()
        for tokens in self._iter_sentences(corpus):
            token_counts.update(tokens)
        words = [word for word, count in token_counts.items() if count >= min_count]
        words.sort(key=lambda word: (-token_counts[word], word))
        if max_size is not None:
            words = words[:max_size]
        self.vocabulary = sorted(words)
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.cooccurrence_matrix = None
        self.vector_norms = None
        return self.vocabulary


    def build_cooccurrence_matrix(self, corpus):
        vocab_size = len(self.vocabulary)
        pair_counts = Counter()
        for tokens in self._iter_sentences(corpus):
            token_indices = [self.word_to_index.get(token, -1) for token in tokens]
            for i, target_index in enumerate(token_indices):
                if target_index == -1:
                    continue
                start = max(0, i - self.window_size)
                end = min(len(token_indices),i + self.window_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    context_index = token_indices[j]
                    if context_index == -1:
                        continue
                    key = target_index * vocab_size + context_index
                    pair_counts[key] += 1
        if pair_counts:
            keys = np.fromiter(pair_counts.keys(), dtype=np.int64, count=len(pair_counts))
            values = np.fromiter(pair_counts.values(), dtype=np.int64, count=len(pair_counts))
            matrix = sparse.csr_matrix((values, (keys // vocab_size, keys % vocab_size)), shape=(vocab_size, vocab_size), dtype=np.int64)
        else:
            matrix = sparse.csr_matrix((vocab_size, vocab_size), dtype=np.int64)
        float_matrix = matrix.astype(np.float64)
        squared_norms = float_matrix.multiply(float_matrix).sum(axis=1)

        self.cooccurrence_matrix = matrix
        self.vector_norms = np.sqrt(
            np.asarray(squared_norms).ravel()
        )

        return self.cooccurrence_matrix


    def cosine_similarity(self, word1, word2):
        if self.cooccurrence_matrix is None or self.vector_norms is None:
            raise RuntimeError("Hãy gọi build_cooccurrence_matrix() trước")
        index1 = self._get_word_index(word1)
        index2 = self._get_word_index(word2)
        denominator = (self.vector_norms[index1]* self.vector_norms[index2])
        if denominator == 0:
            return 0.0
        vector1 = self.cooccurrence_matrix.getrow(index1).astype(np.float64)
        vector2 = self.cooccurrence_matrix.getrow(index2).astype(np.float64)
        dot_product = vector1.multiply(vector2).sum()
        return float(np.clip(dot_product / denominator, -1.0, 1.0))


    def most_similar(self, word, top_k=5, positive_only=False):
        if isinstance(top_k, (bool, np.bool_)) or not isinstance(
            top_k, (int, np.integer)
        ) or top_k < 1:
            raise ValueError("top_k phải là số nguyên dương")
        if self.cooccurrence_matrix is None or self.vector_norms is None:
            raise RuntimeError("Hãy gọi build_cooccurrence_matrix() trước")
        word_index = self._get_word_index(word)
        if self.vector_norms[word_index] == 0:
            return []
        target_vector = self.cooccurrence_matrix.getrow(word_index).astype(np.float64)
        dot_products = (self.cooccurrence_matrix @ target_vector.T).toarray().ravel()
        denominators = (self.vector_norms * self.vector_norms[word_index])
        scores = np.divide(dot_products, denominators, out=np.zeros_like(dot_products), where=denominators > 0)
        scores = np.clip(scores, -1.0, 1.0)

        candidates = (
            (token, float(scores[index]))
            for index, token in enumerate(self.vocabulary)
            if (
                index != word_index
                and self.vector_norms[index] > 0
                and (not positive_only or scores[index] > 0)
            )
        )

        return nsmallest(
            top_k,
            candidates,
            key=lambda item: (-item[1], item[0])
        )