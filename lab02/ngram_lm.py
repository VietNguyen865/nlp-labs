"""A small, self-contained n-gram language model used by experiments.ipynb.

The model deliberately expects tokenized sentences. Text normalization and
tokenization belong in the experiment notebook so that preprocessing choices
remain explicit and reproducible.
"""

from collections import Counter
from heapq import nsmallest
import math


class NGramLanguageModel:

    SUPPORTED_ORDERS = (1, 2, 3)
    SUPPORTED_SMOOTHING = ("mle", "laplace")

    def __init__(self, n, smoothing="mle", min_count=1, unk_token="<unk>"):
        if n not in self.SUPPORTED_ORDERS:
            raise ValueError("n must be 1, 2 or 3")
        if not isinstance(min_count, int) or min_count < 1:
            raise ValueError("min_count must be a positive integer")

        self.n = n
        self.smoothing = self._check_smoothing(smoothing)
        self.min_count = min_count
        self.unk_token = str(unk_token)

        self.vocabulary = set()
        self.total_tokens = 0

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        self.bigram_context_counts = Counter()
        self.trigram_context_counts = Counter()

        self.unigram_probabilities = {}
        self.bigram_probabilities = {}
        self.trigram_probabilities = {}

        self.is_fitted = False

    @classmethod
    def _check_smoothing(cls, smoothing):
        smoothing = str(smoothing).lower()
        if smoothing not in cls.SUPPORTED_SMOOTHING:
            raise ValueError('smoothing must be "mle" or "laplace"')
        return smoothing

    @classmethod
    def _check_order(cls, order):
        if order not in cls.SUPPORTED_ORDERS:
            raise ValueError("order must be 1, 2 or 3")
        return order

    @staticmethod
    def _check_tokenized_sentence(sentence):
        if isinstance(sentence, str):
            raise TypeError(
                "The model expects a tokenized sentence, not a raw string"
            )

        tokens = list(sentence)
        if any(not isinstance(token, str) for token in tokens):
            raise TypeError("Every token must be a string")
        return tokens

    def _require_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit(corpus) before using the model")

    def _map_token(self, token):
        return token if token in self.vocabulary else self.unk_token

    def _resolve_order(self, order):
        requested_order = self.n if order is None else self._check_order(order)
        if requested_order > self.n:
            raise ValueError(
                f"order={requested_order} is unavailable in an n={self.n} model"
            )
        return requested_order

    def build_vocabulary(self, corpus):
        token_counts = Counter()
        for sentence in corpus:
            token_counts.update(self._check_tokenized_sentence(sentence))
        self.vocabulary = {word for word, count in token_counts.items() if count >= self.min_count}
        self.vocabulary.add(self.unk_token)
        return token_counts

    def count_ngrams(self, corpus):
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_context_counts.clear()
        self.trigram_context_counts.clear()

        for sentence in corpus:
            raw_tokens = self._check_tokenized_sentence(sentence)
            if not raw_tokens:
                continue

            tokens = [self._map_token(token) for token in raw_tokens]
            self.unigram_counts.update((word,) for word in tokens)

            if self.n >= 2:
                for index in range(len(tokens) - 1):
                    context = (tokens[index],)
                    bigram = context + (tokens[index + 1],)
                    self.bigram_counts[bigram] += 1
                    self.bigram_context_counts[context] += 1

            if self.n >= 3:
                for index in range(len(tokens) - 2):
                    context = (tokens[index], tokens[index + 1])
                    trigram = context + (tokens[index + 2],)
                    self.trigram_counts[trigram] += 1
                    self.trigram_context_counts[context] += 1

        self.total_tokens = sum(self.unigram_counts.values())

    def fit(self, corpus):
        if isinstance(corpus, str):
            raise TypeError("corpus must be tokenized in experiments.ipynb before fit()")
        if iter(corpus) is corpus:
            corpus = list(corpus)

        self.build_vocabulary(corpus)
        self.count_ngrams(corpus)

        if self.total_tokens == 0:
            raise ValueError("The corpus is empty")

        self.unigram_probabilities.clear()
        self.bigram_probabilities.clear()
        self.trigram_probabilities.clear()
        self.is_fitted = True
        return self

    def _event_probability(self, order, context, word, smoothing):
        vocabulary_size = len(self.vocabulary)

        if order == 1:
            numerator = self.unigram_counts.get((word,), 0)
            denominator = self.total_tokens
        elif order == 2:
            numerator = self.bigram_counts.get(context + (word,), 0)
            denominator = self.bigram_context_counts.get(context, 0)
        else:
            numerator = self.trigram_counts.get(context + (word,), 0)
            denominator = self.trigram_context_counts.get(context, 0)

        if smoothing == "laplace":
            return (numerator + 1) / (denominator + vocabulary_size)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def probability(self, context, word, smoothing=None, order=None):
        self._require_fitted()
        smoothing = self._check_smoothing(smoothing or self.smoothing)
        requested_order = self._resolve_order(order)

        context_tokens = [] if context is None else self._check_tokenized_sentence(context)
        effective_order = min(requested_order, len(context_tokens) + 1)
        mapped_word = self._map_token(word)

        if effective_order == 1:
            mapped_context = ()
        else:
            mapped_context = tuple(self._map_token(token) for token in context_tokens[-(effective_order - 1):])

        return self._event_probability(
            effective_order,
            mapped_context,
            mapped_word,
            smoothing,
        )

    def context_count(self, context, order=None):
        self._require_fitted()
        requested_order = self._resolve_order(order)

        if requested_order == 1:
            return self.total_tokens

        context_tokens = self._check_tokenized_sentence(context)
        if len(context_tokens) < requested_order - 1:
            return 0

        mapped_context = tuple(
            self._map_token(token)
            for token in context_tokens[-(requested_order - 1):]
        )

        if requested_order == 2:
            return self.bigram_context_counts.get(mapped_context, 0)
        return self.trigram_context_counts.get(mapped_context, 0)

    def train_unigram(self, smoothing=None):
        self._require_fitted()
        smoothing = self._check_smoothing(smoothing or self.smoothing)
        self.unigram_probabilities = {
            word: self._event_probability(1, (), word, smoothing)
            for word in self.vocabulary
        }
        return self.unigram_probabilities

    def train_bigram(self, smoothing=None):
        self._require_fitted()
        if self.n < 2:
            return {}
        smoothing = self._check_smoothing(smoothing or self.smoothing)
        self.bigram_probabilities = {
            bigram: self._event_probability(2, (bigram[0],), bigram[1], smoothing,) for bigram in self.bigram_counts}
        return self.bigram_probabilities

    def train_trigram(self, smoothing=None):
        self._require_fitted()
        if self.n < 3:
            return {}
        smoothing = self._check_smoothing(smoothing or self.smoothing)
        self.trigram_probabilities = {
            trigram: self._event_probability(3, trigram[:2], trigram[2], smoothing,) for trigram in self.trigram_counts}
        return self.trigram_probabilities

    def sentence_log_probability(self, sentence, smoothing=None, order=None):
        self._require_fitted()
        tokens = self._check_tokenized_sentence(sentence)
        if not tokens:
            return 0.0

        requested_order = self._resolve_order(order)
        smoothing = self._check_smoothing(smoothing or self.smoothing)
        result = 0.0

        for index, word in enumerate(tokens):
            context_start = max(0, index - requested_order + 1)
            word_probability = self.probability(tokens[context_start:index], word, smoothing=smoothing, order=requested_order,)
            if word_probability == 0:
                return -math.inf
            result += math.log(word_probability)

        return result

    def sentence_probability(self, sentence, smoothing=None, order=None):
        log_probability = self.sentence_log_probability(
            sentence,
            smoothing=smoothing,
            order=order,
        )
        if log_probability == -math.inf:
            return 0.0
        return math.exp(log_probability)

    def perplexity(self, corpus, smoothing=None, order=None):
        self._require_fitted()
        requested_order = self._resolve_order(order)
        smoothing = self._check_smoothing(smoothing or self.smoothing)

        total_log_probability = 0.0
        evaluation_token_count = 0
        vocabulary_size = len(self.vocabulary)

        unigram_counts = self.unigram_counts
        bigram_counts = self.bigram_counts
        trigram_counts = self.trigram_counts
        bigram_context_counts = self.bigram_context_counts
        trigram_context_counts = self.trigram_context_counts
        training_token_count = self.total_tokens
        vocabulary = self.vocabulary
        unk_token = self.unk_token
        use_laplace = smoothing == "laplace"

        for sentence in corpus:
            raw_tokens = self._check_tokenized_sentence(sentence)
            if not raw_tokens:
                continue

            tokens = [
                token if token in vocabulary else unk_token
                for token in raw_tokens
            ]
            evaluation_token_count += len(tokens)

            for index, word in enumerate(tokens):
                effective_order = min(requested_order, index + 1)

                if effective_order == 1:
                    numerator = unigram_counts.get((word,), 0)
                    denominator = training_token_count
                elif effective_order == 2:
                    context = (tokens[index - 1],)
                    numerator = bigram_counts.get(context + (word,), 0)
                    denominator = bigram_context_counts.get(context, 0)
                else:
                    context = (tokens[index - 2], tokens[index - 1])
                    numerator = trigram_counts.get(context + (word,), 0)
                    denominator = trigram_context_counts.get(context, 0)

                if use_laplace:
                    numerator += 1
                    denominator += vocabulary_size
                elif numerator == 0 or denominator == 0:
                    return math.inf

                total_log_probability += math.log(numerator / denominator)

        if evaluation_token_count == 0:
            return math.inf

        return math.exp(-total_log_probability / evaluation_token_count)

    def next_word_distribution(
        self,
        context,
        top_k=None,
        smoothing=None,
        order=None,
        backoff=False,
        include_unk=False,
    ):
        self._require_fitted()
        context_tokens = self._check_tokenized_sentence(context)
        requested_order = self._resolve_order(order)
        smoothing = self._check_smoothing(smoothing or self.smoothing)

        effective_order = min(requested_order, len(context_tokens) + 1)
        if backoff:
            while (
                effective_order > 1
                and self.context_count(context_tokens, order=effective_order) == 0
            ):
                effective_order -= 1

        if effective_order == 1:
            mapped_context = ()
        else:
            mapped_context = tuple(
                self._map_token(token)
                for token in context_tokens[-(effective_order - 1):]
            )

        candidates = (word for word in self.vocabulary if include_unk or word != self.unk_token)
        scored_words = ((word,self._event_probability(effective_order, mapped_context, word, smoothing,),) for word in candidates)

        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError("top_k must be a positive integer or None")
            ranked = nsmallest(top_k, scored_words, key=lambda item: (-item[1], item[0]),)
        else:
            ranked = sorted(scored_words, key=lambda item: (-item[1], item[0]),)
        return dict(ranked)
