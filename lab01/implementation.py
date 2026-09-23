import re
from collections import defaultdict
import numpy as np
import unicodedata

class SimpleTfidf:
    TOKEN_PATTERN = re.compile(r"\b\w+\b")

    def tokenize(self, text):
        return self.TOKEN_PATTERN.findall(text.lower())

    def build_vocabulary(self, documents):
        vocabulary = set()
        for document in documents:
            vocabulary.update(self.tokenize(document))
        return sorted(vocabulary)

    def compute_counts(self, document, vocabulary):
        tokens = self.tokenize(document)
        token_counts = defaultdict(int)

        for token in tokens:
            token_counts[token] += 1

        return np.array([token_counts.get(term, 0) for term in vocabulary], dtype=float)

    def compute_sparse_counts(self, document, term_to_index):
        tokens = self.tokenize(document)
        token_counts = defaultdict(int)

        for token in tokens:
            token_counts[token] += 1

        return {
            term_to_index[term]: count
            for term, count in token_counts.items()
            if term in term_to_index
        }

    def compute_tf(self, counts):
        counts = np.asarray(counts, dtype=float)
        total = counts.sum()
        if total == 0:
            return np.zeros_like(counts)
        return counts / total

    def compute_df(self, documents, vocabulary):
        vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
        df = np.zeros(len(vocabulary), dtype=float)
        for document in documents:
            document_terms = set(self.tokenize(document))
            for term in document_terms:
                if term in vocabulary_index:
                    index = vocabulary_index[term]
                    df[index] += 1
        return df

    def compute_idf(self, df, number_of_documents):
        df = np.asarray(df, dtype=float)
        idf = np.zeros(len(df), dtype=float)
        for index, document_frequency in enumerate(df):
            if document_frequency > 0:
                idf[index] = np.log(number_of_documents / document_frequency)
        return idf

    def compute_tfidf(self, tf, idf):
        tf = np.asarray(tf, dtype=float)
        idf = np.asarray(idf, dtype=float)

        return tf * idf
    
    def cosine_similarity(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            return 0.0

        return float(np.dot(x, y) / (norm_x * norm_y))




class SimpleBPE:
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        num_merges=100,
        min_frequency=1,
        end_of_word="</w>",
        lowercase=True,
        unk_token="<unk>",
    ):
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.end_of_word = end_of_word
        self.lowercase = lowercase
        self.unk_token = unk_token

        self.merges = []
        self.merge_ranks = {}
        self.word_frequency = defaultdict(int)
        self.current_vocab = defaultdict(int)
        self.base_symbols = set()
        self.vocab = {}
        self.id_to_token = {}
        self.is_fitted = False

    def normalize_text(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        return " ".join(text.split())

    def _words_from_corpus(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]

        for document in corpus:
            if not isinstance(document, str):
                raise TypeError("corpus must contain only strings")
            normalized = self.normalize_text(document)
            yield from self.TOKEN_PATTERN.findall(normalized)

    def build_word_frequency(self, corpus):
        word_frequency = defaultdict(int)
        for word in self._words_from_corpus(corpus):
            word_frequency[word] += 1
        return word_frequency

    def initialize_vocab(self, word_frequency):
        current_vocab = defaultdict(int)
        base_symbols = set()

        for word, frequency in word_frequency.items():
            symbols = tuple(word) + (self.end_of_word,)
            current_vocab[symbols] += frequency
            base_symbols.update(symbols)
        return current_vocab, base_symbols

    @staticmethod
    def get_pair_statistics(current_vocab):
        pair_frequency = defaultdict(int)

        for symbols, frequency in current_vocab.items():
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                pair_frequency[pair] += frequency

        return pair_frequency

    @staticmethod
    def merge_sequence(symbols, pair):
        first, second = pair
        result = []
        index = 0

        while index < len(symbols):
            if (index + 1 < len(symbols) and symbols[index] == first and symbols[index + 1] == second):
                result.append(first + second)
                index += 2
            else:
                result.append(symbols[index])
                index += 1

        return tuple(result)

    def merge_vocab(self, current_vocab, pair):
        new_vocab = defaultdict(int)

        for symbols, frequency in current_vocab.items():
            merged_symbols = self.merge_sequence(symbols, pair)
            new_vocab[merged_symbols] += frequency

        return new_vocab

    @staticmethod
    def select_best_pair(pair_frequency):
        if not pair_frequency:
            return None, 0

        best_pair, best_count = max(
            pair_frequency.items(),
            key=lambda item: (item[1], item[0]),
        )
        return best_pair, best_count

    def train_bpe(self, corpus):
        self.merges = []
        self.merge_ranks = {}
        self.word_frequency = self.build_word_frequency(corpus)
        self.current_vocab, self.base_symbols = self.initialize_vocab(
            self.word_frequency
        )

        for _ in range(self.num_merges):
            pair_frequency = self.get_pair_statistics(self.current_vocab)
            best_pair, best_count = self.select_best_pair(pair_frequency)

            if best_pair is None or best_count < self.min_frequency:
                break

            self.merge_ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)
            self.current_vocab = self.merge_vocab(self.current_vocab, best_pair)

        self.build_token_vocabulary()
        self.is_fitted = True
        return self.merges, self.current_vocab, self.base_symbols

    def fit(self, corpus):
        self.train_bpe(corpus)
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() or train_bpe() before encoding text")

    def encode_word(self, word):
        self._check_fitted()
        if not isinstance(word, str):
            raise TypeError("word must be a string")

        tokens = list(word) + [self.end_of_word]

        while len(tokens) > 1:
            pairs = list(zip(tokens, tokens[1:]))
            candidate = min(pairs,key=lambda pair: self.merge_ranks.get(pair, float("inf")),)
            if candidate not in self.merge_ranks:
                break

            first, second = candidate
            merged_tokens = []
            index = 0

            while index < len(tokens):
                if (index + 1 < len(tokens) and tokens[index] == first and tokens[index + 1] == second):
                    merged_tokens.append(first + second)
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            tokens = merged_tokens
        return tokens

    def tokenize_text(self, text):
        self._check_fitted()
        normalized = self.normalize_text(text)
        tokens = []

        for word in self.TOKEN_PATTERN.findall(normalized):
            tokens.extend(self.encode_word(word))

        return tokens

    def build_token_vocabulary(self, tokens=None):
        if tokens is None:
            if not self.current_vocab:
                raise RuntimeError("Train the tokenizer before building its vocabulary")
            unique_tokens = {
                token
                for symbol_sequence in self.current_vocab
                for token in symbol_sequence
            }
        else:
            unique_tokens = set(tokens)
        if self.unk_token is not None:
            unique_tokens.add(self.unk_token)

        ordered_tokens = sorted(unique_tokens)
        self.vocab = {token: index for index, token in enumerate(ordered_tokens)}
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        return self.vocab

    def encode(self, text):
        tokens = self.tokenize_text(text)
        unk_id = self.vocab.get(self.unk_token)
        ids = []

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            elif unk_id is not None:
                ids.append(unk_id)
            else:
                raise ValueError(
                    f"Token {token!r} is not in the vocabulary and no unk token exists"
                )
        return ids

    def decode(self, ids):
        self._check_fitted()
        try:
            tokens = [self.id_to_token[index] for index in ids]
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {error.args[0]}") from error
        text = "".join(tokens)
        return text.replace(self.end_of_word, " ").strip()
