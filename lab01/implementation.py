import numpy as np
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vocabulary(documents):
    tokens = set()
    for doc in documents:
        tokens.update(re.findall(r'\b\w+\b', doc.lower()))
    return sorted(tokens)


def compute_counts(document, vocabulary):
    tokens = re.findall(r'\b\w+\b', document.lower())
    counts = np.array([tokens.count(term) for term in vocabulary])
    return counts


def compute_tf(counts):
    total = np.sum(counts)
    return counts / total


def compute_idf(documents, vocabulary):
    N = len(documents)
    idf = []
    for term in vocabulary:
        df = 0
        for doc in documents:
            tokens = re.findall(r'\b\w+\b', doc.lower())
            if term in tokens:
                df += 1
        idf_value = math.log(N / df)
        idf.append(idf_value)
    return idf


def compute_tfidf(tf, idf):
    return [tf[i] * idf[i] for i in range(len(tf))]


def cosine_similarity(x, y):
    dot = sum(x[i] * y[i] for i in range(len(x)))
    norm_x = sum(xi ** 2 for xi in x) ** 0.5
    norm_y = sum(yi ** 2 for yi in y) ** 0.5
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return dot / (norm_x * norm_y)


# ============================================================
# PHẦN 1: CHẠY THỬ + IN KẾT QUẢ STUDENT IMPLEMENTATION
# ============================================================
print("=" * 60)
print("STUDENT IMPLEMENTATION")
print("=" * 60)

documents = ["cat eats fish", "dog eats fish", "cat likes fish"]
vocab = build_vocabulary(documents)
print("Vocabulary:      ", vocab)

counts_d1 = compute_counts(documents[0], vocab)
print("Counts D1:       ", counts_d1)

tf_d1 = compute_tf(counts_d1)
print("TF D1:           ", np.round(tf_d1, 4))

idf = compute_idf(documents, vocab)
print("IDF:             ", np.round(idf, 4))

tfidf_d1 = compute_tfidf(tf_d1, idf)
print("TF-IDF D1:       ", np.round(tfidf_d1, 4))

sim = cosine_similarity([1, 1, 1], [1, 1, 0])
print("Cosine sim [1,1,1] vs [1,1,0]:", round(sim, 4))


# ============================================================
# PHẦN 2: UNIT TESTS (assert) — yêu cầu mục 8.4
# ============================================================
print("\n" + "=" * 60)
print("UNIT TESTS")
print("=" * 60)

assert vocab == ['cat', 'dog', 'eats', 'fish', 'likes'], "Sai vocabulary"
print("[PASS] build_vocabulary")

assert list(counts_d1) == [1, 0, 1, 1, 0], "Sai count vector D1"
print("[PASS] compute_counts")

assert abs(tf_d1[0] - 1/3) < 1e-9, "Sai tf(cat, D1)"
assert abs(sum(tf_d1) - 1.0) < 1e-9, "Tổng TF phải bằng 1"
print("[PASS] compute_tf")

assert abs(idf[3] - 0.0) < 1e-9, "idf(fish) phải = 0 vì df=N"
print("[PASS] compute_idf")

assert abs(tfidf_d1[3] - 0.0) < 1e-9, "tfidf(fish, D1) phải = 0"
print("[PASS] compute_tfidf")

assert abs(sim - 0.8165) < 1e-3, "Sai cosine similarity"
print("[PASS] cosine_similarity")

print("\nAll unit tests passed!")


# ============================================================
# PHẦN 3: SO SÁNH VỚI THƯ VIỆN (sklearn) — yêu cầu mục 8.5
# ============================================================
print("\n" + "=" * 60)
print("SO SÁNH: STUDENT vs SKLEARN (default settings)")
print("=" * 60)

vectorizer_default = TfidfVectorizer(token_pattern=r'\b\w+\b')
tfidf_default = vectorizer_default.fit_transform(documents)

print("Sklearn vocabulary:", list(vectorizer_default.get_feature_names_out()))
print("Sklearn TF-IDF D1: ", np.round(tfidf_default.toarray()[0], 4))
print("Student TF-IDF D1: ", np.round(tfidf_d1, 4))
print(">> Khác nhau do: sklearn mặc định dùng IDF smoothing + L2 normalization")


print("\n" + "=" * 60)
print("SO SÁNH: STUDENT vs SKLEARN (tắt smoothing + normalization)")
print("=" * 60)

vectorizer_matched = TfidfVectorizer(
    token_pattern=r'\b\w+\b',
    norm=None,          # tắt L2 normalization
    smooth_idf=False,   # dùng công thức idf = log(N/df) giống student
)
tfidf_matched = vectorizer_matched.fit_transform(documents)

print("Sklearn vocabulary:", list(vectorizer_matched.get_feature_names_out()))
print("Sklearn TF-IDF D1: ", np.round(tfidf_matched.toarray()[0], 4))
print("Student TF-IDF D1: ", np.round(tfidf_d1, 4))

is_match = np.allclose(tfidf_matched.toarray()[0], tfidf_d1, atol=1e-4)
print(">> Khớp nhau sau khi đồng bộ convention?", is_match)