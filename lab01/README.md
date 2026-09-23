# LAB 01 — Từ xử lý văn bản đến tìm kiếm tài liệu


## 1. Tổng quan

Lab này nghiên cứu ảnh hưởng của các quyết định tiền xử lý văn bản đến biểu
diễn tài liệu và chất lượng của hệ thống tìm kiếm. Quy trình thực hiện gồm:

```text
Theory
  → Calculation
  → Prediction
  → Experiment
  → Core implementation
  → Document search
  → Evaluation
  → Error analysis
  → Reflection
```

Các thí nghiệm sử dụng corpus gồm khoảng 30.000 documents. Mỗi document được
biểu diễn bằng vector TF-IDF thưa và kết quả tìm kiếm được xếp hạng bằng
cosine similarity.

## 2. Cấu trúc thư mục

```text
lab01/
├── README.md
├── calculations.md
├── prediction.md
├── implementation.py
├── experiments.ipynb
├── results.csv
└── reflection.md
```

| File | Nội dung |
|---|---|
| `README.md` | Mô tả bài lab, phương pháp và hướng dẫn chạy |
| `calculations.md` | Các bài tính tay về count vector, TF, IDF, TF-IDF và cosine similarity |
| `prediction.md` | Các prediction trước khi quan sát corpus |
| `implementation.py` | Cài đặt TF-IDF, cosine similarity, sparse counting và SimpleBPE |
| `experiments.ipynb` | Các thí nghiệm Part D–H và phân tích kết quả tìm kiếm |
| `results.csv` | Top-K retrieval results và các metric đánh giá |
| `reflection.md` | Reflection, limitations, failure cases và khai báo sử dụng AI |

## 3. Cài đặt cốt lõi

Class `SimpleTfidf` cài đặt các hàm chính của Part E:

- `build_vocabulary()`
- `compute_counts()`
- `compute_tf()`
- `compute_df()`
- `compute_idf()`
- `compute_tfidf()`
- `cosine_similarity()`
- `compute_sparse_counts()`

Các công thức được sử dụng:

\[
TF(t,d)=\frac{count(t,d)}{\sum_{t'}count(t',d)}
\]

\[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
\]

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

Implementation của sinh viên không sử dụng IDF smoothing. Part E cũng có unit
tests và so sánh với reference implementation sau khi thống nhất các
convention về TF-IDF.

## 4. Các preprocessing pipeline

### Pipeline A — Minimal

```text
Raw text → Lowercasing → Whitespace tokenization
```

Pipeline A giữ lại punctuation gắn với các token được phân tách bằng khoảng
trắng, do đó tạo ra vocabulary lớn.

### Pipeline B — Normalized

```text
Raw text → Lowercasing → Regex tokenization → Stopword handling
```

Stopword được phát hiện tự động dựa trên document frequency, với ngưỡng bằng
5% tổng số documents. Cách này làm giảm vocabulary size và số token trung
bình, nhưng có thể loại bỏ một số term hữu ích cho truy vấn.

### Pipeline C — Extended

```text
Raw text → Unicode/lowercase normalization → SimpleBPE subword tokenization
```

`SimpleBPE` được huấn luyện với:

```python
num_merges=100
min_frequency=2
lowercase=True
```

Subword tokenization làm giảm đáng kể vocabulary size, nhưng mỗi document được
biểu diễn bởi nhiều subword token hơn.

## 5. Hệ thống tìm kiếm tài liệu

Hệ thống tìm kiếm thực hiện theo pipeline:

```text
Documents
  → Sparse TF-IDF index
  → TF-IDF query vector
  → Cosine similarity
  → Ranking
  → Top-K documents
```

Sparse index lưu vocabulary, ánh xạ term-to-index, IDF, sparse document vectors
và document norms. Không xây dựng dense matrix cho toàn bộ corpus 30K để tránh
sử dụng quá nhiều bộ nhớ.

Các query được sử dụng gồm:

- `medical image classification`
- `transformer language model`
- `deep learning healthcare`
- `natural language processing`
- `computer vision medical imaging`

## 6. Đánh giá

Hệ thống được đánh giá bằng:

- Precision@5;
- Recall@5;
- Mean Reciprocal Rank (MRR).

Notebook hiện tại sử dụng pseudo-relevance labels dựa trên lexical overlap.
Do đó, các metric phản ánh mức độ nhất quán của lexical retrieval, chưa phản
ánh đầy đủ relevance về mặt ngữ nghĩa. Với báo cáo cuối cùng, nên kiểm tra
thủ công các document relevant của những query được chọn.

Cell export cuối notebook dùng để lưu retrieval results và evaluation metrics
vào file `results.csv`.

## 7. Kết quả thực nghiệm hiện tại

Các output thực nghiệm hiện đang được ghi nhận:

| Metric | Pipeline A | Pipeline B | Pipeline C |
|---|---:|---:|---:|
| Vocabulary size | 473,388 | 193,332 | 2,693 |
| Average tokens/document | 361.09 | 151.49 | 1,260.74 |
| Matrix sparsity | 0.999611 | 0.999484 | 0.959530 |
| OOV rate | 0.000000 | 0.000000 | 0.000000 |

Kết quả Part H hiện tại:

```text
P@5:      0.2400
Recall@5: 0.0095
MRR:      0.4667
```

Các giá trị trên cần được cập nhật sau khi chạy lại toàn bộ notebook, đặc biệt
sau khi thay đổi cách xây dựng stopword của Pipeline B hoặc evaluation query
set.

## 8. Hướng dẫn chạy lại

Cài đặt các thư viện cần thiết:

```bash
pip install numpy scikit-learn jupyter
```

Sau đó thực hiện:

1. Đặt file `c4-train.00000-of-01024-30K.json.gz` trong thư mục `lab01`.
2. Mở `experiments.ipynb` bằng Jupyter Notebook hoặc JupyterLab.
3. Chạy toàn bộ các cell theo thứ tự từ đầu đến cuối.
4. Chạy cell export cuối cùng để tạo `results.csv`.
5. Kiểm tra kết quả retrieval và cập nhật `reflection.md` bằng các metric và
   failure cases cuối cùng.

Đường dẫn dataset được khai báo ở phần đầu notebook và cần được cập nhật nếu
project được chuyển sang máy tính khác.

## 9. Hạn chế

TF-IDF là lexical representation, chủ yếu dựa trên sự trùng khớp của các
term. Phương pháp này không mô hình hóa trực tiếp từ đồng nghĩa, ngữ cảnh,
polysemy hoặc semantic equivalence. Ví dụ, document chứa từ `transformer` có
thể được trả về cho query về language model dù nội dung thực tế nói về thiết
bị điện.

Hạn chế này gợi ý việc nghiên cứu các phương pháp tiếp theo như BM25, query
expansion, word embeddings hoặc contextual transformer representations.

## 10. Khai báo sử dụng AI

AI được sử dụng để giải thích yêu cầu bài lab, làm rõ các convention của
TF-IDF, hỗ trợ chẩn đoán lỗi Python, đề xuất sparse-index optimization và tổ
chức tài liệu thí nghiệm. Implementation và kết quả được kiểm tra bằng unit
tests, reference implementation và output thực tế của notebook. Việc diễn
giải kết quả và reflection cuối cùng thuộc trách nhiệm của sinh viên.
