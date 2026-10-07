# LAB 03 — Word-context representation và Word2Vec

## 1. Mục tiêu

Bài lab khảo sát cách biểu diễn từ bằng vector đếm co-occurrence và embedding Word2Vec. Các thí nghiệm so sánh ảnh hưởng của context window, embedding dimension, đồng thời đánh giá similarity, analogy và ứng dụng tìm kiếm tài liệu.

## 2. Cấu trúc thư mục

Thư mục bài làm: `D:\Study\Study_Class\Semester_7\NLP\Lab\lab03`.

```text
Lab/
├── data/
│   └── c4-train.00000-of-01024-30K.json.gz
└── lab03/
    ├── README.md
    ├── cooccurrence.py
    ├── word_embedding.ipynb
    ├── results.csv
    ├── calculations.pdf
    ├── prediction.pdf
    ├── error_analysis.pdf
    ├── reflection.pdf
```

| File/thư mục | Nội dung |
|---|---|
| `cooccurrence.py` | Class `CooccurrenceModel`: tiền xử lý, xây vocabulary, đếm word-context, cosine similarity và tìm từ gần nhất. |
| `word_embedding.ipynb` | Đọc corpus, thực hiện các thí nghiệm, hiển thị bảng/biểu đồ, phân tích context và semantic search. |
| `results.csv` | Kết quả thống kê, similarity, analogy, retrieval và bằng chứng từ corpus. |
| `calculations.pdf` | Phần bài tập tính tay. |
| `prediction.pdf` | Phần dự đoán trước thí nghiệm. |
| `error_analysis.pdf` | Phần phân tích lỗi và kết quả bất ngờ. |
| `reflection.pdf` | Phần reflection về các cách biểu diễn từ. |


## 3. Dữ liệu và tiền xử lý

- Corpus: file C4 `c4-train.00000-of-01024-30K.json.gz`, dạng JSON Lines nén gzip; văn bản nằm trong trường `text`.
- Sử dụng **10.000 documents đầu tiên có text hợp lệ**, thay vì toàn bộ file.
- Chuyển thành chữ thường, tách token bằng regex và tách câu theo `.`, `!`, `?` hoặc xuống dòng.
- Giữ stopwords khi đếm co-occurrence và huấn luyện Word2Vec; lọc stopwords khi phân tích context và tìm kiếm.
- Không lemmatize: `patient` và `patients` là hai token riêng.
- Giữ Doc ID và số dòng JSONL để truy lại ví dụ trong corpus gốc.

Thống kê của cấu hình hiện tại:

| Thông tin | Giá trị |
|---|---:|
| Documents | 10.000 |
| Sentences | 207.954 |
| Tokens | 3.634.366 |
| Vocabulary trước lọc | 102.720 |
| Vocabulary co-occurrence | 5.000 |
| Vocabulary Word2Vec | 55.085 |

Co-occurrence dùng `min_count=5`, `max_size=5000`: chọn các từ đủ tần suất, lấy tối đa 5.000 từ phổ biến nhất, rồi sắp xếp vocabulary theo thứ tự chữ cái. Word2Vec dùng `min_count=2` và không áp dụng giới hạn 5.000 từ này.

## 4. Cài đặt và chạy

Môi trường đã dùng cho bài làm: **Python 3.10**, model đã lưu bằng **Gensim 4.4.0**. Cài các thư viện trong môi trường Python dùng làm kernel của notebook:

```powershell
python -m pip install numpy scipy pandas matplotlib scikit-learn "gensim==4.4.0" notebook ipykernel
```

Mở notebook bằng VS Code hoặc Jupyter Notebook. Với Jupyter:

```powershell
cd "D:\Study\Study_Class\Semester_7\NLP\Lab\lab03"
python -m notebook
```

1. Mở `word_embedding.ipynb` và chọn kernel của môi trường vừa cài thư viện.
2. Kiểm tra `LAB_DIR` và `DATA_PATH` trong cell đầu tiên. Khi chuyển bài sang máy khác, sửa hai đường dẫn cho phù hợp.
3. Chọn **Restart Kernel**, rồi **Run All** để chạy các cell từ trên xuống.
4. Xem bảng, biểu đồ và nhận xét ngay dưới từng thí nghiệm.
5. Cell cuối ghi kết quả vào `results.csv` khi `EXPORT_RESULTS=True`; chạy lại cell này sẽ ghi đè CSV bằng các bảng hiện tại.

`CACHE_MODELS=True` cho phép lưu và dùng lại model trong `models/`. Chữ ký cache xét corpus, tokenizer, cấu hình và phiên bản Gensim. Khi các thông tin này khớp, notebook đọc model đã lưu; khi chưa có cache tương ứng, notebook huấn luyện model mới. Thời gian báo trong bảng cache là thời gian huấn luyện gốc, không phải thời gian đọc file.

Có thể giảm `MAX_DOCUMENTS` để chạy thử. Khi thay đổi corpus hoặc tham số, cần chạy lại các cell và cập nhật những nhận xét có số liệu.

## 5. Các thí nghiệm

| Mục W3 | Nội dung thực hiện |
|---|---|
| 10 — Word-context representation | Kiểm tra cách đếm trên bốn câu nhỏ, sau đó xây ma trận C4 với window 1, 2, 5; so sánh vocabulary, shape, số ô khác 0, dung lượng và similarity. |
| 17 — Word2Vec | Huấn luyện baseline bằng Gensim; báo corpus, vocabulary và đầy đủ cấu hình/objective. |
| 18 — Inspect embeddings | Lấy Top-5 cho `doctor`, `hospital`, `patient`, `disease`, `computer`, `football`, `banana`; phân tích tần suất, context chung và ví dụ có Doc ID. |
| 19 — Context window | So sánh window 2, 5, 10; giữ dimension 100 và các tham số khác cố định. |
| 20 — Embedding dimension | So sánh 50, 100, 300 chiều: thời gian, dung lượng, similarity, analogy và downstream retrieval. |
| 21 — Word similarity | Xếp hạng sáu cặp từ và đối chiếu với kỳ vọng về quan hệ ngữ nghĩa. |
| 22 — Word analogy | Kiểm tra `king − man + woman`, `paris − france + italy`, `man − boy + girl`; báo Top-5 và Hit@1/5. |
| 24 — Semantic search | Tìm Top-5 documents bằng mean Word2Vec và TF-IDF cho bốn query chủ đề; đọc snippet và phân tích kết quả. |

### Co-occurrence

Ma trận có shape `|V| × |V|`, hàng là target words, cột là context words. `window_size=k` là bán kính: đếm tối đa k token bên trái và k token bên phải, bỏ chính vị trí target. Không đếm qua ranh giới câu hoặc document; token ngoài vocabulary vẫn giữ vị trí trong cửa sổ.

Ma trận được lưu dưới dạng sparse CSR để hạn chế chi phí bộ nhớ. Từ ngoài vocabulary được báo OOV/N/A, không thay bằng cosine 0. Với từ có trong vocabulary nhưng vector bằng 0, hàm cosine trả về 0.

### Cấu hình Word2Vec baseline

| Tham số | Giá trị |
|---|---|
| Training objective | CBOW + negative sampling |
| `vector_size` | 100 |
| `window` | 5 |
| `min_count` | 2 |
| `epochs` | 10 |
| `sg` / `negative` / `hs` | 0 / 5 / 0 |
| `sample` | 0.001 |
| `workers` / `seed` | 1 / 42 |
| `shrink_windows` | False |

Baseline 100 chiều/window 5 được dùng lại ở các thí nghiệm sau. Tổng cộng có **5 cấu hình khác nhau**: `(dimension, window) = (100, 5), (100, 2), (100, 10), (50, 5), (300, 5)`.

### Downstream task và semantic search

Downstream task chọn **20 câu thật từ 20 documents C4** làm query bằng seed cố định. Bỏ đúng câu được chọn khỏi biểu diễn của document nguồn, rồi tìm lại Doc ID nguồn trong 10.000 documents. Đánh giá bằng Hit@1, Hit@5, MRR và nDCG@5. Word2Vec vẫn học trên toàn bộ corpus đã chọn, nên đây là đánh giá in-sample ở bước học embedding.

Semantic search dùng các query `medical treatment`, `football match`, `computer software`, `river bank`. Vector query/document được tạo bằng trung bình word vectors rồi chuẩn hóa L2; TF-IDF dùng `TfidfTransformer` với `sublinear_tf=True`. Top-5 có Doc ID, dòng JSONL, cosine và snippet. Các query chủ đề chưa có nhãn relevance, nên phần này phân tích nội dung kết quả thay vì báo accuracy.

## 6. Kết quả chính

Các số liệu sau lấy từ `results.csv` của cấu hình hiện tại:

- Co-occurrence window 1/2/5 có lần lượt **962.037 / 1.801.117 / 3.534.798 ô khác 0**, cùng shape 5.000 × 5.000.
- Cả ba dimension đạt **2/3** ở Hit@1 và Hit@5 trên ba analogies. Câu `paris − france + italy` chưa tìm được `rome` trong Top-5.
- Downstream retrieval của Word2Vec đạt **2/20 query ở Top-1**, **3/20 ở Top-5** cho cả ba dimension. MRR 50/100/300 chiều lần lượt khoảng **0,1326 / 0,1344 / 0,1354**.
- TF-IDF đạt **9/20 ở Top-1**, **12/20 ở Top-5**, MRR khoảng **0,5075** trong phép thử tìm đúng document nguồn này.

Window hoặc dimension lớn hơn không bảo đảm chất lượng tăng. Cosine cao có thể phản ánh cùng chủ đề hoặc cùng context, không nhất thiết đồng nghĩa. Điểm cosine TF-IDF và Word2Vec thuộc các không gian biểu diễn khác nhau, nên không dùng độ lớn của hai điểm để kết luận phương pháp nào tốt hơn.

## 7. Kết quả xuất ra và giới hạn

`results.csv` hiện có **417 dòng dữ liệu**, phân loại bằng `record_type` và `experiment`. Các nhóm gồm thống kê co-occurrence, cấu hình/thời gian training, Top-5 neighbors, bằng chứng context, so sánh window/dimension, analogy, retrieval và semantic search.

Kết luận giới hạn ở 10.000 documents, một seed, ba analogies và 20 query downstream. Corpus web có nội dung lặp và nhiễu; tokenizer tách câu đơn giản có thể tách sai chữ viết tắt. Word2Vec chỉ có một vector cho mỗi token, còn mean document vectors làm mất thứ tự từ, nên chưa giải quyết đầy đủ đa nghĩa như `bank` hoặc ý định của query.


