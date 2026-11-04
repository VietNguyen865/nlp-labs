# LAB 07 — Transformer Architecture

Xây dựng Transformer Encoder block từ các thành phần cơ bản của PyTorch và ứng dụng vào phân loại cảm xúc đánh giá phim IMDb: negative / positive.

## Nội dung

- Multi-Head Self-Attention: tự tạo Q, K, V, tính scaled dot-product attention và ghép các heads.
- Sinusoidal positional encoding, FFN, residual connection và LayerNorm.
- Padding mask cho attention và mean pooling để loại token PAD.
- So sánh số heads, attention patterns, Transformer với LSTM, độ dài input và pooling strategy.
- Phân tích ảnh hưởng của thứ tự token, learning curves, dự đoán văn bản và các trường hợp đúng/sai.
- Benchmark tốc độ inference và training step để đối chiếu prediction 6.

Phần core không sử dụng `TransformerEncoderLayer`.

## Cấu trúc bài nộp

```text
lab07/
├── README.md
├── requirements.txt
├── calculations.pdf
├── prediction.pdf
├── error_analysis.pdf
├── reflection.pdf
├── transformer_block.py
├── positional_encoding.py
├── transformer_classification.ipynb
├── results.csv
├── positional_encoding.png
├── attention_heads.png
├── learning_curves.png
├── confusion_matrix.png
└── outputs/
    └── full/                 # Bảng CSV, hình và cấu hình thực nghiệm bổ sung
```

Các phần trình bày viết tay được lưu dưới dạng PDF:
[Calculations](calculations.pdf), [Prediction](prediction.pdf),
[Error analysis](error_analysis.pdf), [Reflection](reflection.pdf).

## Dữ liệu

Sử dụng IMDb Dataset gồm 50.000 đánh giá, hai nhãn `positive` và `negative`.
Sau khi chuẩn hóa thẻ xuống dòng, khoảng trắng và loại nội dung trùng, còn 49.578 review.

| Tập | Số review |
|---|---:|
| Train | 29.746 |
| Validation | 9.916 |
| Test | 9.916 |

Chia tập phân tầng theo tỷ lệ 60/20/20, seed 2026. Loại trùng trước split; vocabulary chỉ được xây dựng từ train.

Dữ liệu không được đưa vào repository. Chuẩn bị `archive.zip`, bên trong có `IMDB Dataset.csv` với các cột `review`, `sentiment`, tại đường dẫn:

```text
nlp-labs/
├── data/
│   └── archive.zip
└── lab07/
```

## Cách chạy

Yêu cầu Python 3.10 trở lên. Từ thư mục `lab07`, cài thư viện và mở Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter lab transformer_classification.ipynb
```

Chọn kernel đã cài thư viện, sau đó dùng `Restart Kernel` → `Run All`.
Kiểm tra `LAB_DIR` và `DATA_PATH` ở cell cấu hình nếu chạy tại vị trí khác. Notebook tự dùng CUDA nếu PyTorch nhận diện được GPU, còn lại sử dụng CPU.

Notebook đã lưu output của lần chạy `full`. Có thể đọc các bảng và hình trực tiếp mà không cần huấn luyện lại.

## Cấu hình thực nghiệm

| Tham số | Giá trị |
|---|---|
| Seed | 2026 |
| Vocabulary tối đa | 20.000 |
| Embedding / d_model | 128 |
| d_ff | 256 |
| Số encoder blocks | 1 |
| Batch size | 64 |
| Epochs | 10 |
| Optimizer | Adam, learning rate 0,001 |
| Dropout | 0,1 |
| Loss | BCEWithLogitsLoss |
| Gradient clipping | 1,0 |
| Ngưỡng phân loại | 0,5 |

Checkpoint được chọn theo validation loss thấp nhất. Model tham chiếu dùng 4 heads, 128 token nội dung, positional encoding và masked mean pooling; không chọn cấu hình dựa trên test score.

`full` dùng toàn bộ dữ liệu sau làm sạch. `quick` dùng 3.000 train / 1.000 validation / 1.000 test và vocabulary tối đa 8.000 để kiểm tra pipeline. Số epochs của cả hai chế độ tuân theo biến `EPOCHS` trong notebook, hiện là 10. Chỉ kết quả `full` được đưa vào bài nộp này.

## Các thực nghiệm

| Phần | Nội dung |
|---|---|
| Experiment 1 | Có/không PE; hoán vị “I love NLP” và “dog bites man”; so sánh representation. |
| Experiment 2 | Giữ d_model=128, thử 1/2/4/8 heads. |
| Experiment 3 | Heatmap từng head, entropy và cosine similarity giữa các heads. |
| Experiment 4 | Transformer và LSTM trên cùng split, vocabulary và ngân sách epochs. |
| Experiment 5 | Max length 32/64/128/256; token retention, chi phí scores và chất lượng phân loại. |
| Experiment 6 | Masked mean pooling và token CLS được học từ đầu. |
| Bổ sung | Ablation PE, shuffle sau truncation, benchmark tốc độ tại 8–256 token. |
| Application | Câu ngắn, dài, phủ định, tương phản và sentiment mơ hồ. |
| Error analysis | 3 dự đoán đúng, 3 dự đoán sai và đối chiếu attention. |

## Kết quả chính

Kết quả test của lần chạy full, seed 2026:

| Cấu hình | Accuracy | F1 |
|---|---:|---:|
| Transformer, 4 heads, 128 tokens, mean pooling | 0,8463 | 0,8509 |
| LSTM, 128 tokens, last hidden state | 0,8320 | 0,8316 |
| Transformer, 4 heads, 256 tokens, mean pooling | 0,8796 | 0,8787 |
| Transformer, 4 heads, 128 tokens, CLS pooling | 0,8256 | 0,8221 |

Các bảng đầy đủ nằm trong [results.csv](results.csv) và [outputs/full](outputs/full).
Đây là kết quả một seed; chưa đủ để kết luận một cấu hình luôn tốt hơn.

Tăng từ 4 lên 8 heads không cải thiện accuracy trong lần chạy này.
Tăng input từ 128 lên 256 token giúp giữ thêm ngữ cảnh, đồng thời làm số phần tử attention scores tăng gấp 4.
Model không PE kết hợp mean pooling giữ nguyên dự đoán khi hoán vị các token đã được giữ lại.

Benchmark prediction 6 cho thấy Transformer nhanh hơn LSTM tại các cấu hình CPU đã thử; chưa tìm thấy phản ví dụ để xác nhận dự đoán “Transformer không luôn nhanh hơn RNN”. Kết quả không đại diện mọi phần cứng hay sequence length.

## Hình và bảng bổ sung

- [Positional encoding](positional_encoding.png)
- [Attention heads](attention_heads.png)
- [Learning curves](learning_curves.png)
- [Confusion matrix](confusion_matrix.png)
- [So sánh các thực nghiệm trên test](outputs/full/test_comparisons.png)
- [Phân tích attention của một dự đoán sai](outputs/full/error_analysis_attention.png)
- [Benchmark prediction 6](outputs/full/prediction6_speed.png)
- [Cấu hình lần chạy](outputs/full/run_config.json)

Attention visualization chỉ mô tả phân phối attention weights; không chứng minh nguyên nhân model đưa ra dự đoán.

Các checkpoint, dự đoán từng mẫu, dữ liệu thô và kết quả quick được giữ cục bộ để repository gọn hơn. Chạy notebook để tạo lại các file này.
