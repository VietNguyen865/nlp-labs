# Lab 02 - N-gram Language Models

Bài thực hành xây dựng language model unigram, bigram và trigram từ đầu bằng Python. Notebook thống kê corpus, so sánh Maximum Likelihood Estimation (MLE) với Laplace smoothing, tính perplexity và minh họa dự đoán từ tiếp theo, xếp hạng câu, phân tích lỗi.

## Nội dung thư mục

| Tệp | Nội dung |
| --- | --- |
| [`ngram_lm.py`](ngram_lm.py) | Lớp `NGramLanguageModel`: đếm n-gram, tính xác suất, log probability, perplexity và phân phối từ tiếp theo. |
| [`experiments.ipynb`](experiments.ipynb) | Tiền xử lý, chia dữ liệu, các thí nghiệm, bảng kết quả, biểu đồ và ứng dụng. Notebook đã lưu output từ lần chạy thực nghiệm. |
| [`results.csv`](results.csv) | Kết quả perplexity của Experiment 2/3 và bảng next-word prediction. |
| [`calculations.pdf`](calculations.pdf) | Bài tính tay về unigram, bigram, xác suất câu, smoothing và perplexity. |
| [`prediction.pdf`](prediction.pdf) | Dự đoán trước khi chạy thực nghiệm. |
| [`error_analysis.pdf`](error_analysis.pdf) | Phân tích các trường hợp dự đoán đúng và sai. |
| [`reflection.pdf`](reflection.pdf) | Câu trả lời phần Reflection. |

## Dữ liệu và môi trường

Notebook sử dụng tệp C4 gồm 30.000 documents do môn học cung cấp. Đặt tệp tại:

```text
Lab/
├── data/
│   └── c4-train.00000-of-01024-30K.json.gz
└── lab02/
    ├── experiments.ipynb
    └── ngram_lm.py
```

Dataset `.gz` không được đưa lên GitHub vì repository loại trừ định dạng này trong `.gitignore`. Để xem code và output đã lưu trong notebook, không cần tải lại dataset; để chạy lại toàn bộ thí nghiệm thì cần dataset.

Yêu cầu Python 3.10 trở lên, Jupyter (hoặc VS Code với Python/Jupyter extensions), `numpy`, `pandas` và `matplotlib`. Có thể cài các thư viện Python bằng:

```powershell
python -m pip install numpy pandas matplotlib ipykernel
```

## Chạy lại thí nghiệm trong VS Code

1. Mở thư mục `Lab` trong VS Code và mở `lab02/experiments.ipynb`.
2. Ở code cell đầu tiên, sửa `LAB_DIR` thành đường dẫn tuyệt đối đến thư mục `lab02` trên máy của bạn. `DATA_PATH` được tạo từ `LAB_DIR.parent / "data" / ...`; kiểm tra tệp dữ liệu tồn tại ở vị trí đó.
3. Chọn Python kernel có các thư viện ở trên, sau đó chọn **Restart Kernel** và **Run All**.
4. Xem bảng và biểu đồ trong notebook. Cell cuối xuất kết quả gộp ra `lab02/results.csv`; Experiment 2 và 3 còn xuất riêng `experiment_2_results.csv` và `experiment_3_results.csv` khi chạy lại.

Lưu ý: `LAB_DIR` hiện được đặt theo đường dẫn trên máy thực hiện thí nghiệm. Nếu clone repository sang máy khác mà không sửa biến này, notebook sẽ không tìm thấy dataset hoặc ghi kết quả sai vị trí.

## Thiết kế thí nghiệm

- Chuyển văn bản thành câu và token trong notebook; `NGramLanguageModel` chỉ nhận câu đã tokenize.
- Chia train/validation/test theo **document** với tỉ lệ 80/10/10 và `random seed = 42`, tránh các câu của cùng document xuất hiện ở nhiều tập.
- Fit một trigram model trên **training set**. Bộ đếm của model cũng hỗ trợ đánh giá unigram và bigram, vì vậy không cần huấn luyện lại ba model lớn.
- Experiment 2 so sánh Bigram/Trigram với MLE và Laplace; Experiment 3 so sánh Unigram/Bigram/Trigram bằng perplexity trên cùng ba tập dữ liệu.
- Next-word prediction ưu tiên trigram, lùi về bigram hoặc unigram nếu context chưa xuất hiện. Sentence ranking dùng average log probability để giảm thiên lệch do độ dài candidate.
- `inf` trong cột perplexity của MLE là kết quả có thể xảy ra khi tập đánh giá chứa một n-gram chưa thấy trong train và xác suất của nó bằng 0.

Các số liệu trong [`results.csv`](results.csv) thuộc về lần chạy đã lưu trong notebook. Để so sánh lại công bằng, hãy giữ nguyên preprocessing, cách chia dữ liệu và công thức đánh giá.
