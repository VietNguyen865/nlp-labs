# LAB 04 — Sentiment Classification

## 1. Mục tiêu

Bài lab xây dựng mô hình phân loại cảm xúc cho các nhận xét phim IMDb thành hai lớp:

- `positive`
- `negative`

Các biểu diễn văn bản được so sánh gồm TF-IDF và Average Word2Vec. Các bộ phân loại được sử dụng là Logistic Regression và Linear SVM.

## 2. Cấu trúc thư mục

- `classification.ipynb`: Notebook chính, chứa toàn bộ quá trình đọc dữ liệu, tiền xử lý, huấn luyện, đánh giá và trực quan hóa.
- `preprocessing.py`: Class `TextPreprocessor`, dùng để chuyển chữ thường, giải mã HTML, xóa HTML tag và chuẩn hóa khoảng trắng.
- `results.csv`: Kết quả tổng hợp của các thí nghiệm.
- `confusion_matrix.png`: Ma trận nhầm lẫn của mô hình được chọn.
- `calculations.pdf`: Phần tính toán thủ công.
- `prediction.pdf`: Các dự đoán trước khi chạy thí nghiệm.
- `reflection.pdf`: Phần reflection sau khi hoàn thành bài lab.
- `error_analysis.pdf`: Phân tích các mẫu dự đoán đúng và dự đoán sai.

## 3. Dữ liệu và cách chia tập

Dữ liệu sử dụng là IMDb Dataset gồm 50.000 nhận xét phim. Sau khi làm sạch và loại bỏ dữ liệu trùng lặp, còn 49.578 mẫu.

Dữ liệu được chia theo tỉ lệ:

- Training set: 60%, gồm 29.746 mẫu.
- Validation set: 20%, gồm 9.916 mẫu.
- Test set: 20%, gồm 9.916 mẫu.

Việc chia dữ liệu có stratification để giữ gần nguyên tỉ lệ hai lớp. Vectorizer, vocabulary và các tham số mô hình được lựa chọn dựa trên training set và validation set. Test set chỉ được sử dụng để đánh giá cuối cùng.

## 4. Cách chạy chương trình

Mở terminal tại thư mục `lab04` và cài đặt các thư viện cần thiết:

```bash
pip install numpy pandas matplotlib scikit-learn gensim scipy
```

Sau đó mở notebook:

```bash
jupyter notebook classification.ipynb
```

Chạy các cell theo đúng thứ tự từ trên xuống dưới. Nếu chuyển notebook sang máy khác, cần cập nhật biến `LAB_DIR` và đường dẫn đến file dữ liệu.

## 5. Các thí nghiệm

### Experiment 1 — Dataset exploration

Kiểm tra số lượng mẫu, phân bố nhãn, độ dài văn bản, dữ liệu trùng lặp và dữ liệu thiếu.

### Experiment 2 — Baseline

Sử dụng TF-IDF kết hợp Logistic Regression. Các chỉ số được báo cáo gồm training time, vocabulary size, accuracy, precision, recall và F1-score.

### Experiment 3 — Representation comparison

So sánh các mô hình:

- TF-IDF + Logistic Regression.
- Average Word2Vec + Logistic Regression.
- TF-IDF + Linear SVM.

### Experiment 4 — Word order

So sánh các câu có và không có phủ định, đồng thời kiểm tra ảnh hưởng của việc thay đổi thứ tự từ.

### Experiment 5 — N-gram features

So sánh TF-IDF dùng unigram với TF-IDF dùng unigram kết hợp bigram.

### Model interpretation và error analysis

Phân tích các feature có trọng số dương và âm, ma trận nhầm lẫn, ba mẫu dự đoán đúng và ba mẫu dự đoán sai.

## 6. Kết quả chính trên tập test

| Representation | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| TF-IDF | Logistic Regression | 0.9117 | 0.9063 | 0.9190 | 0.9126 |
| Average Word2Vec | Logistic Regression | 0.8812 | 0.8801 | 0.8837 | 0.8819 |
| TF-IDF | Linear SVM | 0.9125 | 0.9080 | 0.9186 | 0.9133 |
| TF-IDF unigram + bigram | Logistic Regression | 0.9190 | 0.9113 | 0.9291 | 0.9201 |

Kết quả cho thấy unigram kết hợp bigram đạt F1-score cao nhất. Bigram giúp mô hình nhận biết một số cụm từ như `not good` và `very good`, nhưng làm vocabulary và thời gian huấn luyện tăng đáng kể. Average Word2Vec có kết quả thấp hơn vì phép lấy trung bình vector làm mất thông tin thứ tự từ và ngữ cảnh.

## 7. Kết luận

TF-IDF kết hợp Logistic Regression là một baseline mạnh và đơn giản cho bài toán sentiment classification. Mô hình hoạt động tốt với các từ khóa cảm xúc rõ ràng nhưng còn hạn chế khi xử lý phủ định, mỉa mai, câu có nhiều ý trái chiều và phụ thuộc vào ngữ cảnh. Việc sử dụng bigram giúp cải thiện khả năng biểu diễn các cụm từ cảm xúc.

