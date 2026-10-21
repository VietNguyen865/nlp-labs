# LAB 05 — Sequence classification

Mở `sequence_classification.ipynb`, chọn kernel Python có NumPy, pandas,
matplotlib, scikit-learn, PyTorch và IPython, rồi **Restart Kernel → Run All**.
Dataset được đọc từ `../data/archive.zip` (file `IMDB Dataset.csv`).
Giữ `rnn_cell.py` trong cùng thư mục với notebook.

## Cách đánh giá

- Làm sạch và loại duplicate trước khi stratified split 60/20/20; assert không
  có nội dung trùng giữa các split. Vocabulary chỉ xây từ train.
- Cùng tokenizer, embedding/hidden dimension, batch size, Adam, learning rate,
  epochs và thứ tự batch cho RNN/LSTM/GRU. Test không dùng để chọn model.
- Giữ checkpoint có validation loss thấp nhất; chọn architecture bằng
  validation F1. Application chọn trong LSTM/GRU.
- Precision/Recall/F1 cho lớp positive. Training time gồm train + validation;
  thời gian baseline gồm TF-IDF fit và LR fit. Memory chỉ đo GPU, để trống trên CPU.
- Experiment 5 tái sử dụng đúng các lần train RNN của Experiment 2.
- Word-order test giữ nguyên tập token trong prefix 100 token; việc thay đổi
  prediction chứng minh độ nhạy với thứ tự trên các mẫu đã kiểm tra, không chứng
  minh model hiểu đúng mọi câu.
- Experiment 6 không phải phép đo gradient và không chứng minh vanishing gradient.

## Kết quả

Cell export cuối cùng ghi **một CSV duy nhất**: `lab05_results.csv`.
CSV gồm configuration, dataset/split, token retention, kiểm tra RNN cell,
LSTM gate calculation, experiments, histories, confusion counts, 3 mẫu đúng +
3 mẫu sai có full text/input thực tế, bảng so sánh mục 25 và Application.
Các cột trống là trường không áp dụng cho loại kết quả tương ứng.

Biểu đồ được tái tạo khi Run All:

- `dataset_analysis.png`
- `model_comparison.png`
- `word_order.png`
- `sequence_length.png`
- `long_range_dependency.png`
- `learning_curves.png`
- `confusion_matrix.png`

## Các bài viết riêng

Notebook không thay đổi `calculations.pdf`, `prediction.pdf`,
`error_analysis.pdf` hoặc `reflection.pdf`. Cần đối chiếu các bài viết với số liệu
mới sau khi loại trùng. Bảng Error Analysis phải điền review và nhãn thật,
không giữ các dòng hướng dẫn như “Copy review…”.

Đính chính bài RNN trong đề: W_xh = [[0.5, 0.2], [0.1, 0.4]],
W_hh = [[0.3, 0.1], [0.2, 0.5]], a_t = [0.67, 0.30],
h_t ≈ [0.584980, 0.291313]. Cell kiểm tra có in từng thành phần.

AI assistance statement nằm trong notebook; sinh viên cần tự giải thích
prediction, learning curves, nguyên nhân lỗi và reflection dựa trên kết quả thực tế.
