# LAB 08 — Pretrained Transformers cho phân loại cảm xúc

Bài thực hành khảo sát tokenization và dùng Transformer pretrained cho bài toán phân loại review phim. Thực nghiệm chính sử dụng IMDb; một bộ Amazon Reviews được dùng để thống kê dữ liệu và đánh giá khả năng chuyển miền.

## Nội dung

- `tokenization.ipynb`: thống kê dữ liệu IMDb; xem token, token ID, special token và attention mask; so sánh từ với subword; phân tích truncation, fragmentation và padding.
- `pretrained_classification.py`: cài đặt `PretrainedClassificationHead` — encoder Transformer, dropout và classification head hai lớp. File này không chứa vòng lặp huấn luyện hay các thực nghiệm.
- `fine_tuning.ipynb`: dữ liệu, huấn luyện, đánh giá, trực quan hóa và các mục thực nghiệm 14–22.

Các thực nghiệm trong `fine_tuning.ipynb` gồm pretrained so với khởi tạo ngẫu nhiên, đóng băng encoder so với fine-tuning, lượng dữ liệu train, độ dài chuỗi, domain shift IMDb/Amazon, ứng dụng sentiment classification, phân tích lỗi, confidence và ablation.

## Dữ liệu

Đặt `archive.zip` trong thư mục `data` nằm cạnh `lab08`:

```text
NLP/Lab/
├── data/
│   └── archive.zip
└── lab08/
    ├── fine_tuning.ipynb
    ├── tokenization.ipynb
    └── pretrained_classification.py
```

Notebook tokenization đọc CSV IMDb hoặc cấu trúc `aclImdb` trong ZIP. Notebook fine-tuning cần CSV IMDb có cột review và sentiment. Các nhãn được chuẩn hóa thành `0 = negative`, `1 = positive`.

Amazon Reviews được tải qua KaggleHub với dataset `kritanjalijain/amazon-reviews`. Lần đầu chạy cần kết nối Kaggle và truy cập được dataset; các lần sau notebook ưu tiên dữ liệu đã cache.

## Cài đặt

Khuyến nghị Python 3.10 trở lên và Jupyter Notebook hoặc JupyterLab. Cài các thư viện cần thiết trong môi trường đang dùng cho notebook:

```bash
pip install torch transformers huggingface_hub pandas numpy scikit-learn matplotlib seaborn tqdm kagglehub jupyter
```

## Cách chạy

1. Mở terminal tại thư mục `lab08`.
2. Khởi động Jupyter:

   ```bash
   jupyter lab
   ```

3. Chạy `tokenization.ipynb` từ đầu đến cuối để xem thống kê và khảo sát tokenizer.
4. Chạy `fine_tuning.ipynb` từ đầu đến cuối để chạy các mô hình, đánh giá và lưu biểu đồ.

Cả hai notebook dùng seed `2026`. Mô hình mặc định là `prajjwal1/bert-tiny`. Lần đầu sử dụng cần tải checkpoint từ Hugging Face; notebook fine-tuning lưu checkpoint cục bộ và ưu tiên dùng cache hoàn chỉnh ở những lần chạy sau.

Notebook tự chọn GPU CUDA nếu có. Khi chạy CPU, fine-tuning giới hạn mỗi dataset còn tối đa 2.400 mẫu train, 800 validation và 800 test nhằm giảm thời gian chạy. Khi có CUDA, notebook dùng cấu hình không giới hạn các split theo các cap CPU này. Thời gian huấn luyện và chất lượng kết quả phụ thuộc thiết bị, thư viện, checkpoint và trạng thái cache.

### Chạy trên Google Colab

Có thể mở notebook trong Colab, đưa `lab08` lên Google Drive hoặc clone repository, rồi tải lên hoặc mount `archive.zip`. `fine_tuning.ipynb` dò các vị trí phổ biến như `/content/archive.zip` và `/content/drive/MyDrive/archive.zip`. Nếu dùng vị trí khác, sửa `IMDB_ZIP_CANDIDATES` ở cell nạp dữ liệu để trỏ đến file ZIP. Để import `pretrained_classification.py` thành công, file Python cần nằm cùng thư mục notebook đang chạy.

## Kết quả được lưu

Cell cuối của `fine_tuning.ipynb` xuất các tệp vào thư mục `lab08` khi chạy trên máy cá nhân:

- `data_efficiency.png`: hiệu năng và thời gian theo lượng dữ liệu train.
- `confusion_matrix.png`: ma trận nhầm lẫn dạng số lượng và chuẩn hóa theo nhãn thật.
- `confidence_analysis.png`: phân bố confidence, reliability diagram và risk–coverage.
- `results.csv`: metrics của các thực nghiệm cùng kết quả ứng dụng sentiment classification.

Các notebook cũng hiển thị trực tiếp biểu đồ, bảng số liệu, đường loss train/validation, ví dụ dự đoán và các review bị phân loại sai.

## Lưu ý khi đọc kết quả

- Phân tích tokenization khảo sát tối đa 5.000 review để tính toán nhanh; thống kê dataset overview dùng toàn bộ dữ liệu IMDb đã làm sạch.
- So sánh giữa các thực nghiệm chỉ có ý nghĩa khi giữ nguyên split và các thiết lập còn lại như mô tả trong notebook.
- Mục data efficiency hiện đo các tỷ lệ dữ liệu bằng cấu hình sweep được khai báo trong notebook. Cần xem số epoch thực tế được in ở output khi diễn giải.
- Nếu chạy lại một cell nhiều lần, hãy chạy lại cell tổng kết và cell xuất CSV để cập nhật kết quả mới nhất. Đặc biệt, kiểm tra các dòng trùng tên ở sequence-length experiment trước khi dùng bảng tổng kết.
- Ví dụ lỗi trong notebook có thể bị rút gọn khi hiển thị; cần đọc review đầy đủ trước khi gán nguyên nhân như sarcasm, negation hoặc ambiguous sentiment.
