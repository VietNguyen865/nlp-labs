# Reflection

Dự đoán về vocabulary lớn và ma trận thưa nhìn chung phù hợp với kết quả. Với
30.000 documents, Pipeline A có 473.388 terms, Pipeline B có 193.332 terms và
Pipeline C có 2.693 subword tokens. Tỷ lệ zero của Pipeline A và B xấp xỉ
99,9%, trong khi Pipeline C là 95,953%.

Dự đoán sai quan trọng nhất là các documents đứng đầu sẽ luôn gần nghĩa nhất
với query. Với query `transformer language model`, document đứng đầu nói về
transformer trong mạch điện. TF-IDF chỉ khai thác lexical overlap và cosine
similarity, nên không phân biệt được các nghĩa khác nhau của cùng một term.

Kết quả bất ngờ nhất là Pipeline C có vocabulary nhỏ nhất nhưng average
tokens/document cao nhất, khoảng 1.260,74. Vì vậy vocabulary nhỏ hơn không
đồng nghĩa với matrix sparse hơn.

Evidence mạnh nhất đến từ preprocessing ablation trong Part F. Việc thay đổi
tokenizer, xử lý punctuation, stopword removal và subword tokenization làm
thay đổi rõ rệt vocabulary, số token và sparsity. Baseline Part H đạt
P@5 = 0,2400, Recall@5 = 0,0095 và MRR = 0,4667. Trong đánh giá A/B/C lưu ở
`results.csv`, Pipeline B có MRR cao nhất là 0,5400; Pipeline C đạt 0 vì
relevance labels được tạo theo word-level lexical overlap, không phù hợp trực
tiếp với subword representation.

Failure case quan trọng nhất là query về `transformer`. Đây không phải lỗi của
công thức TF, IDF hay cosine similarity mà là giới hạn của lexical
representation. Nếu xây dựng lại search engine, em sẽ giữ sparse inverted index
nhưng bổ sung relevance labels thủ công, thử BM25 và dùng dense semantic
embeddings hoặc contextual Transformer representation để xử lý ngữ nghĩa,
đồng nghĩa và ngữ cảnh.

AI được sử dụng để giải thích yêu cầu, hỗ trợ debug, kiểm tra công thức,
đề xuất sparse-index optimization và tổ chức thí nghiệm.
