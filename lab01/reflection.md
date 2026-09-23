# Reflection

Prediction về quy mô vocabulary và sparsity nhìn chung đúng một phần. Corpus gồm
30.000 documents tạo ra vocabulary lớn ở word-level: 193.837 terms trong
Pipeline D, 473.388 terms trong Pipeline A và 193.332 terms trong Pipeline B.
TF-IDF matrix cũng rất sparse, với tỷ lệ zero entries xấp xỉ 99.9%.

Prediction sai quan trọng nhất là giả định các documents đứng đầu kết quả tìm
kiếm sẽ gần nghĩa nhất với query. Kết quả thực nghiệm cho thấy TF-IDF chủ yếu
dựa trên lexical overlap. Ví dụ, với query “transformer language model”,
document đứng đầu nói về transformer trong mạch điện và không liên quan đến
mô hình ngôn ngữ. Điều này cho thấy cosine similarity có thể ưu tiên một term
riêng lẻ có trọng số cao thay vì đánh giá toàn bộ ngữ nghĩa của document.

Kết quả bất ngờ nhất là Pipeline C có vocabulary nhỏ nhất, chỉ gồm 2.693
subword tokens, nhưng average tokens/document lại cao nhất, khoảng 1.260,74.
Do đó, Pipeline C có matrix sparsity thấp hơn Pipeline A và B
(0,959530 so với 0,999611 và 0,999484). Vocabulary nhỏ hơn không đồng nghĩa
với matrix sparse hơn.

Experiment cung cấp evidence mạnh nhất là preprocessing ablation trong Part F.
Thí nghiệm này cho thấy rõ preprocessing là một modeling decision: thay đổi
tokenizer, punctuation handling, stopword removal và subword tokenization làm
thay đổi vocabulary size, số token trung bình và sparsity.

Failure case quan trọng nhất là việc query “transformer language model” trả về
document về mạch điện. Nguyên nhân là hệ thống nhận diện từ khóa “transformer”
nhưng không mô hình hóa quan hệ ngữ nghĩa giữa “transformer”, “language” và
“model”. P@5 = 0,1500 và Recall@5 = 0,0071 cũng cho thấy khả năng truy hồi còn
hạn chế. Tuy nhiên, các metric này đang dùng pseudo-relevance labels dựa trên
lexical overlap nên cần được diễn giải thận trọng.

Nếu xây dựng lại search engine, em sẽ giữ sparse inverted index nhưng bổ sung
document norms được tính trước, đánh giá bằng relevance labels thủ công và
thử BM25 hoặc dense semantic embeddings để xử lý từ đồng nghĩa và ngữ cảnh.

AI được sử dụng để giải thích yêu cầu bài lab, hỗ trợ debug Python, đề xuất
cấu trúc TF-IDF sparse index, kiểm tra công thức và tổ chức thí nghiệm