# Xử lý ngôn ngữ tự nhiên: Pretraining
:label:`chap_nlp_pretrain`


Con người cần giao tiếp.
Từ nhu cầu cơ bản này của con người, một lượng lớn văn bản viết được tạo ra hằng ngày.
Với nguồn văn bản phong phú từ mạng xã hội, ứng dụng trò chuyện, email, đánh giá sản phẩm, bài báo thời sự, bài nghiên cứu và sách, việc giúp máy tính hiểu chúng để hỗ trợ hoặc đưa ra quyết định dựa trên ngôn ngữ của con người trở nên vô cùng quan trọng.

*Xử lý ngôn ngữ tự nhiên* nghiên cứu sự tương tác giữa máy tính và con người bằng ngôn ngữ tự nhiên.
Trong thực tế, các kỹ thuật xử lý ngôn ngữ tự nhiên thường được sử dụng để xử lý và phân tích dữ liệu văn bản (ngôn ngữ tự nhiên của con người), chẳng hạn như các mô hình ngôn ngữ trong :numref:`sec_language-model` và các mô hình machine translation trong :numref:`sec_machine_translation`.

Để hiểu văn bản, chúng ta có thể bắt đầu bằng việc học
các biểu diễn của nó.
Tận dụng các chuỗi văn bản hiện có
từ những kho ngữ liệu lớn,
*self-supervised learning*
đã được sử dụng rộng rãi
để pretrain các biểu diễn văn bản,
chẳng hạn như dự đoán một phần bị che giấu của văn bản
bằng cách sử dụng một phần khác trong văn bản xung quanh nó.
Bằng cách này,
các mô hình học thông qua sự giám sát
từ dữ liệu văn bản *khổng lồ*
mà không cần *nỗ lực gán nhãn tốn kém*!


Như chúng ta sẽ thấy trong chương này,
khi xem mỗi từ hoặc subword là một token riêng lẻ,
biểu diễn của mỗi token có thể được pretrain
bằng word2vec, GloVe hoặc các mô hình subword embedding
trên những kho ngữ liệu lớn.
Sau khi pretrain, biểu diễn của mỗi token có thể là một vector,
tuy nhiên, nó vẫn không thay đổi bất kể ngữ cảnh là gì.
Ví dụ, biểu diễn vector của "bank" giống nhau
trong cả hai câu
"go to the bank to deposit some money"
và
"go to the bank to sit down".
Vì vậy, nhiều mô hình pretraining gần đây hơn điều chỉnh biểu diễn của cùng một token
cho các ngữ cảnh khác nhau.
Trong số đó có BERT, một mô hình self-supervised sâu hơn nhiều dựa trên Transformer encoder.
Trong chương này, chúng ta sẽ tập trung vào cách pretrain những biểu diễn như vậy cho văn bản,
như được nhấn mạnh trong :numref:`fig_nlp-map-pretrain`.

![Các biểu diễn văn bản đã pretrain có thể được đưa vào nhiều kiến trúc deep learning khác nhau cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream khác nhau. Chương này tập trung vào việc pretrain biểu diễn văn bản upstream.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`


Để có cái nhìn tổng quan,
:numref:`fig_nlp-map-pretrain` cho thấy rằng
các biểu diễn văn bản đã pretrain có thể được đưa vào
nhiều kiến trúc deep learning khác nhau cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream khác nhau.
Chúng ta sẽ trình bày chúng trong :numref:`chap_nlp_app`.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```
