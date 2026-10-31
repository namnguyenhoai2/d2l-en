# Các ứng dụng của Xử lý ngôn ngữ tự nhiên
:label:`chap_nlp_app`

Chúng ta đã tìm hiểu cách biểu diễn các token trong các chuỗi văn bản và huấn luyện các biểu diễn của chúng trong :numref:`chap_nlp_pretrain`.
Các biểu diễn văn bản được pretrained này có thể được đưa vào nhiều mô hình khác nhau cho các tác vụ xử lý ngôn ngữ tự nhiên downstream.

Trên thực tế,
các chương trước đã thảo luận một số ứng dụng xử lý ngôn ngữ tự nhiên
*không sử dụng pretraining*,
chỉ nhằm giải thích các kiến trúc deep learning.
Chẳng hạn, trong :numref:`chap_rnn`,
chúng ta đã sử dụng RNN để thiết kế các language model nhằm sinh văn bản giống tiểu thuyết.
Trong :numref:`chap_modern_rnn` và :numref:`chap_attention-and-transformers`,
chúng ta cũng đã thiết kế các mô hình dựa trên RNN và cơ chế attention cho machine translation.

Tuy nhiên, cuốn sách này không nhằm trình bày toàn diện tất cả các ứng dụng như vậy.
Thay vào đó,
trọng tâm của chúng ta là *cách áp dụng việc học biểu diễn (sâu) cho ngôn ngữ để giải quyết các bài toán xử lý ngôn ngữ tự nhiên*.
Với các biểu diễn văn bản được pretrained,
chương này sẽ tìm hiểu hai
tác vụ xử lý ngôn ngữ tự nhiên downstream
phổ biến và tiêu biểu:
phân tích cảm xúc và suy luận ngôn ngữ tự nhiên,
lần lượt phân tích một văn bản đơn lẻ và mối quan hệ giữa các cặp văn bản.

![Các biểu diễn văn bản được pretrained có thể được đưa vào nhiều kiến trúc deep learning khác nhau cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream khác nhau. Chương này tập trung vào cách thiết kế mô hình cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream khác nhau.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

Như được minh họa trong :numref:`fig_nlp-map-app`,
chương này tập trung trình bày những ý tưởng cơ bản trong việc thiết kế các mô hình xử lý ngôn ngữ tự nhiên bằng nhiều loại kiến trúc deep learning khác nhau, chẳng hạn như MLP, CNN, RNN và attention.
Mặc dù có thể kết hợp bất kỳ biểu diễn văn bản được pretrained nào với bất kỳ kiến trúc nào cho mỗi ứng dụng trong :numref:`fig_nlp-map-app`,
chúng ta chọn một vài tổ hợp tiêu biểu.
Cụ thể, chúng ta sẽ tìm hiểu các kiến trúc phổ biến dựa trên RNN và CNN cho bài toán phân tích cảm xúc.
Đối với suy luận ngôn ngữ tự nhiên, chúng ta chọn attention và MLP để minh họa cách phân tích các cặp văn bản.
Cuối cùng, chúng ta giới thiệu cách fine-tune một mô hình BERT được pretrained
cho nhiều ứng dụng xử lý ngôn ngữ tự nhiên,
chẳng hạn như ở cấp độ chuỗi (phân loại văn bản đơn lẻ và phân loại cặp văn bản)
và ở cấp độ token (gắn thẻ văn bản và trả lời câu hỏi).
Là một trường hợp thực nghiệm cụ thể,
chúng ta sẽ fine-tune BERT cho bài toán suy luận ngôn ngữ tự nhiên.

Như đã giới thiệu trong :numref:`sec_bert`,
BERT yêu cầu thay đổi kiến trúc ở mức tối thiểu
cho nhiều ứng dụng xử lý ngôn ngữ tự nhiên.
Tuy nhiên, lợi ích này đi kèm với chi phí fine-tune
một số lượng khổng lồ tham số BERT cho các ứng dụng downstream.
Khi tài nguyên về không gian hoặc thời gian bị giới hạn,
những mô hình được thiết kế riêng dựa trên MLP, CNN, RNN và attention
sẽ khả thi hơn.
Sau đây, chúng ta bắt đầu với ứng dụng phân tích cảm xúc
và lần lượt minh họa cách thiết kế mô hình dựa trên RNN và CNN.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
