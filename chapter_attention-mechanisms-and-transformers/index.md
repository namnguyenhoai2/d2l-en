# Cơ chế Attention và Transformers
:label:`chap_attention-and-transformers`


Những năm đầu của làn sóng bùng nổ deep learning chủ yếu được thúc đẩy
bởi các kết quả đạt được với kiến trúc perceptron đa tầng,
mạng tích chập và mạng hồi tiếp.
Đáng chú ý, các kiến trúc mô hình làm nền tảng
cho nhiều đột phá của deep learning trong thập niên 2010
hầu như thay đổi rất ít so với các kiến trúc tiền thân,
mặc dù đã trải qua gần 30 năm.
Mặc dù nhiều đổi mới về phương pháp
đã được đưa vào bộ công cụ của hầu hết người thực hành---ReLU
activation, residual layer, batch normalization, dropout
và adaptive learning rate schedule là những ví dụ điển hình---các kiến trúc
nền tảng cốt lõi vẫn dễ dàng được nhận ra
là những hiện thực hóa được mở rộng quy mô của các ý tưởng kinh điển.
Dù có hàng nghìn bài báo đề xuất các ý tưởng thay thế,
các mô hình tương tự mạng nơ-ron tích chập kinh điển (:numref:`chap_cnn`)
vẫn giữ vị thế *state-of-the-art* trong computer vision,
và các mô hình tương tự thiết kế ban đầu của Sepp Hochreiter
cho mạng nơ-ron hồi tiếp LSTM (:numref:`sec_lstm`)
thống trị phần lớn ứng dụng trong xử lý ngôn ngữ tự nhiên.
Có thể cho rằng, cho đến thời điểm đó, sự trỗi dậy nhanh chóng của deep learning
dường như chủ yếu bắt nguồn từ những thay đổi
về tài nguyên tính toán sẵn có
(nhờ các đổi mới trong tính toán song song với GPU)
và sự sẵn có của nguồn dữ liệu khổng lồ
(nhờ lưu trữ giá rẻ và các dịch vụ Internet).
Mặc dù các yếu tố này thực sự có thể vẫn là động lực chính
đằng sau sức mạnh ngày càng tăng của công nghệ này,
cuối cùng chúng ta cũng đang chứng kiến
một thay đổi mang tính bước ngoặt trong bối cảnh các kiến trúc chủ đạo.

Hiện tại, các mô hình chủ đạo
cho gần như mọi tác vụ xử lý ngôn ngữ tự nhiên
đều dựa trên kiến trúc Transformer.
Với bất kỳ tác vụ mới nào trong xử lý ngôn ngữ tự nhiên, cách tiếp cận mặc định ở bước đầu
là lấy một mô hình pretrained lớn dựa trên Transformer,
(ví dụ: BERT :cite:`Devlin.Chang.Lee.ea.2018`, ELECTRA :cite:`clark2019electra`, RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`, hoặc Longformer :cite:`beltagy2020longformer`)
điều chỉnh các output layer khi cần thiết,
và fine-tune mô hình trên dữ liệu sẵn có
cho tác vụ downstream.
Nếu bạn đã theo dõi những năm gần đây
với các bản tin sôi nổi xoay quanh
các large language model của OpenAI, thì bạn đã theo dõi một cuộc thảo luận
tập trung vào các mô hình GPT-2 và GPT-3 dựa trên Transformer :cite:`Radford.Wu.Child.ea.2019,brown2020language`.
Trong khi đó, vision Transformer đã trở thành
mô hình mặc định cho nhiều tác vụ thị giác đa dạng,
bao gồm nhận dạng ảnh, phát hiện đối tượng,
semantic segmentation và superresolution :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`.
Transformer cũng xuất hiện như những phương pháp cạnh tranh
cho nhận dạng giọng nói :cite:`gulati2020conformer`,
reinforcement learning :cite:`chen2021decision`,
và graph neural network :cite:`dwivedi2020generalization`.

Ý tưởng cốt lõi đằng sau mô hình Transformer là *cơ chế attention*,
một đổi mới ban đầu được hình dung như một cải tiến
cho các RNN encoder--decoder áp dụng cho các ứng dụng sequence-to-sequence,
chẳng hạn như dịch máy :cite:`Bahdanau.Cho.Bengio.2014`.
Bạn có thể còn nhớ rằng trong các mô hình sequence-to-sequence đầu tiên
dành cho dịch máy :cite:`Sutskever.Vinyals.Le.2014`,
toàn bộ input được encoder nén
thành một vector có độ dài cố định để đưa vào decoder.
Trực giác đằng sau attention là thay vì nén input,
có lẽ sẽ tốt hơn nếu decoder xem lại chuỗi input tại mỗi bước.
Hơn nữa, thay vì luôn nhìn thấy cùng một biểu diễn của input,
ta có thể hình dung rằng decoder nên tập trung có chọn lọc
vào những phần cụ thể của chuỗi input tại các bước decoding cụ thể.
Cơ chế attention của Bahdanau cung cấp một cách đơn giản
để decoder có thể linh hoạt *chú ý* đến các
phần khác nhau của input tại mỗi bước decoding.
Ý tưởng ở mức cao là encoder có thể tạo ra một biểu diễn
có độ dài bằng chuỗi input ban đầu.
Sau đó, tại thời điểm decoding, decoder có thể (thông qua một cơ chế điều khiển nào đó)
nhận làm input một context vector bao gồm tổng có trọng số
của các biểu diễn của input tại mỗi time step.
Theo trực giác, các trọng số quyết định mức độ
mà context của mỗi bước “tập trung” vào từng token input,
và điểm then chốt là làm cho quá trình
gán trọng số này khả vi
để nó có thể được học cùng với
tất cả các tham số mạng nơ-ron khác.

Ban đầu, ý tưởng này là một cải tiến thành công đáng kể
đối với các mạng nơ-ron hồi tiếp
vốn đã thống trị các ứng dụng dịch máy.
Các mô hình hoạt động tốt hơn các kiến trúc
sequence-to-sequence encoder--decoder ban đầu.
Hơn nữa, các nhà nghiên cứu nhận thấy rằng đôi khi có thể rút ra
một số hiểu biết định tính hữu ích khi kiểm tra mẫu của các attention weight.
Trong các tác vụ dịch, mô hình attention
thường gán attention weight cao cho các từ đồng nghĩa xuyên ngôn ngữ
khi sinh ra các từ tương ứng trong ngôn ngữ đích.
Ví dụ, khi dịch câu “my feet hurt”
sang “j'ai mal au pieds”, mạng nơ-ron có thể gán
attention weight cao cho biểu diễn của “feet”
khi sinh từ tiếng Pháp tương ứng là “pieds”.
Những hiểu biết này đã thúc đẩy các tuyên bố rằng mô hình attention mang lại “khả năng diễn giải”
mặc dù chính xác attention weight có nghĩa là gì---tức là,
chúng nên được *diễn giải* như thế nào, nếu có thể---vẫn là một chủ đề nghiên cứu chưa rõ ràng.

Tuy nhiên, các cơ chế attention nhanh chóng trở nên quan trọng hơn,
vượt ngoài tính hữu ích của chúng như một cải tiến cho mạng nơ-ron hồi tiếp encoder--decoder
và tính hữu ích được cho là có thể giúp chọn ra các input nổi bật.
:citet:`Vaswani.Shazeer.Parmar.ea.2017` đã đề xuất
kiến trúc Transformer cho dịch máy,
loại bỏ hoàn toàn các kết nối hồi tiếp,
và thay vào đó dựa vào các cơ chế attention được sắp xếp khéo léo
để nắm bắt mọi mối quan hệ giữa token input và output.
Kiến trúc này hoạt động tốt đáng kể,
và đến năm 2018, Transformer bắt đầu xuất hiện
trong phần lớn các hệ thống xử lý ngôn ngữ tự nhiên state-of-the-art.
Hơn nữa, cùng thời điểm đó, thực hành chủ đạo trong xử lý ngôn ngữ tự nhiên
đã trở thành pretrain các mô hình quy mô lớn
trên các corpus nền tổng quát khổng lồ
để tối ưu một mục tiêu pretraining self-supervised,
sau đó fine-tune các mô hình này
bằng dữ liệu downstream sẵn có.
Khoảng cách giữa Transformer và các kiến trúc truyền thống
trở nên đặc biệt lớn khi được áp dụng trong mô hình pretraining này,
và vì thế sự vươn lên của Transformer trùng hợp
với sự vươn lên của các mô hình pretrained quy mô lớn như vậy,
hiện nay đôi khi được gọi là *foundation model* :cite:`bommasani2021opportunities`.


Trong chương này, chúng ta giới thiệu các mô hình attention,
bắt đầu với những trực giác cơ bản nhất
và các hiện thực hóa đơn giản nhất của ý tưởng này.
Sau đó, chúng ta sẽ từng bước tìm hiểu kiến trúc Transformer,
vision Transformer và bối cảnh
của các mô hình pretrained hiện đại dựa trên Transformer.

```toc
:maxdepth: 2

queries-keys-values
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```
