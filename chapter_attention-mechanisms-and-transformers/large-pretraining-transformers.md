# Tiền huấn luyện quy mô lớn với Transformers
:label:`sec_large-pretraining-transformers`

Cho đến nay, trong các thí nghiệm phân loại ảnh và dịch máy của chúng ta,
các mô hình đã được huấn luyện trên các tập dữ liệu có ví dụ đầu vào--đầu ra
*từ đầu* để thực hiện các tác vụ cụ thể.
Ví dụ, một Transformer đã được huấn luyện
với các cặp tiếng Anh--tiếng Pháp (:numref:`sec_transformer`)
để mô hình này có thể dịch văn bản tiếng Anh đầu vào sang tiếng Pháp.
Do đó, mỗi mô hình trở thành một *chuyên gia cụ thể*
nhạy cảm ngay cả với những thay đổi nhỏ trong phân phối dữ liệu
(:numref:`sec_environment-and-distribution-shift`).
Để có các mô hình tổng quát hóa tốt hơn, hoặc thậm chí các *mô hình tổng quát*
có năng lực hơn có thể thực hiện nhiều tác vụ dù có hoặc không cần thích ứng,
việc *tiền huấn luyện* mô hình trên dữ liệu lớn ngày càng trở nên phổ biến.

Với dữ liệu tiền huấn luyện lớn hơn, kiến trúc Transformer
hoạt động tốt hơn khi kích thước mô hình và năng lực tính toán huấn luyện tăng lên,
cho thấy hành vi *mở rộng* vượt trội.
Cụ thể, hiệu năng của các mô hình ngôn ngữ dựa trên Transformer
mở rộng theo luật lũy thừa với số lượng tham số mô hình,
token huấn luyện và năng lực tính toán huấn luyện :cite:`kaplan2020scaling`.
Khả năng mở rộng của Transformers cũng được chứng minh
bởi hiệu năng tăng đáng kể
từ các vision Transformer lớn hơn được huấn luyện trên dữ liệu lớn hơn
(được thảo luận trong :numref:`sec_vision-transformer`).
Các câu chuyện thành công gần đây hơn bao gồm Gato, một mô hình *tổng quát*
có thể chơi Atari, tạo chú thích ảnh, trò chuyện và vận hành như robot :cite:`reed2022generalist`. Gato là một Transformer duy nhất có khả năng mở rộng tốt khi được tiền huấn luyện trên các modality đa dạng,
bao gồm văn bản, ảnh, mô-men xoắn khớp và thao tác nhấn nút.
Đáng chú ý, toàn bộ dữ liệu đa phương thức như vậy được tuần tự hóa thành một chuỗi token phẳng,
có thể được Transformers xử lý tương tự token văn bản (:numref:`sec_transformer`)
hoặc các patch ảnh (:numref:`sec_vision-transformer`).

Trước thành công thuyết phục của việc tiền huấn luyện Transformers cho dữ liệu đa phương thức,
Transformers đã được tiền huấn luyện rộng rãi với lượng văn bản phong phú.
Ban đầu được đề xuất cho dịch máy,
kiến trúc Transformer trong :numref:`fig_transformer`
bao gồm một encoder để biểu diễn các chuỗi đầu vào
và một decoder để tạo các chuỗi đích.
Chủ yếu, Transformers có thể được sử dụng theo ba chế độ khác nhau:
*chỉ encoder*, *encoder--decoder* và *chỉ decoder*.
Để kết thúc chương này, chúng ta sẽ xem lại ba chế độ này
và giải thích khả năng mở rộng khi tiền huấn luyện Transformers.

## Chỉ Encoder

Khi chỉ sử dụng Transformer encoder,
một chuỗi token đầu vào được chuyển đổi
thành cùng số lượng biểu diễn
có thể tiếp tục được chiếu sang đầu ra
(ví dụ: phân loại). Một Transformer encoder
bao gồm các lớp self-attention,
trong đó mọi token đầu vào đều attention lẫn nhau.
Ví dụ, các vision Transformer được minh họa trong :numref:`fig_vit`
là loại chỉ encoder, chuyển một chuỗi patch ảnh đầu vào thành
biểu diễn của một token đặc biệt “&lt;cls&gt;”.
Vì biểu diễn này phụ thuộc vào tất cả token đầu vào,
nó tiếp tục được chiếu sang các nhãn phân loại.
Thiết kế này được lấy cảm hứng từ một Transformer chỉ encoder trước đó
được tiền huấn luyện trên văn bản: BERT (Bidirectional Encoder Representations from Transformers) :cite:`Devlin.Chang.Lee.ea.2018`.


### Tiền huấn luyện BERT

![Bên trái: Tiền huấn luyện BERT bằng mô hình hóa ngôn ngữ có che mặt nạ. Việc dự đoán token "love" bị che phụ thuộc vào tất cả token đầu vào trước và sau "love". Bên phải: Mẫu attention trong Transformer encoder. Mỗi token dọc theo trục đứng attention đến tất cả token đầu vào dọc theo trục ngang.](../img/bert-encoder-only.svg)
:label:`fig_bert-encoder-only`

BERT được tiền huấn luyện trên các chuỗi văn bản bằng *mô hình hóa ngôn ngữ có che mặt nạ*:
văn bản đầu vào với các token được che ngẫu nhiên được đưa
vào một Transformer encoder để dự đoán các token bị che.
Như được minh họa trong :numref:`fig_bert-encoder-only`,
một chuỗi văn bản gốc gồm "I", "love", "this", "red", "car"
được thêm tiền tố bằng token “&lt;cls&gt;”, và token “&lt;mask&gt;”
thay thế ngẫu nhiên cho "love"; sau đó loss cross-entropy giữa token bị che "love"
và dự đoán của nó được tối thiểu hóa trong quá trình tiền huấn luyện.
Lưu ý rằng không có ràng buộc nào trong mẫu attention của Transformer encoder
(bên phải :numref:`fig_bert-encoder-only`)
nên tất cả token có thể attention đến nhau.
Vì vậy, dự đoán của "love" phụ thuộc vào các token đầu vào đứng trước và sau nó trong chuỗi.
Đây là lý do BERT là một "encoder hai chiều".
Không cần gán nhãn thủ công, dữ liệu văn bản quy mô lớn
từ sách và Wikipedia có thể được sử dụng để tiền huấn luyện BERT.


### Fine-Tuning BERT

BERT đã tiền huấn luyện có thể được *fine-tune* cho các tác vụ encoding hạ nguồn liên quan đến một văn bản đơn lẻ hoặc cặp văn bản. Trong quá trình fine-tuning, có thể thêm các lớp bổ sung vào BERT với các tham số được khởi tạo ngẫu nhiên: các tham số này và các tham số BERT đã tiền huấn luyện sẽ được *cập nhật* để phù hợp với dữ liệu huấn luyện của các tác vụ hạ nguồn.

![Fine-tuning BERT cho phân tích cảm xúc.](../img/bert-finetune-classification.svg)
:label:`fig_bert-finetune-classification`

:numref:`fig_bert-finetune-classification` minh họa
việc fine-tuning BERT cho phân tích cảm xúc.
Transformer encoder là một BERT đã tiền huấn luyện,
nhận một chuỗi văn bản làm đầu vào
và đưa biểu diễn “&lt;cls&gt;”
(biểu diễn toàn cục của đầu vào)
vào một lớp fully connected bổ sung
để dự đoán cảm xúc.
Trong quá trình fine-tuning, loss cross-entropy
giữa dự đoán và nhãn
trên dữ liệu phân tích cảm xúc
được tối thiểu hóa thông qua các thuật toán dựa trên gradient,
trong đó lớp bổ sung được huấn luyện từ đầu
còn các tham số đã tiền huấn luyện của BERT được cập nhật.
BERT không chỉ thực hiện phân tích cảm xúc.
Các biểu diễn ngôn ngữ tổng quát mà BERT với 350 triệu tham số học được
từ 250 tỷ token huấn luyện
đã nâng cao state of the art cho các tác vụ ngôn ngữ tự nhiên
như phân loại văn bản đơn lẻ,
phân loại hoặc hồi quy cặp văn bản,
gán thẻ văn bản và trả lời câu hỏi.

Bạn có thể nhận thấy rằng các tác vụ hạ nguồn này bao gồm việc hiểu cặp văn bản.
Tiền huấn luyện BERT có thêm một loss để dự đoán
liệu một câu có ngay sau câu kia hay không.
Tuy nhiên, loss này về sau được phát hiện là ít hữu ích hơn khi tiền huấn luyện RoBERTa,
một biến thể BERT có cùng kích thước, trên 2000 tỷ token :cite:`Liu.Ott.Goyal.ea.2019`.
Các biến thể khác của BERT đã cải thiện kiến trúc mô hình hoặc mục tiêu tiền huấn luyện,
chẳng hạn ALBERT (áp dụng chia sẻ tham số) :cite:`lan2019albert`,
SpanBERT (biểu diễn và dự đoán các đoạn văn bản) :cite:`joshi2020spanbert`,
DistilBERT (nhẹ hơn nhờ knowledge distillation) :cite:`sanh2019distilbert`,
và ELECTRA (phát hiện token bị thay thế) :cite:`clark2019electra`.
Hơn nữa, BERT đã truyền cảm hứng cho tiền huấn luyện Transformer trong thị giác máy tính,
chẳng hạn với vision Transformers :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`,
Swin Transformers :cite:`liu2021swin`,
và MAE (masked autoencoders) :cite:`he2022masked`.

## Encoder--Decoder

Vì một Transformer encoder chuyển đổi một chuỗi token đầu vào
thành cùng số lượng biểu diễn đầu ra,
chế độ chỉ encoder không thể tạo ra một chuỗi có độ dài tùy ý như trong dịch máy.
Như đã được đề xuất ban đầu cho dịch máy,
kiến trúc Transformer có thể được trang bị một decoder
để dự đoán tự hồi quy chuỗi đích
có độ dài tùy ý, từng token một,
có điều kiện theo cả đầu ra encoder và đầu ra decoder:
(i) để phụ thuộc vào đầu ra encoder, encoder--decoder cross-attention
(multi-head attention của decoder trong :numref:`fig_transformer`)
cho phép các token đích attention đến *tất cả* token đầu vào;
(ii) việc phụ thuộc vào đầu ra decoder đạt được
bằng một mẫu attention gọi là *causal*
(tên này phổ biến trong tài liệu nhưng gây hiểu lầm
vì nó ít liên hệ với nghiên cứu đúng nghĩa về quan hệ nhân quả)
(masked multi-head attention của decoder trong :numref:`fig_transformer`),
trong đó bất kỳ token đích nào chỉ có thể attention đến các token *trước đó* và *hiện tại* trong chuỗi đích.

Để tiền huấn luyện các Transformer encoder--decoder vượt ra ngoài dữ liệu dịch máy được con người gán nhãn,
BART :cite:`lewis2019bart` và T5 :cite:`raffel2020exploring`
là hai Transformer encoder--decoder được đề xuất đồng thời
và tiền huấn luyện trên các corpus văn bản quy mô lớn.
Cả hai đều cố gắng tái tạo văn bản gốc trong các mục tiêu tiền huấn luyện,
trong đó mô hình thứ nhất nhấn mạnh việc gây nhiễu đầu vào
(ví dụ: che mặt nạ, xóa, hoán vị và xoay)
còn mô hình thứ hai nhấn mạnh việc thống nhất đa tác vụ
với các nghiên cứu ablation toàn diện.


### Tiền huấn luyện T5


Là một ví dụ về Transformer encoder--decoder đã tiền huấn luyện,
T5 (Text-to-Text Transfer Transformer)
thống nhất nhiều tác vụ thành cùng một bài toán text-to-text:
với bất kỳ tác vụ nào, đầu vào của encoder là phần mô tả tác vụ
(ví dụ: "Summarize", ":") theo sau bởi đầu vào tác vụ
(ví dụ: một chuỗi token từ một bài báo),
và decoder dự đoán đầu ra tác vụ
(ví dụ: một chuỗi token tóm tắt bài báo đầu vào).
Để hoạt động theo dạng text-to-text, T5 được huấn luyện
để tạo một số văn bản đích có điều kiện theo văn bản đầu vào.


![Bên trái: Tiền huấn luyện T5 bằng cách dự đoán các span liên tiếp. Câu gốc là "I", "love", "this", "red", "car", trong đó "love" được thay bằng token đặc biệt “&lt;X&gt;”, và "red", "car" liên tiếp được thay bằng token đặc biệt “&lt;Y&gt;”. Chuỗi đích kết thúc bằng token đặc biệt “&lt;Z&gt;”. Bên phải: Mẫu attention trong Transformer encoder--decoder. Trong encoder self-attention (hình vuông phía dưới), mọi token đầu vào attention lẫn nhau; trong encoder--decoder cross-attention (hình chữ nhật phía trên), mỗi token đích attention đến tất cả token đầu vào; trong decoder self-attention (tam giác phía trên), mỗi token đích chỉ attention đến các token đích hiện tại và trước đó (causal).](../img/t5-encoder-decoder.svg)
:label:`fig_t5-encoder-decoder`

Để thu được đầu vào và đầu ra từ bất kỳ văn bản gốc nào,
T5 được pretrain để dự đoán các span liên tiếp.
Cụ thể, các token trong văn bản được thay thế ngẫu nhiên
bằng các token đặc biệt, trong đó mỗi span liên tiếp
được thay thế bằng cùng một token đặc biệt.
Hãy xem xét ví dụ trong :numref:`fig_t5-encoder-decoder`,
trong đó văn bản gốc là "I", "love", "this", "red", "car".
Các token "love", "red", "car" được thay thế ngẫu nhiên bằng các token đặc biệt.
Vì "red" và "car" là một span liên tiếp,
chúng được thay thế bằng cùng một token đặc biệt.
Kết quả là, chuỗi đầu vào là "I", "&lt;X&gt;", "this", "&lt;Y&gt;",
và chuỗi đích là
"&lt;X&gt;", "love", "&lt;Y&gt;", "red", "car", "&lt;Z&gt;",
trong đó "&lt;Z&gt;" là một token đặc biệt khác đánh dấu điểm kết thúc.
Như minh họa trong :numref:`fig_t5-encoder-decoder`,
decoder có mẫu attention nhân quả để ngăn nó
attention đến các token trong tương lai trong khi dự đoán chuỗi.

Trong T5, việc dự đoán span liên tiếp cũng được gọi là
tái tạo văn bản bị hỏng.
Với mục tiêu này, T5 được pretrain
với 1000 tỷ token từ dữ liệu C4
(Colossal Clean Crawled Corpus),
bao gồm văn bản tiếng Anh sạch
từ web :cite:`raffel2020exploring`.

### Fine-Tuning T5

Tương tự BERT, T5 cần được fine-tune (cập nhật các tham số T5)
trên dữ liệu huấn luyện dành riêng cho tác vụ để thực hiện tác vụ này.
Các điểm khác biệt chính so với fine-tuning BERT bao gồm:
(i) đầu vào của T5 bao gồm mô tả tác vụ;
(ii) T5 có thể tạo các chuỗi
có độ dài tùy ý
bằng Transformer decoder của nó;
(iii) Không cần thêm layer nào.

![Fine-tune T5 cho tác vụ tóm tắt văn bản. Cả phần mô tả tác vụ và các token của bài viết đều được đưa vào Transformer encoder để dự đoán bản tóm tắt.](../img/t5-finetune-summarization.svg)
:label:`fig_t5-finetune-summarization`

:numref:`fig_t5-finetune-summarization`
giải thích cách fine-tune T5
lấy tóm tắt văn bản làm ví dụ.
Trong tác vụ downstream này,
các token mô tả tác vụ "Summarize", ":"
theo sau bởi các token của bài viết được đưa vào encoder.

Sau khi fine-tune, T5 với 11 tỷ tham số (T5-11B)
đạt kết quả state-of-the-art trên nhiều benchmark về encoding (ví dụ: classification)
và generation (ví dụ: summarization).
Kể từ khi được phát hành, T5 đã được sử dụng rộng rãi trong các nghiên cứu sau đó.
Ví dụ, switch Transformers được thiết kế dựa trên T5
để kích hoạt một tập con các tham số
nhằm đạt hiệu quả tính toán tốt hơn :cite:`fedus2022switch`.
Trong một mô hình text-to-image có tên Imagen,
văn bản được đưa vào một T5 encoder (T5-XXL) đã được frozen
với 4,6 tỷ tham số :cite:`saharia2022photorealistic`.
Các ví dụ text-to-image chân thực như ảnh chụp trong :numref:`fig_imagen`
cho thấy rằng riêng T5 encoder có thể biểu diễn hiệu quả
văn bản ngay cả khi không fine-tune.

![Các ví dụ text-to-image của mô hình Imagen, có text encoder từ T5 (các hình được lấy từ :citet:`saharia2022photorealistic`).](../img/imagen.png)
:width:`700px`
:label:`fig_imagen`


## Chỉ Decoder


Chúng ta đã xem xét các Transformer chỉ encoder và encoder--decoder.
Ngoài ra, Transformer chỉ decoder
loại bỏ toàn bộ encoder và sublayer decoder
có encoder--decoder cross-attention
khỏi kiến trúc encoder--decoder ban đầu
được minh họa trong :numref:`fig_transformer`.
Ngày nay, Transformer chỉ decoder đã trở thành kiến trúc *de facto*
trong language modeling quy mô lớn (:numref:`sec_language-model`),
tận dụng các kho văn bản không gán nhãn phong phú của thế giới thông qua self-supervised learning.



### GPT và GPT-2

Sử dụng language modeling làm mục tiêu huấn luyện,
mô hình GPT (generative pre-training)
chọn một Transformer decoder
làm backbone :cite:`Radford.Narasimhan.Salimans.ea.2018`.

![Bên trái: Pretrain GPT bằng language modeling. Chuỗi đích là chuỗi đầu vào được dịch đi một token. Cả “&lt;bos&gt;” và “&lt;eos&gt;” đều là các token đặc biệt lần lượt đánh dấu điểm bắt đầu và kết thúc của chuỗi. Bên phải: Mẫu attention trong Transformer decoder. Mỗi token dọc theo trục dọc chỉ attention đến các token trước đó của nó dọc theo trục ngang (nhân quả).](../img/gpt-decoder-only.svg)
:label:`fig_gpt-decoder-only`

Theo phương pháp huấn luyện autoregressive language model
được mô tả trong :numref:`subsec_partitioning-seqs`,
:numref:`fig_gpt-decoder-only` minh họa
việc pretrain GPT bằng một Transformer encoder,
trong đó chuỗi đích là chuỗi đầu vào được dịch đi một token.
Lưu ý rằng mẫu attention trong Transformer decoder
buộc mỗi token chỉ có thể attention đến các token trước đó của nó
(các token trong tương lai không thể được attention vì chúng chưa được chọn).


GPT có 100 triệu tham số và cần được
fine-tune cho từng tác vụ downstream.
Một language model Transformer-decoder lớn hơn nhiều,
GPT-2, được giới thiệu một năm sau đó :cite:`Radford.Wu.Child.ea.2019`.
So với Transformer decoder ban đầu trong GPT, pre-normalization
(được thảo luận trong :numref:`subsec_vit-encoder`)
cùng với việc khởi tạo và weight-scaling cải tiến đã được áp dụng trong GPT-2.
Được pretrain trên 40 GB văn bản, GPT-2 với 1,5 tỷ tham số
đạt kết quả state-of-the-art trên các benchmark language modeling
và kết quả đầy hứa hẹn trên nhiều tác vụ khác
*mà không cập nhật tham số hoặc kiến trúc*.


### GPT-3 và các mô hình tiếp theo

GPT-2 đã chứng minh tiềm năng sử dụng cùng một language model
cho nhiều tác vụ mà không cập nhật mô hình.
Điều này hiệu quả tính toán hơn fine-tuning,
vốn yêu cầu cập nhật mô hình thông qua tính toán gradient.


![In-context learning zero-shot, one-shot, few-shot với các language model (Transformer decoder). Không cần cập nhật tham số.](../img/gpt-3-xshot.svg)
:label:`fig_gpt-3-xshot`

Trước khi giải thích cách sử dụng hiệu quả tính toán hơn
của language model mà không cập nhật tham số,
hãy nhớ lại trong :numref:`sec_rnn-scratch` rằng một language model
có thể được huấn luyện để tạo một chuỗi văn bản
có điều kiện theo một chuỗi văn bản tiền tố nào đó.
Do đó, một language model đã pretrain có thể tạo đầu ra tác vụ
dưới dạng một chuỗi *mà không cập nhật tham số*,
có điều kiện theo một chuỗi đầu vào với mô tả tác vụ,
các ví dụ đầu vào--đầu ra dành riêng cho tác vụ và một prompt (đầu vào tác vụ).
Mô hình học này được gọi là *in-context learning* :cite:`brown2020language`,
và có thể được phân loại thêm
thành *zero-shot*, *one-shot*, và *few-shot*,
khi không có, có một, và có một vài ví dụ đầu vào--đầu ra dành riêng cho tác vụ (:numref:`fig_gpt-3-xshot`).


![Hiệu năng tổng hợp của GPT-3 trên toàn bộ 42 benchmark được đánh giá bằng accuracy (chú thích được điều chỉnh và hình được lấy từ :citet:`brown2020language`).](../img/gpt3-xshot-scaling.png)
:width:`400px`
:label:`fig_gpt3-xshot-scaling`

Ba thiết lập này đã được thử nghiệm trong GPT-3 :cite:`brown2020language`,
trong đó phiên bản lớn nhất sử dụng dữ liệu và kích thước mô hình
lớn hơn khoảng hai bậc độ lớn so với GPT-2.
GPT-3 sử dụng cùng kiến trúc Transformer decoder
như phiên bản tiền nhiệm trực tiếp GPT-2
ngoại trừ việc các mẫu attention
(ở bên phải trong :numref:`fig_gpt-decoder-only`)
thưa hơn tại các layer xen kẽ.
Được pretrain với 300 tỷ token,
GPT-3 hoạt động tốt hơn khi kích thước mô hình lớn hơn,
trong đó hiệu năng few-shot tăng nhanh nhất (:numref:`fig_gpt3-xshot-scaling`).

Mô hình GPT-4 tiếp theo đã không công bố đầy đủ các chi tiết kỹ thuật trong báo cáo của mình :cite:`openai2023gpt4`.
Trái với các mô hình tiền nhiệm, GPT-4
là một mô hình đa phương thức quy mô lớn
có thể nhận cả văn bản và hình ảnh làm đầu vào
và tạo đầu ra văn bản.


## Khả năng mở rộng

:numref:`fig_gpt3-xshot-scaling` minh họa thực nghiệm khả năng mở rộng
của Transformer trong language model GPT-3.
Đối với language modeling, các nghiên cứu thực nghiệm toàn diện hơn
về khả năng mở rộng của Transformer đã khiến các nhà nghiên cứu thấy được tiềm năng
trong việc huấn luyện các Transformer lớn hơn với nhiều dữ liệu và tài nguyên tính toán hơn :cite:`kaplan2020scaling`.

![Hiệu năng của language model Transformer cải thiện mượt mà khi chúng ta tăng kích thước mô hình, kích thước dataset và lượng tài nguyên tính toán dùng cho huấn luyện. Để đạt hiệu năng tối ưu, cả ba yếu tố phải được mở rộng đồng thời. Hiệu năng thực nghiệm có quan hệ theo luật lũy thừa với từng yếu tố riêng lẻ khi không bị giới hạn bởi hai yếu tố còn lại (chú thích được điều chỉnh và hình được lấy từ :citet:`kaplan2020scaling`).](../img/scaling-power-law.png)
:width:`700px`
:label:`fig_scaling-power-law3`

Như minh họa trong :numref:`fig_scaling-power-law3`,
có thể quan sát thấy *mở rộng theo luật lũy thừa* trong hiệu năng
theo kích thước mô hình (số lượng tham số, không bao gồm embedding layer),
kích thước dataset (số lượng token huấn luyện),
và lượng tài nguyên tính toán huấn luyện (PetaFLOP/s-ngày, không bao gồm embedding layer).
Nhìn chung, tăng đồng thời cả ba yếu tố này sẽ mang lại hiệu năng tốt hơn.
Tuy nhiên, *cách* tăng chúng đồng thời
vẫn còn là vấn đề gây tranh luận :cite:`hoffmann2022training`.

![Các lần chạy huấn luyện language model Transformer (hình được lấy từ :citet:`kaplan2020scaling`).](../img/scaling-sample-conv.png)
:width:`700px`
:label:`fig_scaling-sample-conv`

Bên cạnh hiệu năng tăng lên, các mô hình lớn cũng có hiệu quả mẫu tốt hơn các mô hình nhỏ. :numref:`fig_scaling-sample-conv` cho thấy các mô hình lớn cần ít mẫu huấn luyện hơn (các token được xử lý) để đạt cùng mức hiệu năng của các mô hình nhỏ, và hiệu năng được mở rộng mượt mà theo tài nguyên tính toán.



![Hiệu năng của GPT-3 (mất mát xác thực cross-entropy) tuân theo xu hướng luật lũy thừa (power-law) theo lượng compute được sử dụng để huấn luyện. Hành vi luật lũy thừa được quan sát trong :citet:`kaplan2020scaling` tiếp tục thêm hai bậc độ lớn, chỉ có những sai lệch nhỏ so với đường cong dự đoán. Các tham số embedding được loại khỏi số lượng compute và số lượng tham số (chú thích được điều chỉnh và hình được lấy từ :citet:`brown2020language`).](../img/scaling-gpt3.png)
:width:`250px`
:label:`fig_scaling-gpt3`


Các hành vi mở rộng theo thực nghiệm trong :citet:`kaplan2020scaling` đã được kiểm chứng trên các mô hình Transformer lớn sau đó. Ví dụ, GPT-3 đã hỗ trợ giả thuyết này với thêm hai bậc độ lớn trong :numref:`fig_scaling-gpt3`.





## Các Mô Hình Ngôn Ngữ Lớn

Khả năng mở rộng của Transformers trong dòng GPT đã truyền cảm hứng cho các mô hình ngôn ngữ lớn sau này. 
Transformer decoder của GPT-2 được dùng để huấn luyện Megatron-Turing NLG :cite:`smith2022using` với 530 tỷ tham số trên 270 tỷ token huấn luyện. Theo thiết kế của GPT-2, Gopher :cite:`rae2021scaling` với 280 tỷ tham số, được pretrain bằng 300 tỷ token, đã đạt hiệu năng cạnh tranh trên nhiều tác vụ đa dạng. 
Kế thừa cùng kiến trúc và sử dụng cùng ngân sách compute như Gopher, Chinchilla :cite:`hoffmann2022training` là một mô hình nhỏ hơn đáng kể (70 tỷ tham số) nhưng được huấn luyện lâu hơn nhiều (1,4 nghìn tỷ token huấn luyện), vượt Gopher trên nhiều tác vụ và chú trọng số lượng token hơn số lượng tham số.
Để tiếp tục xu hướng mở rộng của mô hình hóa ngôn ngữ, 
PaLM (Pathway Language Model) :cite:`chowdhery2022palm`, một Transformer decoder 540 tỷ tham số với các thiết kế được sửa đổi, được pretrain trên 780 tỷ token, đã vượt hiệu năng trung bình của con người trên benchmark BIG-Bench :cite:`srivastava2022beyond`. Phiên bản sau đó, PaLM 2 :cite:`anil2023palm`, đã mở rộng dữ liệu và mô hình theo tỷ lệ xấp xỉ 1:1, đồng thời cải thiện các năng lực đa ngôn ngữ và suy luận. 
Các mô hình ngôn ngữ lớn khác, chẳng hạn như Minerva :cite:`lewkowycz2022solving` tiếp tục huấn luyện một mô hình tổng quát (PaLM), và Galactica :cite:`taylor2022galactica` không được huấn luyện trên một corpus tổng quát, đã cho thấy các năng lực đầy hứa hẹn về suy luận định lượng và khoa học.


Các bản phát hành open-source, như OPT (Open Pretrained Transformers) :cite:`zhang2022opt`, BLOOM :cite:` scao2022bloom`, và FALCON :cite:`penedo2023refinedweb`,
đã dân chủ hóa việc nghiên cứu và sử dụng các mô hình ngôn ngữ lớn.
Tập trung vào hiệu quả tính toán tại thời điểm inference,
Llama 1 :cite:`touvron2023llama` open-source đã vượt các mô hình lớn hơn nhiều nhờ huấn luyện trên số lượng token nhiều hơn mức thường được sử dụng. Llama 2 :cite:`touvron2023llama2` được cập nhật tiếp tục tăng corpus pretraining thêm 40%, dẫn đến các mô hình sản phẩm có thể sánh ngang hiệu năng của các mô hình close-source cạnh tranh. 



:citet:`wei2022emergent` đã thảo luận về các năng lực mới nổi của các mô hình ngôn ngữ lớn, vốn hiện diện ở các mô hình lớn hơn nhưng không có ở các mô hình nhỏ hơn.
Tuy nhiên, chỉ đơn giản tăng kích thước mô hình không tự động khiến mô hình tuân theo chỉ dẫn của con người tốt hơn.
:citet:`wei2021finetuned,sanh2021multitask` nhận thấy rằng fine-tuning các mô hình ngôn ngữ lớn
trên một loạt dataset được mô tả bằng *chỉ dẫn*
có thể cải thiện hiệu năng zero-shot trên các tác vụ held-out.
Sử dụng *reinforcement learning from human feedback*,
:citet:`ouyang2022training` đã fine-tune GPT-3
để tuân theo một tập hợp chỉ dẫn đa dạng.
Theo InstructGPT thu được, mô hình
căn chỉnh các mô hình ngôn ngữ với ý định của con người
thông qua fine-tuning :cite:`ouyang2022training`,
[ChatGPT](https://chat.openai.com/)
có thể tạo ra các phản hồi giống con người (ví dụ: debug code và viết sáng tạo)
dựa trên các cuộc hội thoại với con người
và có thể thực hiện nhiều tác vụ xử lý ngôn ngữ tự nhiên
theo cách zero-shot :cite:`qin2023chatgpt`.
:citet:`bai2022constitutional` đã thay thế đầu vào của con người (ví dụ: dữ liệu do con người gán nhãn) bằng đầu ra của mô hình
để tự động hóa một phần quy trình instruction tuning, còn được gọi là *reinforcement learning from AI feedback*.


Các mô hình ngôn ngữ lớn mang lại triển vọng thú vị
về việc xây dựng đầu vào văn bản để khiến mô hình thực hiện các tác vụ mong muốn thông qua in-context learning,
còn được gọi là *prompting*.
Đáng chú ý,
*chain-of-thought prompting* :cite:`wei2022chain`,
một phương pháp in-context learning
với các ví dụ few-shot theo dạng "câu hỏi, các bước suy luận trung gian, câu trả lời",
khơi gợi các năng lực suy luận phức tạp của
các mô hình ngôn ngữ lớn
để giải quyết các tác vụ suy luận toán học, thường thức và ký hiệu.
Lấy mẫu nhiều đường suy luận :cite:`wang2023self`, đa dạng hóa các ví dụ few-shot :cite:`zhang2023automatic`, 
và giảm các vấn đề phức tạp thành các bài toán con :cite:`zhou2023least`
đều có thể cải thiện độ chính xác suy luận. Thực tế, chỉ với các prompt đơn giản như "Let's think step by step" ngay trước mỗi câu trả lời,
các mô hình ngôn ngữ lớn thậm chí có thể thực hiện suy luận chain-of-thought *zero-shot*
với độ chính xác khá tốt :cite:`kojima2022large`.
Ngay cả với đầu vào đa phương thức bao gồm cả văn bản và hình ảnh,
các mô hình ngôn ngữ có thể thực hiện suy luận chain-of-thought đa phương thức với độ chính xác cao hơn so với chỉ sử dụng đầu vào văn bản :cite:`zhang2023multicot`.




## Tóm Tắt và Thảo Luận

Transformers đã được pretrain dưới dạng chỉ encoder (ví dụ: BERT), encoder--decoder (ví dụ: T5), và chỉ decoder (ví dụ: dòng GPT). Các mô hình pretrain có thể được điều chỉnh để thực hiện các tác vụ khác nhau có cập nhật mô hình (ví dụ: fine-tuning) hoặc không cần cập nhật (ví dụ: few-shot). Khả năng mở rộng của Transformers cho thấy hiệu năng tốt hơn được hưởng lợi từ các mô hình lớn hơn, nhiều dữ liệu huấn luyện hơn và nhiều compute huấn luyện hơn. Vì Transformers lần đầu được thiết kế và pretrain cho dữ liệu văn bản, phần này hơi thiên về xử lý ngôn ngữ tự nhiên. Tuy nhiên, các mô hình được thảo luận ở trên thường có thể được tìm thấy trong các mô hình gần đây hơn trên nhiều modality. Ví dụ,
(i) Chinchilla :cite:`hoffmann2022training` sau đó được mở rộng thành Flamingo :cite:`alayrac2022flamingo`, một mô hình ngôn ngữ thị giác cho few-shot learning;
(ii) GPT-2 :cite:`Radford.Wu.Child.ea.2019` và vision Transformer mã hóa văn bản và hình ảnh trong CLIP (Contrastive Language-Image Pre-training) :cite:`radford2021learning`, mà các embedding hình ảnh và văn bản của nó sau đó được áp dụng trong hệ thống text-to-image DALL-E 2 :cite:`ramesh2022hierarchical`. Mặc dù vẫn chưa có các nghiên cứu hệ thống về khả năng mở rộng của Transformer trong pretraining đa phương thức, một mô hình text-to-image hoàn toàn dựa trên Transformer có tên Parti :cite:`yu2022scaling` cho thấy tiềm năng mở rộng trên các modality:
Parti lớn hơn có khả năng tạo ảnh có độ trung thực cao và hiểu văn bản giàu nội dung tốt hơn (:numref:`fig_parti`).


![Các ví dụ hình ảnh được tạo từ cùng một văn bản bởi mô hình Parti có kích thước tăng dần (350M, 750M, 3B, 20B) (các ví dụ lấy từ :citet:`yu2022scaling`).](../img/parti.png)
:width:`700px`
:label:`fig_parti`




## Bài Tập

1. Có thể fine-tune T5 bằng một minibatch gồm các tác vụ khác nhau không? Vì sao có hoặc không? Còn GPT-2 thì sao?
1. Với một mô hình ngôn ngữ mạnh, bạn có thể nghĩ đến những ứng dụng nào?
1. Giả sử bạn được yêu cầu fine-tune một mô hình ngôn ngữ để thực hiện phân loại văn bản bằng cách thêm các layer bổ sung. Bạn sẽ thêm chúng ở đâu? Vì sao?
1. Hãy xem xét các bài toán sequence-to-sequence (ví dụ: dịch máy), trong đó chuỗi đầu vào luôn có sẵn trong toàn bộ quá trình dự đoán chuỗi đích. Việc mô hình hóa bằng Transformers chỉ có decoder có thể gặp những hạn chế nào? Vì sao?


[Thảo Luận](https://discuss.d2l.ai/t/9232)