# Fine-Tuning BERT cho các ứng dụng ở cấp độ chuỗi và cấp độ token
:label:`sec_finetuning-bert`


Trong các phần trước của chương này,
chúng ta đã thiết kế nhiều mô hình khác nhau cho các ứng dụng xử lý ngôn ngữ tự nhiên,
chẳng hạn như các mô hình dựa trên RNN, CNN, attention và MLP.
Những mô hình này hữu ích khi bị giới hạn về không gian hoặc thời gian,
tuy nhiên,
việc xây dựng một mô hình cụ thể cho từng tác vụ xử lý ngôn ngữ tự nhiên
trên thực tế là không khả thi.
Trong :numref:`sec_bert`,
chúng ta đã giới thiệu một mô hình pretraining là BERT,
đòi hỏi rất ít thay đổi về kiến trúc
cho nhiều tác vụ xử lý ngôn ngữ tự nhiên.
Một mặt,
tại thời điểm được đề xuất,
BERT đã cải thiện state of the art trên nhiều tác vụ xử lý ngôn ngữ tự nhiên.
Mặt khác,
như đã lưu ý trong :numref:`sec_bert-pretraining`,
hai phiên bản của mô hình BERT ban đầu
có lần lượt 110 triệu và 340 triệu tham số.
Do đó, khi có đủ tài nguyên tính toán,
chúng ta có thể cân nhắc
fine-tuning BERT cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream.

Trong phần sau,
chúng ta khái quát một nhóm các ứng dụng xử lý ngôn ngữ tự nhiên
thành các ứng dụng ở cấp độ chuỗi và cấp độ token.
Ở cấp độ chuỗi,
chúng ta giới thiệu cách chuyển đổi biểu diễn BERT của đầu vào văn bản
thành nhãn đầu ra
trong phân loại văn bản đơn
và phân loại hoặc hồi quy cặp văn bản.
Ở cấp độ token, chúng ta sẽ giới thiệu ngắn gọn các ứng dụng mới
như gán nhãn văn bản và question answering,
đồng thời làm rõ cách BERT biểu diễn đầu vào của chúng và chuyển đổi thành các nhãn đầu ra.
Trong quá trình fine-tuning,
“thay đổi tối thiểu về kiến trúc” mà BERT yêu cầu cho các ứng dụng khác nhau
là các lớp fully connected bổ sung.
Trong quá trình supervised learning của một ứng dụng downstream,
các tham số của những lớp bổ sung được học từ đầu,
còn toàn bộ tham số trong mô hình BERT đã pretrain được fine-tune.


## Phân loại văn bản đơn

*Phân loại văn bản đơn* nhận một chuỗi văn bản duy nhất làm đầu vào và xuất ra kết quả phân loại.
Bên cạnh sentiment analysis mà chúng ta đã nghiên cứu trong chương này,
Corpus of Linguistic Acceptability (CoLA)
cũng là một dataset cho phân loại văn bản đơn,
nhằm đánh giá xem một câu cho trước có được chấp nhận về mặt ngữ pháp hay không :cite:`Warstadt.Singh.Bowman.2019`.
Ví dụ, “I should study.” được chấp nhận, nhưng “I should studying.” thì không.

![Fine-tuning BERT cho các ứng dụng phân loại văn bản đơn, chẳng hạn như sentiment analysis và kiểm tra tính chấp nhận được về mặt ngôn ngữ. Giả sử văn bản đơn đầu vào có sáu token.](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

:numref:`sec_bert` mô tả biểu diễn đầu vào của BERT.
Chuỗi đầu vào BERT biểu diễn rõ ràng cả văn bản đơn lẫn cặp văn bản,
trong đó token phân loại đặc biệt
“&lt;cls&gt;” được sử dụng cho việc phân loại chuỗi
và token phân loại đặc biệt
“&lt;sep&gt;” đánh dấu phần cuối của văn bản đơn hoặc phân tách một cặp văn bản.
Như minh họa trong :numref:`fig_bert-one-seq`,
trong các ứng dụng phân loại văn bản đơn,
biểu diễn của token phân loại đặc biệt
“&lt;cls&gt;” mã hóa thông tin của toàn bộ chuỗi văn bản đầu vào.
Với tư cách là biểu diễn của văn bản đơn đầu vào,
nó sẽ được đưa vào một MLP nhỏ gồm các lớp fully connected (dense)
để xuất ra phân phối của tất cả các giá trị nhãn rời rạc.


## Phân loại hoặc hồi quy cặp văn bản

Trong chương này, chúng ta cũng đã tìm hiểu về natural language inference.
Đây là một dạng *phân loại cặp văn bản*,
tức là một loại ứng dụng phân loại một cặp văn bản.

Nhận một cặp văn bản làm đầu vào nhưng xuất ra một giá trị liên tục,
*semantic textual similarity* là một tác vụ *hồi quy cặp văn bản* phổ biến.
Tác vụ này đo lường độ tương đồng ngữ nghĩa giữa các câu.
Ví dụ, trong dataset Semantic Textual Similarity Benchmark,
điểm tương đồng của một cặp câu
là một thang đo thứ bậc từ 0 (không có phần nghĩa trùng lặp) đến 5 (tương đương về nghĩa) :cite:`Cer.Diab.Agirre.ea.2017`.
Mục tiêu là dự đoán các điểm số này.
Các ví dụ từ dataset Semantic Textual Similarity Benchmark gồm (câu 1, câu 2, điểm tương đồng):

* “A plane is taking off.”, “An air plane is taking off.”, 5.000;
* “A woman is eating something.”, “A woman is eating meat.”, 3.000;
* “A woman is dancing.”, “A man is talking.”, 0.000.


![Fine-tuning BERT cho các ứng dụng phân loại hoặc hồi quy cặp văn bản, chẳng hạn như natural language inference và semantic textual similarity. Giả sử cặp văn bản đầu vào lần lượt có hai và ba token.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`

So với phân loại văn bản đơn trong :numref:`fig_bert-one-seq`,
fine-tuning BERT cho phân loại cặp văn bản trong :numref:`fig_bert-two-seqs`
khác biệt ở biểu diễn đầu vào.
Đối với các tác vụ hồi quy cặp văn bản như semantic textual similarity,
có thể áp dụng những thay đổi đơn giản như xuất ra một giá trị nhãn liên tục
và sử dụng mean squared loss: đây là những cách phổ biến cho hồi quy.


## Gán nhãn văn bản

Bây giờ hãy xét các tác vụ ở cấp độ token, chẳng hạn như *gán nhãn văn bản*,
trong đó mỗi token được gán một nhãn.
Trong số các tác vụ gán nhãn văn bản,
*gán nhãn part-of-speech* gán cho mỗi từ một nhãn part-of-speech (ví dụ: tính từ và từ hạn định)
dựa trên vai trò của từ đó trong câu.
Ví dụ,
theo bộ nhãn Penn Treebank II,
câu “John Smith 's car is new”
sẽ được gán nhãn là
“NNP (danh từ riêng, số ít) NNP POS (hậu tố sở hữu) NN (danh từ, số ít hoặc không đếm được) VB (động từ, dạng nguyên mẫu) JJ (tính từ)”.

![Fine-tuning BERT cho các ứng dụng gán nhãn văn bản, chẳng hạn như gán nhãn part-of-speech. Giả sử văn bản đơn đầu vào có sáu token.](../img/bert-tagging.svg)
:label:`fig_bert-tagging`

Fine-tuning BERT cho các ứng dụng gán nhãn văn bản
được minh họa trong :numref:`fig_bert-tagging`.
So với :numref:`fig_bert-one-seq`,
điểm khác biệt duy nhất là
trong gán nhãn văn bản, biểu diễn BERT của *mỗi token* trong văn bản đầu vào
được đưa vào cùng một lớp fully connected bổ sung để xuất ra nhãn của token đó,
chẳng hạn như nhãn part-of-speech.



## Question Answering

Là một ứng dụng khác ở cấp độ token,
*question answering* phản ánh khả năng đọc hiểu.
Ví dụ, Stanford Question Answering Dataset (SQuAD v1.1)
bao gồm các đoạn văn và câu hỏi,
trong đó câu trả lời cho mỗi câu hỏi
chỉ là một đoạn văn bản (text span) từ đoạn văn mà câu hỏi đề cập đến :cite:`Rajpurkar.Zhang.Lopyrev.ea.2016`.
Để giải thích,
hãy xét đoạn văn
“Some experts report that a mask's efficacy is inconclusive. However, mask makers insist that their products, such as N95 respirator masks, can guard against the virus.”
và câu hỏi “Who say that N95 respirator masks can guard against the virus?”.
Câu trả lời phải là text span “mask makers” trong đoạn văn.
Do đó, mục tiêu trong SQuAD v1.1 là dự đoán vị trí bắt đầu và kết thúc của text span trong đoạn văn khi đã cho một cặp câu hỏi và đoạn văn.

![Fine-tuning BERT cho question answering. Giả sử cặp văn bản đầu vào lần lượt có hai và ba token.](../img/bert-qa.svg)
:label:`fig_bert-qa`

Để fine-tune BERT cho question answering,
câu hỏi và đoạn văn lần lượt được đóng gói
thành chuỗi văn bản thứ nhất và thứ hai
trong đầu vào của BERT.
Để dự đoán vị trí bắt đầu của text span,
cùng một lớp fully connected bổ sung sẽ chuyển đổi
biểu diễn BERT của bất kỳ token nào ở vị trí $i$ trong đoạn văn
thành một điểm số vô hướng $s_i$.
Các điểm số của tất cả token trong đoạn văn
sau đó được chuyển đổi bằng phép toán softmax
thành một phân phối xác suất,
sao cho mỗi vị trí token $i$ trong đoạn văn được gán
một xác suất $p_i$ là vị trí bắt đầu của text span.
Việc dự đoán phần kết thúc của text span
được thực hiện tương tự như trên, ngoại trừ việc
các tham số trong lớp fully connected bổ sung của nó
độc lập với các tham số dùng để dự đoán phần bắt đầu.
Khi dự đoán phần kết thúc,
bất kỳ token nào trong đoạn văn ở vị trí $i$
đều được chuyển đổi bởi cùng một lớp fully connected
thành một điểm số vô hướng $e_i$.
:numref:`fig_bert-qa`
minh họa việc fine-tuning BERT cho question answering.

Đối với question answering,
mục tiêu training của supervised learning đơn giản là
tối đa hóa log-likelihood của vị trí bắt đầu và kết thúc đúng.
Khi dự đoán span,
chúng ta có thể tính điểm số $s_i + e_j$ cho một span hợp lệ
từ vị trí $i$ đến vị trí $j$ ($i \leq j$),
và xuất ra span có điểm số cao nhất.


## Tóm tắt

* BERT yêu cầu rất ít thay đổi về kiến trúc (các lớp fully connected bổ sung) cho các ứng dụng xử lý ngôn ngữ tự nhiên ở cấp độ chuỗi và cấp độ token, chẳng hạn như phân loại văn bản đơn (ví dụ: sentiment analysis và kiểm tra tính chấp nhận được về mặt ngôn ngữ), phân loại hoặc hồi quy cặp văn bản (ví dụ: natural language inference và semantic textual similarity), gán nhãn văn bản (ví dụ: gán nhãn part-of-speech) và question answering.
* Trong quá trình supervised learning của một ứng dụng downstream, các tham số của những lớp bổ sung được học từ đầu, còn toàn bộ tham số trong mô hình BERT đã pretrain được fine-tune.


## Bài tập

1. Hãy thiết kế một thuật toán search engine cho các bài báo tin tức. Khi hệ thống nhận được một query (ví dụ: “oil industry during the coronavirus outbreak”), hệ thống sẽ trả về một danh sách các bài báo tin tức được xếp hạng theo mức độ liên quan đến query. Giả sử chúng ta có một tập rất lớn các bài báo tin tức và một số lượng lớn query. Để đơn giản hóa bài toán, giả sử bài báo liên quan nhất đã được gán nhãn cho mỗi query. Làm thế nào chúng ta có thể áp dụng negative sampling (xem :numref:`subsec_negative-sampling`) và BERT trong thiết kế thuật toán?
1. Làm thế nào chúng ta có thể tận dụng BERT trong việc training language model?
1. Chúng ta có thể tận dụng BERT trong machine translation không?

[Thảo luận](https://discuss.d2l.ai/t/396)