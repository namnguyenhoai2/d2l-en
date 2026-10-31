# Mạng nơ-ron hồi quy
:label:`chap_rnn`

Cho đến nay, chúng ta chủ yếu tập trung vào dữ liệu có độ dài cố định.
Khi giới thiệu hồi quy tuyến tính và hồi quy logistic
trong :numref:`chap_regression` và :numref:`chap_classification`
cùng perceptron đa tầng trong :numref:`chap_perceptrons`,
chúng ta có thể giả định rằng mỗi vector đặc trưng $\mathbf{x}_i$
bao gồm một số lượng cố định các thành phần $x_1, \dots, x_d$,
trong đó mỗi đặc trưng số $x_j$
tương ứng với một thuộc tính cụ thể.
Các bộ dữ liệu này đôi khi được gọi là *dạng bảng*,
vì chúng có thể được sắp xếp thành các bảng,
trong đó mỗi ví dụ $i$ có một hàng riêng,
và mỗi thuộc tính có một cột riêng.
Điều quan trọng là, với dữ liệu dạng bảng, chúng ta hiếm khi
giả định bất kỳ cấu trúc cụ thể nào giữa các cột.

Sau đó, trong :numref:`chap_cnn`,
chúng ta chuyển sang dữ liệu ảnh, trong đó đầu vào bao gồm
các giá trị pixel thô tại từng tọa độ trong ảnh.
Dữ liệu ảnh hầu như không phù hợp với mô hình
của một bộ dữ liệu dạng bảng điển hình.
Ở đó, chúng ta cần sử dụng mạng nơ-ron tích chập (CNN)
để xử lý cấu trúc phân cấp và các tính bất biến.
Tuy nhiên, dữ liệu của chúng ta vẫn có độ dài cố định.
Mỗi ảnh Fashion-MNIST được biểu diễn
dưới dạng lưới $28 \times 28$ các giá trị pixel.
Hơn nữa, mục tiêu của chúng ta là phát triển một mô hình
chỉ xem một ảnh rồi
đưa ra một dự đoán duy nhất.
Nhưng chúng ta nên làm gì khi phải xử lý
một chuỗi ảnh, như trong video,
hoặc khi được giao nhiệm vụ tạo ra
một dự đoán có cấu trúc tuần tự,
như trong trường hợp tạo chú thích ảnh?

Rất nhiều nhiệm vụ học máy đòi hỏi phải xử lý dữ liệu tuần tự.
Tạo chú thích ảnh, tổng hợp giọng nói và tạo nhạc
đều yêu cầu mô hình tạo ra đầu ra bao gồm các chuỗi.
Trong những lĩnh vực khác, chẳng hạn như dự đoán chuỗi thời gian,
phân tích video và truy xuất thông tin âm nhạc,
một mô hình phải học từ các đầu vào là chuỗi.
Các yêu cầu này thường xuất hiện đồng thời:
những nhiệm vụ như dịch các đoạn văn bản
từ ngôn ngữ tự nhiên này sang ngôn ngữ tự nhiên khác,
tham gia hội thoại hoặc điều khiển robot,
đòi hỏi mô hình vừa tiếp nhận vừa tạo ra
dữ liệu có cấu trúc tuần tự.


Mạng nơ-ron hồi quy (RNN) là các mô hình deep learning
nắm bắt động lực của chuỗi thông qua
các kết nối *hồi quy*, có thể được xem như
các chu trình trong mạng lưới nút.
Thoạt đầu điều này có thể có vẻ phản trực giác.
Xét cho cùng, chính tính feedforward của mạng nơ-ron
làm cho thứ tự tính toán trở nên rõ ràng.
Tuy nhiên, các cạnh hồi quy được định nghĩa một cách chính xác
để bảo đảm không thể xảy ra sự mơ hồ như vậy.
Mạng nơ-ron hồi quy được *trải* theo các bước thời gian (hoặc bước chuỗi),
với *cùng* các tham số nền tảng được áp dụng tại mỗi bước.
Trong khi các kết nối chuẩn được áp dụng *đồng bộ*
để truyền các kích hoạt của từng tầng
đến tầng tiếp theo *trong cùng một bước thời gian*,
các kết nối hồi quy là *động*,
truyền thông tin qua các bước thời gian kề nhau.
Như góc nhìn được mở ra trong :numref:`fig_unfolded-rnn` cho thấy,
RNN có thể được xem là mạng nơ-ron feedforward
trong đó các tham số của mỗi tầng (cả thông thường lẫn hồi quy)
được chia sẻ qua các bước thời gian.


![Bên trái, các kết nối hồi quy được minh họa bằng các cạnh tuần hoàn. Bên phải, chúng ta trải RNN theo các bước thời gian. Ở đây, các cạnh hồi quy trải qua các bước thời gian kề nhau, trong khi các kết nối thông thường được tính toán đồng bộ.](../img/unfolded-rnn.svg)
:label:`fig_unfolded-rnn`


Giống như mạng nơ-ron nói chung,
RNN có một lịch sử lâu dài trải rộng qua nhiều lĩnh vực, 
khởi nguồn là các mô hình về não bộ được các nhà khoa học nhận thức phổ biến hóa
và sau đó được cộng đồng machine learning sử dụng
như các công cụ mô hình hóa thực tiễn.
Cũng như cách chúng ta tiếp cận deep learning nói chung,
trong cuốn sách này, chúng ta theo góc nhìn machine learning,
tập trung vào RNN như những công cụ thực tiễn trở nên
phổ biến vào thập niên 2010 nhờ
các kết quả đột phá trong những nhiệm vụ đa dạng như
nhận dạng chữ viết tay :cite:`graves2008novel`,
dịch máy :cite:`Sutskever.Vinyals.Le.2014`,
và nhận diện chẩn đoán y khoa :cite:`Lipton.Kale.2016`.
Chúng tôi hướng những độc giả quan tâm đến
tài liệu nền tảng hơn tới một bài tổng quan toàn diện
được công khai :cite:`Lipton.Berkowitz.Elkan.2015`.
Chúng tôi cũng lưu ý rằng tính tuần tự không chỉ có ở RNN.
Ví dụ, các CNN mà chúng ta đã giới thiệu
có thể được điều chỉnh để xử lý dữ liệu có độ dài thay đổi,
chẳng hạn ảnh có độ phân giải khác nhau.
Hơn nữa, gần đây RNN đã nhường một phần đáng kể
thị phần cho các mô hình Transformer,
sẽ được trình bày trong :numref:`chap_attention-and-transformers`.
Tuy nhiên, RNN đã trở nên nổi bật như các mô hình mặc định
để xử lý cấu trúc tuần tự phức tạp trong deep learning,
và cho đến ngày nay vẫn là các mô hình chủ lực cho mô hình hóa tuần tự.
Câu chuyện về RNN và về mô hình hóa chuỗi
gắn bó không thể tách rời, và đây vừa là
một chương về những điều cơ bản của các bài toán mô hình hóa chuỗi
vừa là một chương về RNN.


Một nhận thức then chốt đã mở đường cho một cuộc cách mạng trong mô hình hóa chuỗi.
Mặc dù đầu vào và mục tiêu của nhiều nhiệm vụ cơ bản trong machine learning
không thể dễ dàng biểu diễn dưới dạng vector có độ dài cố định,
chúng thường vẫn có thể được biểu diễn dưới dạng
các chuỗi có độ dài thay đổi gồm các vector có độ dài cố định.
Ví dụ, tài liệu có thể được biểu diễn dưới dạng chuỗi từ;
hồ sơ y tế thường có thể được biểu diễn dưới dạng chuỗi sự kiện
(lần khám, thuốc men, thủ thuật, xét nghiệm, chẩn đoán);
video có thể được biểu diễn dưới dạng các chuỗi ảnh tĩnh có độ dài thay đổi.


Mặc dù mô hình chuỗi đã xuất hiện trong nhiều lĩnh vực ứng dụng,
nghiên cứu cơ bản trong lĩnh vực này chủ yếu được thúc đẩy
bởi các tiến bộ trong những nhiệm vụ cốt lõi của xử lý ngôn ngữ tự nhiên.
Do đó, xuyên suốt chương này, chúng ta sẽ tập trung
phần trình bày và các ví dụ vào dữ liệu văn bản.
Nếu bạn nắm vững các ví dụ này,
thì việc áp dụng các mô hình cho những dạng dữ liệu khác
sẽ tương đối đơn giản.
Trong vài phần tiếp theo, chúng ta giới thiệu
ký hiệu cơ bản cho chuỗi và một số thước đo đánh giá
để đánh giá chất lượng đầu ra mô hình có cấu trúc tuần tự.
Sau đó, chúng ta thảo luận các khái niệm cơ bản của mô hình ngôn ngữ
và dùng phần thảo luận này để thúc đẩy các mô hình RNN đầu tiên.
Cuối cùng, chúng ta mô tả phương pháp tính gradient
khi backpropagation qua RNN và khảo sát một số thách thức
thường gặp khi huấn luyện những mạng như vậy,
từ đó thúc đẩy các kiến trúc RNN hiện đại sẽ được trình bày tiếp
trong :numref:`chap_modern_rnn`.

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
rnn-scratch
rnn-concise
bptt
```
