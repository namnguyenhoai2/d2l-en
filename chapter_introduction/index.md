# Giới thiệu
:label:`chap_introduction`

Cho đến gần đây, hầu hết mọi chương trình máy tính
mà bạn có thể đã tương tác trong
một ngày bình thường
đều được lập trình dưới dạng một tập hợp quy tắc cứng nhắc,
quy định chính xác cách chương trình phải hoạt động.
Giả sử chúng ta muốn viết một ứng dụng
để quản lý một nền tảng thương mại điện tử.
Sau vài giờ tụm lại quanh bảng trắng
để suy nghĩ về vấn đề,
chúng ta có thể thống nhất những nét chính
của một giải pháp khả thi, chẳng hạn:
(i) người dùng tương tác với ứng dụng thông qua một giao diện
chạy trong trình duyệt web hoặc ứng dụng di động;
(ii) ứng dụng của chúng ta tương tác với một công cụ cơ sở dữ liệu cấp thương mại
để theo dõi trạng thái của từng người dùng và duy trì hồ sơ
về các giao dịch trong quá khứ;
và (iii) ở trung tâm ứng dụng,
*logic nghiệp vụ* (hay có thể gọi là *bộ não*) của ứng dụng
đưa ra một tập hợp quy tắc ánh xạ mọi tình huống có thể hình dung được
sang hành động tương ứng mà chương trình phải thực hiện.

Để xây dựng bộ não cho ứng dụng,
chúng ta có thể liệt kê tất cả các sự kiện phổ biến
mà chương trình cần xử lý.
Ví dụ, bất cứ khi nào một khách hàng nhấp chuột
để thêm một sản phẩm vào giỏ hàng,
chương trình của chúng ta phải thêm một mục
vào bảng cơ sở dữ liệu giỏ hàng,
liên kết ID của người dùng đó
với ID của sản phẩm được yêu cầu.
Sau đó, chúng ta có thể cố gắng lần lượt xem xét
mọi trường hợp đặc biệt có thể xảy ra,
kiểm tra tính phù hợp của các quy tắc
và thực hiện mọi sửa đổi cần thiết.
Điều gì xảy ra nếu một người dùng
bắt đầu mua hàng khi giỏ hàng trống?
Mặc dù hiếm có developer nào làm đúng hoàn toàn
ngay từ lần đầu tiên
(có thể cần chạy thử vài lần để khắc phục các lỗi),
phần lớn chúng ta vẫn có thể viết những chương trình như vậy
và tự tin đưa chúng vào hoạt động
*trước khi* từng gặp một khách hàng thực tế.
Khả năng tự thiết kế thủ công các hệ thống tự động
để vận hành những sản phẩm và hệ thống hoạt động được,
thường là trong các tình huống mới lạ,
là một kỳ tích nhận thức đáng kinh ngạc.
Và khi bạn có thể tạo ra các giải pháp
hoạt động đúng $100\%$ thời gian,
thông thường bạn không nên
bận tâm đến machine learning.

May mắn thay đối với cộng đồng ngày càng lớn
của các nhà khoa học machine learning,
nhiều tác vụ mà chúng ta muốn tự động hóa
không dễ dàng khuất phục trước sự khéo léo của con người.
Hãy tưởng tượng bạn đang tụm lại quanh bảng trắng
với những bộ óc thông minh nhất mà bạn biết,
nhưng lần này bạn đang giải quyết
một trong những bài toán sau:

* Viết một chương trình dự đoán thời tiết ngày mai dựa trên thông tin địa lý, ảnh vệ tinh và một khoảng thời gian gần nhất của dữ liệu thời tiết trong quá khứ.
* Viết một chương trình nhận vào một câu hỏi về sự kiện, được diễn đạt dưới dạng văn bản tự do, và trả lời chính xác câu hỏi đó.
* Viết một chương trình mà khi được cung cấp một hình ảnh, sẽ xác định mọi người được thể hiện trong ảnh và vẽ đường viền quanh từng người.
* Viết một chương trình giới thiệu cho người dùng những sản phẩm mà họ có khả năng yêu thích nhưng khó có thể tình cờ bắt gặp trong quá trình duyệt web tự nhiên.

Đối với những bài toán này,
ngay cả các lập trình viên ưu tú cũng sẽ gặp khó khăn
khi viết lời giải từ đầu.
Lý do có thể khác nhau.
Đôi khi chương trình mà chúng ta cần tìm
tuân theo một khuôn mẫu thay đổi theo thời gian,
nên không có câu trả lời đúng cố định nào cả!
Trong những trường hợp như vậy, mọi lời giải thành công
đều phải thích ứng linh hoạt với một thế giới đang thay đổi.
Những lúc khác, mối quan hệ (chẳng hạn giữa các pixel
và các danh mục trừu tượng) có thể quá phức tạp,
đòi hỏi hàng nghìn hoặc hàng triệu phép tính
và tuân theo những nguyên lý chưa được biết đến.
Trong trường hợp nhận dạng hình ảnh,
các bước chính xác cần thiết để thực hiện tác vụ
nằm ngoài khả năng hiểu biết có ý thức của chúng ta,
mặc dù các quá trình nhận thức tiềm thức
vẫn thực hiện tác vụ đó một cách dễ dàng.

*Machine learning* là lĩnh vực nghiên cứu các thuật toán
có khả năng học hỏi từ kinh nghiệm.
Khi một thuật toán machine learning tích lũy thêm kinh nghiệm,
thường dưới dạng dữ liệu quan sát
hoặc các tương tác với một môi trường,
hiệu năng của nó được cải thiện.
Hãy đối chiếu điều này với nền tảng thương mại điện tử tất định của chúng ta,
vốn tuân theo cùng một logic nghiệp vụ,
bất kể tích lũy được bao nhiêu kinh nghiệm,
cho đến khi chính các developer học hỏi và quyết định
đã đến lúc cập nhật phần mềm.
Trong cuốn sách này, chúng tôi sẽ hướng dẫn bạn
về những nền tảng của machine learning,
đặc biệt tập trung vào *deep learning*,
một tập hợp kỹ thuật mạnh mẽ
đang thúc đẩy những đổi mới trong các lĩnh vực đa dạng như thị giác máy tính,
xử lý ngôn ngữ tự nhiên, chăm sóc sức khỏe và genomics.

## Một ví dụ tạo động lực

Trước khi bắt đầu viết, các tác giả của cuốn sách này,
giống như phần lớn lực lượng lao động,
đã phải nạp caffeine.
Chúng tôi lên xe và bắt đầu lái.
Sử dụng iPhone, Alex gọi "Hey Siri",
đánh thức hệ thống nhận dạng giọng nói của điện thoại.
Sau đó Mu ra lệnh "directions to Blue Bottle coffee shop".
Điện thoại nhanh chóng hiển thị bản chép lại câu lệnh của anh ấy.
Nó cũng nhận ra rằng chúng tôi đang yêu cầu chỉ đường
và khởi chạy ứng dụng Maps để thực hiện yêu cầu.
Sau khi được khởi chạy, ứng dụng Maps xác định một số tuyến đường.
Bên cạnh mỗi tuyến đường, điện thoại hiển thị thời gian di chuyển dự đoán.
Mặc dù câu chuyện này được dựng lên để thuận tiện cho việc giảng dạy,
nó cho thấy rằng chỉ trong vài giây,
những tương tác hằng ngày của chúng ta với một chiếc điện thoại thông minh
có thể sử dụng nhiều mô hình machine learning.

Hãy tưởng tượng chỉ viết một chương trình phản hồi với một *từ đánh thức*
chẳng hạn như "Alexa", "OK Google" và "Hey Siri".
Hãy thử lập trình chương trình đó trong một căn phòng một mình
với không gì ngoài một máy tính và trình soạn thảo mã,
như minh họa trong :numref:`fig_wake_word`.
Bạn sẽ viết một chương trình như vậy từ những nguyên lý đầu tiên như thế nào?
Hãy suy nghĩ về điều đó... bài toán này rất khó.
Mỗi giây, microphone sẽ thu thập khoảng
44.000 mẫu.
Mỗi mẫu là một phép đo biên độ của sóng âm thanh.
Quy tắc nào có thể ánh xạ một cách đáng tin cậy từ một đoạn âm thanh thô sang các dự đoán chắc chắn
$\{\textrm{yes}, \textrm{no}\}$
về việc đoạn âm thanh đó có chứa từ đánh thức hay không?
Nếu bạn đang bế tắc, đừng lo.
Chúng tôi cũng không biết cách viết một chương trình như vậy từ đầu.
Đó là lý do chúng ta sử dụng machine learning.

![Xác định một từ đánh thức.](../img/wake-word.svg)
:label:`fig_wake_word`


Đây là mấu chốt.
Thông thường, ngay cả khi chúng ta không biết cách chỉ dẫn rõ ràng cho máy tính
cách ánh xạ từ đầu vào sang đầu ra,
chúng ta vẫn có thể tự mình thực hiện kỳ tích nhận thức đó.
Nói cách khác, ngay cả khi bạn không biết
cách lập trình máy tính để nhận dạng từ "Alexa",
bản thân bạn vẫn có thể nhận ra từ đó.
Với khả năng này, chúng ta có thể thu thập một *dataset* khổng lồ
chứa các ví dụ về những đoạn âm thanh và các nhãn đi kèm,
cho biết đoạn âm thanh nào chứa từ đánh thức.
Trong cách tiếp cận machine learning đang chiếm ưu thế hiện nay,
chúng ta không cố gắng thiết kế một hệ thống
*explicitly* để nhận dạng các từ đánh thức.
Thay vào đó, chúng ta định nghĩa một chương trình linh hoạt
mà hành vi của nó được quyết định bởi một số *parameter*.
Sau đó, chúng ta sử dụng dataset để xác định các giá trị parameter tốt nhất có thể,
tức là những giá trị giúp cải thiện hiệu năng của chương trình
theo một thước đo hiệu năng đã chọn.

Bạn có thể hình dung các parameter như những núm điều chỉnh mà chúng ta có thể xoay,
qua đó thao tác lên hành vi của chương trình.
Khi các parameter đã được cố định, chúng ta gọi chương trình đó là một *model*.
Tập hợp tất cả các chương trình khác nhau (các ánh xạ đầu vào--đầu ra)
mà chúng ta có thể tạo ra chỉ bằng cách thao tác các parameter
được gọi là một *họ model*.
Còn "meta-program" sử dụng dataset của chúng ta
để lựa chọn các parameter được gọi là một *learning algorithm*.

Trước khi có thể tiến hành và sử dụng learning algorithm,
chúng ta phải xác định bài toán một cách chính xác,
làm rõ bản chất cụ thể của đầu vào và đầu ra,
đồng thời chọn một họ model phù hợp.
Trong trường hợp này,
model của chúng ta nhận một đoạn âm thanh làm *input*,
và model
tạo ra một lựa chọn trong số
$\{\textrm{yes}, \textrm{no}\}$ làm *output*.
Nếu mọi việc diễn ra đúng như kế hoạch,
các dự đoán của model
thường sẽ chính xác về việc
đoạn âm thanh có chứa từ đánh thức hay không.

Nếu chọn đúng họ model,
sẽ phải tồn tại một thiết lập các núm điều chỉnh
sao cho model phát tín hiệu "yes" mỗi khi nghe thấy từ "Alexa".
Vì lựa chọn chính xác của từ đánh thức là tùy ý,
có lẽ chúng ta sẽ cần một họ model đủ phong phú để,
thông qua một thiết lập khác của các núm điều chỉnh, nó có thể phát tín hiệu "yes"
chỉ khi nghe thấy từ "Apricot".
Chúng ta kỳ vọng cùng một họ model sẽ phù hợp
cho việc nhận dạng "Alexa" và nhận dạng "Apricot"
vì trực giác cho thấy đây là những tác vụ tương tự.
Tuy nhiên, chúng ta có thể cần một họ model hoàn toàn khác
nếu muốn xử lý các đầu vào hoặc đầu ra về bản chất khác nhau,
chẳng hạn nếu muốn ánh xạ từ hình ảnh sang chú thích,
hoặc từ câu tiếng Anh sang câu tiếng Trung.

Như bạn có thể đoán, nếu chỉ đặt tất cả các núm điều chỉnh một cách ngẫu nhiên,
model của chúng ta khó có khả năng nhận dạng "Alexa",
"Apricot" hoặc bất kỳ từ tiếng Anh nào khác.
Trong machine learning,
*learning* là quá trình
mà qua đó chúng ta tìm ra thiết lập đúng của các núm điều chỉnh
để buộc model tạo ra hành vi mong muốn.
Nói cách khác,
chúng ta *train* model bằng dữ liệu.
Như minh họa trong :numref:`fig_ml_loop`, quy trình training thường diễn ra như sau:

1. Bắt đầu với một model được khởi tạo ngẫu nhiên và không thể làm được việc gì hữu ích.
1. Lấy một phần dữ liệu của bạn (ví dụ: các đoạn âm thanh và các nhãn $\{\textrm{yes}, \textrm{no}\}$ tương ứng).
1. Điều chỉnh các núm điều chỉnh để model hoạt động tốt hơn, dựa trên đánh giá từ những ví dụ đó.
1. Lặp lại Bước 2 và 3 cho đến khi model trở nên tuyệt vời.

![Một quy trình training điển hình.](../img/ml-loop.svg)
:label:`fig_ml_loop`

Tóm lại, thay vì lập trình một bộ nhận dạng wake word,
chúng ta lập trình một chương trình có thể *học* cách nhận dạng wake word
khi được cung cấp một dataset lớn có gán nhãn.
Bạn có thể hình dung hành động xác định hành vi của một chương trình
bằng cách cung cấp cho nó một dataset là *lập trình bằng dữ liệu*.
Nói cách khác, chúng ta có thể "lập trình" một bộ phát hiện mèo
bằng cách cung cấp cho hệ thống machine learning
nhiều ví dụ về mèo và chó.
Bằng cách này, bộ phát hiện cuối cùng sẽ học cách phát ra
một số dương rất lớn nếu đó là mèo,
một số âm rất lớn nếu đó là chó,
và một giá trị gần bằng không hơn nếu nó không chắc chắn.
Đây mới chỉ là phần bề mặt của những gì machine learning có thể làm.
Deep learning, mà chúng ta sẽ giải thích chi tiết hơn ở phần sau,
chỉ là một trong nhiều phương pháp phổ biến
để giải quyết các bài toán machine learning.


## Các thành phần chính

Trong ví dụ về wake word, chúng ta đã mô tả một dataset
gồm các đoạn âm thanh và các nhãn nhị phân,
đồng thời trình bày một cách khái quát rằng chúng ta có thể huấn luyện
một model để xấp xỉ ánh xạ từ các đoạn âm thanh sang các lớp phân loại.
Loại bài toán này,
trong đó chúng ta cố gắng dự đoán một nhãn chưa biết được chỉ định
dựa trên các đầu vào đã biết
với một dataset gồm các ví dụ
mà nhãn của chúng đã được biết,
được gọi là *học có giám sát*.
Đây chỉ là một trong nhiều loại bài toán machine learning.
Trước khi tìm hiểu các loại khác,
chúng ta muốn làm rõ hơn
một số thành phần cốt lõi sẽ luôn đồng hành cùng chúng ta,
bất kể chúng ta giải quyết loại bài toán machine learning nào:

1. *Dữ liệu* mà chúng ta có thể học từ đó.
1. Một *model* về cách biến đổi dữ liệu.
1. Một *hàm mục tiêu* định lượng mức độ model đang hoạt động tốt (hoặc kém) như thế nào.
1. Một *thuật toán* để điều chỉnh các tham số của model nhằm tối ưu hóa hàm mục tiêu.

### Dữ liệu

Có lẽ không cần phải nói rằng bạn không thể làm data science nếu không có dữ liệu.
Chúng ta có thể mất hàng trăm trang để suy ngẫm chính xác dữ liệu *là gì*,
nhưng hiện tại, chúng ta sẽ tập trung vào các thuộc tính chính
của những dataset mà chúng ta quan tâm.
Nhìn chung, chúng ta làm việc với một tập hợp các ví dụ.
Để sử dụng dữ liệu một cách hiệu quả, thông thường chúng ta
cần tìm ra một cách biểu diễn số phù hợp.
Mỗi *ví dụ* (hay *điểm dữ liệu*, *mẫu dữ liệu*, *sample*)
thường gồm một tập hợp các thuộc tính
được gọi là *đặc trưng* (đôi khi được gọi là *covariate* hoặc *đầu vào*),
dựa trên đó model phải đưa ra các dự đoán.
Trong các bài toán học có giám sát,
mục tiêu của chúng ta là dự đoán giá trị của một thuộc tính đặc biệt,
được gọi là *nhãn* (hay *target*),
không nằm trong đầu vào của model.

Nếu làm việc với dữ liệu hình ảnh,
mỗi ví dụ có thể gồm một
bức ảnh riêng lẻ (các đặc trưng)
và một con số cho biết danh mục
mà bức ảnh đó thuộc về (nhãn).
Bức ảnh sẽ được biểu diễn bằng số
dưới dạng ba lưới giá trị số biểu diễn
độ sáng của ánh sáng đỏ, xanh lá cây và xanh dương
tại vị trí của từng pixel.
Ví dụ, một bức ảnh màu
có kích thước $200\times 200$ pixel
sẽ gồm $200\times200\times3=120000$ giá trị số.

Ngoài ra, chúng ta có thể làm việc với dữ liệu hồ sơ sức khỏe điện tử
và giải quyết bài toán dự đoán khả năng
một bệnh nhân cụ thể sẽ sống sót trong 30 ngày tiếp theo.
Ở đây, các đặc trưng của chúng ta có thể gồm một tập hợp
các thuộc tính dễ dàng có sẵn
và các phép đo được ghi nhận thường xuyên,
bao gồm tuổi, dấu hiệu sinh tồn, bệnh đồng mắc,
các loại thuốc hiện tại và các thủ thuật gần đây.
Nhãn có sẵn để huấn luyện sẽ là một giá trị nhị phân
cho biết mỗi bệnh nhân trong dữ liệu lịch sử
có sống sót trong khoảng thời gian 30 ngày hay không.

Trong những trường hợp như vậy, khi mỗi ví dụ được đặc trưng
bởi cùng một số lượng đặc trưng số,
chúng ta nói rằng các đầu vào là các vector có độ dài cố định
và gọi độ dài (không đổi) của các vector
là *số chiều* của dữ liệu.
Như bạn có thể hình dung, đầu vào có độ dài cố định có thể rất tiện lợi,
giúp chúng ta bớt phải lo thêm một vấn đề.
Tuy nhiên, không phải mọi dữ liệu đều có thể dễ dàng
được biểu diễn dưới dạng các vector *có độ dài cố định*.
Mặc dù chúng ta có thể mong đợi các ảnh chụp qua kính hiển vi
được tạo ra từ những thiết bị tiêu chuẩn,
chúng ta không thể kỳ vọng những hình ảnh lấy từ Internet
đều có cùng độ phân giải hoặc hình dạng.
Đối với hình ảnh, chúng ta có thể cân nhắc
cắt chúng về một kích thước tiêu chuẩn,
nhưng chiến lược đó chỉ giải quyết được một phần.
Chúng ta có nguy cơ làm mất thông tin ở những phần bị cắt.
Hơn nữa, dữ liệu văn bản còn chống lại các cách biểu diễn
có độ dài cố định một cách mạnh mẽ hơn.
Hãy xem xét những đánh giá của khách hàng
trên các trang thương mại điện tử như Amazon, IMDb và TripAdvisor.
Một số đánh giá rất ngắn: "nó thật tệ!".
Những đánh giá khác lại dài đến hàng trang.
Một ưu điểm lớn của deep learning so với các phương pháp truyền thống
là các model hiện đại có thể xử lý dữ liệu *có độ dài thay đổi*
một cách tương đối linh hoạt.

Nhìn chung, càng có nhiều dữ liệu thì công việc của chúng ta càng dễ dàng.
Khi có nhiều dữ liệu hơn, chúng ta có thể huấn luyện các model mạnh hơn
và ít phụ thuộc hơn vào những giả định định sẵn.
Sự chuyển đổi từ chế độ dữ liệu (tương đối) nhỏ sang dữ liệu lớn
là một yếu tố quan trọng góp phần vào thành công của deep learning hiện đại.
Để nhấn mạnh điều này, nhiều model
thú vị nhất trong deep learning
sẽ không hoạt động nếu không có các dataset lớn.
Một số model khác có thể hoạt động trong chế độ dữ liệu nhỏ,
nhưng không tốt hơn các phương pháp truyền thống.

Cuối cùng, chỉ có nhiều dữ liệu
và xử lý dữ liệu một cách thông minh là chưa đủ.
Chúng ta cần đúng loại dữ liệu.
Nếu dữ liệu chứa đầy sai sót,
hoặc nếu các đặc trưng được chọn không có khả năng dự đoán
đại lượng mục tiêu mà chúng ta quan tâm,
quá trình học sẽ thất bại.
Tình huống này được diễn đạt rất đúng qua câu nói sáo ngữ:
*rác vào, rác ra*.
Hơn nữa, hiệu năng dự đoán kém
không phải là hậu quả tiềm ẩn duy nhất.
Trong các ứng dụng machine learning nhạy cảm,
chẳng hạn như cảnh sát dự đoán, sàng lọc hồ sơ xin việc
và các model rủi ro được sử dụng trong hoạt động cho vay,
chúng ta phải đặc biệt cảnh giác
với hậu quả của dữ liệu rác.
Một dạng lỗi thường gặp liên quan đến các dataset
trong đó một số nhóm người không được đại diện
trong dữ liệu huấn luyện.
Hãy hình dung việc áp dụng một hệ thống nhận dạng ung thư da
chưa từng thấy làn da của người da đen.
Lỗi cũng có thể xảy ra khi dữ liệu
không chỉ đại diện thiếu cho một số nhóm
mà còn phản ánh những định kiến trong xã hội.
Ví dụ, nếu các quyết định tuyển dụng trong quá khứ
được sử dụng để huấn luyện một model dự đoán
nhằm sàng lọc hồ sơ xin việc,
thì các model machine learning có thể vô tình
nắm bắt và tự động hóa những bất công trong lịch sử.
Lưu ý rằng tất cả điều này có thể xảy ra mà không cần nhà khoa học dữ liệu
chủ động thông đồng, hoặc thậm chí nhận thức được điều đó.


### Model

Hầu hết machine learning đều liên quan đến việc biến đổi dữ liệu theo một cách nào đó.
Chúng ta có thể muốn xây dựng một hệ thống tiếp nhận ảnh và dự đoán mức độ vui vẻ.
Hoặc,
chúng ta có thể muốn tiếp nhận một tập hợp các chỉ số cảm biến
và dự đoán mức độ bình thường so với bất thường của các chỉ số đó.
Từ *model*, chúng ta chỉ một cơ chế tính toán tiếp nhận dữ liệu
thuộc một loại,
và đưa ra các dự đoán thuộc một loại có thể khác.
Cụ thể, chúng ta quan tâm đến các *model thống kê*
có thể được ước lượng từ dữ liệu.
Mặc dù các model đơn giản hoàn toàn có khả năng giải quyết
những bài toán đơn giản tương ứng,
các bài toán
mà chúng ta tập trung vào trong cuốn sách này vượt quá giới hạn của các phương pháp kinh điển.
Deep learning khác biệt với các phương pháp kinh điển
chủ yếu nhờ tập hợp các model mạnh mà nó tập trung vào.
Các model này gồm nhiều phép biến đổi dữ liệu liên tiếp
được kết nối với nhau từ trên xuống dưới, do đó có tên là *deep learning*.
Trên con đường tìm hiểu các model deep learning,
chúng ta cũng sẽ thảo luận về một số phương pháp truyền thống hơn.

### Hàm mục tiêu

Trước đó, chúng ta đã giới thiệu machine learning là việc học từ kinh nghiệm.
Ở đây, *học*
có nghĩa là cải thiện khả năng thực hiện một nhiệm vụ theo thời gian.
Nhưng ai có thể nói điều gì tạo nên một sự cải thiện?
Bạn có thể hình dung rằng chúng ta đề xuất cập nhật model,
và một số người có thể không đồng ý về việc đề xuất của chúng ta
có phải là một sự cải thiện hay không.

Để phát triển một hệ thống toán học hình thức cho các máy học,
chúng ta cần có những thước đo hình thức về mức độ tốt (hoặc kém) của các model.
Trong machine learning, và nói chung trong tối ưu hóa,
chúng được gọi là *hàm mục tiêu*.
Theo quy ước, chúng ta thường định nghĩa các hàm mục tiêu
sao cho giá trị thấp hơn là tốt hơn.
Đây chỉ là một quy ước.
Bạn có thể lấy bất kỳ hàm nào
mà giá trị cao hơn là tốt hơn, rồi biến nó thành một hàm mới
có tính chất tương đương về mặt định tính nhưng giá trị thấp hơn là tốt hơn
bằng cách đổi dấu.
Vì chúng ta chọn giá trị thấp hơn là tốt hơn, đôi khi các hàm này được gọi là
*các hàm mất mát*.

Khi cố gắng dự đoán các giá trị số,
hàm mất mát phổ biến nhất là *sai số bình phương*,
tức là bình phương của hiệu giữa
giá trị dự đoán và target thực tế.
Đối với bài toán phân loại, mục tiêu phổ biến nhất
là giảm tỷ lệ lỗi,
tức là tỷ lệ các ví dụ mà
dự đoán của chúng ta không khớp với giá trị thực tế.
Một số hàm mục tiêu (ví dụ, sai số bình phương) dễ tối ưu hóa,
trong khi những hàm khác (ví dụ, tỷ lệ lỗi) khó tối ưu hóa trực tiếp,
do không khả vi hoặc vì những vấn đề phức tạp khác.
Trong những trường hợp này, thay vào đó, người ta thường tối ưu hóa một *mục tiêu thay thế*.

Trong quá trình tối ưu hóa, chúng ta xem hàm loss
là một hàm của các tham số của mô hình
và coi tập dữ liệu huấn luyện là một hằng số.
Chúng ta tìm hiểu
các giá trị tốt nhất của những tham số trong mô hình
bằng cách tối thiểu hóa loss phát sinh trên một tập
gồm một số lượng ví dụ được thu thập để huấn luyện.
Tuy nhiên, đạt kết quả tốt trên dữ liệu huấn luyện
không đảm bảo rằng chúng ta sẽ làm tốt trên dữ liệu chưa từng thấy.
Vì vậy, thông thường chúng ta sẽ muốn chia dữ liệu hiện có thành hai phần:
*tập dữ liệu huấn luyện* (hoặc *tập huấn luyện*), dùng để học các tham số của mô hình;
và *tập dữ liệu kiểm tra* (hoặc *tập kiểm tra*), được giữ lại để đánh giá.
Sau cùng, thông thường chúng ta báo cáo
mô hình hoạt động như thế nào trên cả hai phần.
Bạn có thể xem hiệu năng huấn luyện
tương tự như điểm số mà một học sinh đạt được
trong các bài thi thử dùng để chuẩn bị cho một kỳ thi cuối kỳ thực sự.
Ngay cả khi kết quả đáng khích lệ,
điều đó cũng không đảm bảo thành công trong kỳ thi cuối kỳ.
Trong quá trình học, học sinh
có thể bắt đầu ghi nhớ các câu hỏi luyện tập,
trông như đã nắm vững chủ đề nhưng lại lúng túng
khi đối mặt với những câu hỏi chưa từng thấy
trong kỳ thi cuối kỳ thực tế.
Khi một mô hình hoạt động tốt trên tập huấn luyện
nhưng không thể tổng quát hóa sang dữ liệu chưa từng thấy,
chúng ta nói rằng mô hình đã *quá khớp (overfitting)*
với dữ liệu huấn luyện.


### Các thuật toán tối ưu hóa

Khi đã có một nguồn dữ liệu và cách biểu diễn dữ liệu,
một mô hình và một hàm mục tiêu được xác định rõ,
chúng ta cần một thuật toán có khả năng tìm kiếm
các tham số tốt nhất có thể để tối thiểu hóa hàm loss.
Các thuật toán tối ưu hóa phổ biến cho deep learning
dựa trên một phương pháp gọi là *gradient descent*.
Tóm lại, ở mỗi bước, phương pháp này
kiểm tra xem, với từng tham số,
loss trên tập huấn luyện sẽ thay đổi như thế nào
nếu ta làm nhiễu tham số đó chỉ một lượng nhỏ.
Sau đó, phương pháp sẽ cập nhật tham số
theo hướng làm giảm loss.


## Các dạng bài toán machine learning

Bài toán từ đánh thức trong ví dụ tạo động lực của chúng ta
chỉ là một trong rất nhiều bài toán
mà machine learning có thể giải quyết.
Để tiếp tục tạo động lực cho người đọc
và cung cấp một số thuật ngữ chung
sẽ được sử dụng xuyên suốt cuốn sách,
giờ đây chúng ta sẽ trình bày một cái nhìn tổng quan
về bức tranh toàn cảnh của các bài toán machine learning.

### Học có giám sát

Học có giám sát mô tả những nhiệm vụ
trong đó chúng ta được cung cấp một tập dữ liệu
chứa cả các đặc trưng và nhãn
và
được yêu cầu xây dựng một mô hình dự đoán các nhãn khi
được cung cấp các đặc trưng đầu vào.
Mỗi cặp đặc trưng--nhãn được gọi là một ví dụ.
Đôi khi, khi ngữ cảnh đã rõ ràng,
chúng ta có thể dùng thuật ngữ *các ví dụ*
để chỉ một tập hợp các đầu vào,
ngay cả khi chưa biết các nhãn tương ứng.
Yếu tố giám sát xuất hiện
vì để lựa chọn các tham số,
chúng ta (những người giám sát) cung cấp cho mô hình
một tập dữ liệu gồm các ví dụ đã được gán nhãn.
Xét theo phương diện xác suất, thông thường chúng ta quan tâm đến việc ước lượng
xác suất có điều kiện của một nhãn khi biết các đặc trưng đầu vào.
Dù chỉ là một trong số nhiều mô hình học,
học có giám sát chiếm phần lớn các
ứng dụng machine learning thành công trong ngành công nghiệp.
Một phần là vì nhiều nhiệm vụ quan trọng
có thể được mô tả rõ ràng là ước lượng xác suất
của một điều chưa biết khi biết một tập dữ liệu khả dụng cụ thể:

* Dự đoán có ung thư hay không dựa trên ảnh chụp cắt lớp vi tính.
* Dự đoán bản dịch đúng sang tiếng Pháp dựa trên một câu tiếng Anh.
* Dự đoán giá cổ phiếu vào tháng tới dựa trên dữ liệu báo cáo tài chính của tháng này.

Mặc dù mọi bài toán học có giám sát
đều được bao quát bởi mô tả đơn giản
“dự đoán các nhãn dựa trên các đặc trưng đầu vào”,
bản thân học có giám sát có thể có nhiều dạng khác nhau
và đòi hỏi rất nhiều quyết định về modeling,
tùy thuộc vào (trong số những yếu tố khác)
kiểu, kích thước và số lượng của các đầu vào và đầu ra.
Ví dụ, chúng ta sử dụng các mô hình khác nhau
để xử lý các chuỗi có độ dài tùy ý
và các biểu diễn vector có độ dài cố định.
Chúng ta sẽ tìm hiểu sâu
nhiều bài toán trong số này xuyên suốt cuốn sách.

Nói một cách không chính thức, quá trình học có thể diễn ra như sau.
Đầu tiên, lấy một tập hợp lớn các ví dụ trong đó các đặc trưng đã biết
và chọn ngẫu nhiên một tập con từ đó,
thu thập nhãn ground truth cho từng ví dụ.
Đôi khi những nhãn này có thể là dữ liệu sẵn có đã được thu thập
(ví dụ: bệnh nhân có tử vong trong năm tiếp theo hay không?)
và đôi khi chúng ta cần thuê người gán nhãn để gán nhãn cho dữ liệu,
(ví dụ: phân loại hình ảnh vào các danh mục).
Các đầu vào này cùng với những nhãn tương ứng tạo thành tập huấn luyện.
Chúng ta đưa tập dữ liệu huấn luyện vào một thuật toán học có giám sát,
một hàm nhận một tập dữ liệu làm đầu vào
và xuất ra một hàm khác: mô hình đã học.
Cuối cùng, chúng ta có thể đưa các đầu vào chưa từng thấy trước đây vào mô hình đã học,
sử dụng các đầu ra của mô hình làm dự đoán cho nhãn tương ứng.
Toàn bộ quy trình được minh họa trong :numref:`fig_supervised_learning`.

![Học có giám sát.](../img/supervised-learning.svg)
:label:`fig_supervised_learning`

#### Hồi quy

Có lẽ nhiệm vụ học có giám sát đơn giản nhất
mà bạn có thể dễ hình dung là *hồi quy*.
Ví dụ, hãy xem xét một tập dữ liệu được thu thập
từ cơ sở dữ liệu về các giao dịch bán nhà.
Chúng ta có thể lập một bảng,
trong đó mỗi hàng tương ứng với một căn nhà khác nhau,
và mỗi cột tương ứng với một thuộc tính liên quan,
chẳng hạn như diện tích của căn nhà,
số phòng ngủ, số phòng tắm,
và số phút đi bộ đến trung tâm thành phố.
Trong tập dữ liệu này, mỗi ví dụ sẽ là một căn nhà cụ thể,
và vector đặc trưng tương ứng sẽ là một hàng trong bảng.
Nếu bạn sống ở New York hoặc San Francisco,
và bạn không phải là CEO của Amazon, Google, Microsoft hay Facebook,
vector đặc trưng (diện tích, số phòng ngủ, số phòng tắm, khoảng cách đi bộ)
cho ngôi nhà của bạn có thể trông như sau: $[600, 1, 1, 60]$.
Tuy nhiên, nếu bạn sống ở Pittsburgh, nó có thể giống $[3000, 4, 3, 10]$ hơn.
Các vector đặc trưng có độ dài cố định như vậy là thiết yếu
đối với hầu hết các thuật toán machine learning kinh điển.

Điều thực sự khiến một bài toán trở thành hồi quy
là dạng của mục tiêu.
Giả sử bạn đang tìm mua một ngôi nhà mới.
Bạn có thể muốn ước tính giá trị thị trường hợp lý của một căn nhà,
dựa trên một số đặc trưng như trên.
Dữ liệu ở đây có thể gồm các tin đăng bán nhà trong quá khứ
và các nhãn có thể là giá bán quan sát được.
Khi các nhãn nhận các giá trị số tùy ý
(dù nằm trong một khoảng nào đó),
chúng ta gọi đây là bài toán *hồi quy*.
Mục tiêu là tạo ra một mô hình có các dự đoán
xấp xỉ sát với các giá trị nhãn thực tế.


Nhiều bài toán thực tế có thể dễ dàng được mô tả là bài toán hồi quy.
Dự đoán mức đánh giá mà người dùng sẽ gán cho một bộ phim
có thể được xem là một bài toán hồi quy
và nếu bạn thiết kế được một thuật toán tuyệt vời
để thực hiện việc này vào năm 2009,
bạn có thể đã giành được [giải Netflix trị giá 1 triệu đô la](https://en.wikipedia.org/wiki/Netflix_Prize).
Dự đoán thời gian bệnh nhân lưu lại trong bệnh viện
cũng là một bài toán hồi quy.
Một quy tắc kinh nghiệm hữu ích là bất kỳ bài toán *bao nhiêu?* hoặc *có bao nhiêu?*
nào cũng có khả năng là bài toán hồi quy. Ví dụ:

* Ca phẫu thuật này sẽ mất bao nhiêu giờ?
* Thị trấn này sẽ có bao nhiêu lượng mưa trong sáu giờ tới?


Ngay cả khi bạn chưa từng làm việc với machine learning,
có lẽ bạn đã từng giải một bài toán hồi quy theo cách không chính thức.
Ví dụ, hãy tưởng tượng bạn đã sửa xong hệ thống thoát nước
và người thợ đã dành 3 giờ
để loại bỏ cặn bẩn khỏi các đường ống nước thải của bạn.
Sau đó, họ gửi cho bạn hóa đơn 350 đô la.
Bây giờ hãy tưởng tượng bạn của bạn thuê cùng người thợ đó trong 2 giờ
và nhận được hóa đơn 250 đô la.
Nếu sau đó có người hỏi bạn nên dự kiến phải trả bao nhiêu
cho hóa đơn loại bỏ cặn bẩn sắp tới của họ,
bạn có thể đưa ra một số giả định hợp lý,
chẳng hạn như làm việc nhiều giờ hơn thì tốn nhiều tiền hơn.
Bạn cũng có thể giả định rằng có một khoản phí cơ bản
và người thợ sau đó tính phí theo giờ.
Nếu các giả định này đúng, thì dựa trên hai ví dụ dữ liệu này,
bạn đã có thể xác định cấu trúc giá của người thợ:
100 đô la mỗi giờ cộng với 50 đô la phí đến nhà.
Nếu bạn hiểu đến đây, thì bạn đã nắm được
ý tưởng cấp cao đằng sau hồi quy *tuyến tính*.

Trong trường hợp này, chúng ta có thể tạo ra các tham số
khớp chính xác với mức giá của người thợ.
Đôi khi điều này là không thể,
ví dụ, nếu một phần biến thiên
phát sinh từ các yếu tố nằm ngoài hai đặc trưng của bạn.
Trong những trường hợp này, chúng ta sẽ cố gắng học các mô hình
tối thiểu hóa khoảng cách giữa dự đoán của chúng ta và các giá trị quan sát được.
Trong hầu hết các chương, chúng ta sẽ tập trung vào
việc tối thiểu hóa hàm loss sai số bình phương.
Như chúng ta sẽ thấy sau này, loss này tương ứng với giả định
rằng dữ liệu của chúng ta bị nhiễu Gaussian.

#### Phân loại

Trong khi các mô hình hồi quy rất hữu ích
cho việc giải quyết những câu hỏi *có bao nhiêu?*,
nhiều bài toán không phù hợp với khuôn mẫu này.
Ví dụ, hãy xem xét một ngân hàng muốn
phát triển tính năng quét séc cho ứng dụng di động của mình.
Lý tưởng nhất là khách hàng chỉ cần chụp ảnh một tờ séc
và ứng dụng sẽ tự động nhận dạng văn bản từ hình ảnh.
Giả sử chúng ta có khả năng nào đó
để phân đoạn các mảnh hình ảnh
tương ứng với từng ký tự viết tay,
thì nhiệm vụ chính còn lại sẽ là
xác định ký tự nào trong một tập hợp đã biết
được mô tả trong mỗi mảnh hình ảnh.
Những bài toán dạng *là cái nào?* này được gọi là *phân loại*
và đòi hỏi một tập công cụ khác
so với những công cụ được sử dụng cho hồi quy,
mặc dù nhiều kỹ thuật vẫn có thể được áp dụng.

Trong *phân loại*, chúng ta muốn mô hình xem xét các đặc trưng,
chẳng hạn như giá trị pixel trong một hình ảnh,
rồi dự đoán một mẫu thuộc *danh mục* nào
(đôi khi được gọi là *lớp*)
trong một tập hợp rời rạc các lựa chọn.
Đối với các chữ số viết tay, chúng ta có thể có mười lớp,
tương ứng với các chữ số từ 0 đến 9.
Dạng phân loại đơn giản nhất là khi chỉ có hai lớp,
một bài toán được gọi là *phân loại nhị phân*.
Ví dụ, tập dữ liệu của chúng ta có thể gồm các hình ảnh động vật
và các nhãn của chúng ta có thể là các lớp $\textrm{\{cat, dog\}}$.
Trong khi với hồi quy, chúng ta tìm một bộ hồi quy để xuất ra một giá trị số,
thì với phân loại, chúng ta tìm một bộ phân loại,
có đầu ra là phép gán lớp được dự đoán.

Vì những lý do mà chúng ta sẽ tìm hiểu khi cuốn sách đi sâu hơn về mặt kỹ thuật,
việc tối ưu hóa một mô hình chỉ có thể xuất ra
một phép gán phân loại *dứt khoát*,
chẳng hạn như chỉ có thể là "mèo" hoặc "chó",
có thể khó khăn.
Trong những trường hợp này, việc biểu diễn mô hình
bằng ngôn ngữ xác suất thường dễ hơn nhiều.
Với các đặc trưng của một mẫu,
mô hình của chúng ta gán một xác suất
cho mỗi lớp có thể có.
Quay lại ví dụ phân loại động vật,
trong đó các lớp là $\textrm{\{cat, dog\}}$,
một bộ phân loại có thể nhìn thấy một hình ảnh và xuất ra xác suất
hình ảnh đó là một con mèo bằng 0.9.
Chúng ta có thể diễn giải con số này bằng cách nói rằng bộ phân loại
tin chắc 90\% hình ảnh mô tả một con mèo.
Độ lớn của xác suất đối với lớp được dự đoán
truyền tải một khái niệm về mức độ không chắc chắn.
Đó không phải là cách duy nhất hiện có
và chúng ta sẽ thảo luận về những cách khác trong các chương
trình bày những chủ đề nâng cao hơn.

Khi có nhiều hơn hai lớp có thể có,
chúng ta gọi bài toán này là *phân loại đa lớp*.
Các ví dụ phổ biến bao gồm nhận dạng ký tự viết tay
$\textrm{\{0, 1, 2, ... 9, a, b, c, ...\}}$.
Trong khi chúng ta giải quyết các bài toán hồi quy bằng cách cố gắng
tối thiểu hóa hàm mất mát sai số bình phương,
hàm mất mát phổ biến cho các bài toán phân loại được gọi là *cross-entropy*,
tên gọi này sẽ được giải thích rõ hơn
khi chúng ta giới thiệu lý thuyết thông tin trong các chương sau.

Lưu ý rằng lớp có khả năng xảy ra cao nhất không nhất thiết
là lớp mà bạn sẽ dùng để đưa ra quyết định.
Giả sử bạn tìm thấy một cây nấm tuyệt đẹp trong sân sau
như minh họa trong :numref:`fig_death_cap`.

![Mũ tử thần---không được ăn!](../img/death-cap.jpg)
:width:`200px`
:label:`fig_death_cap`

Bây giờ, giả sử bạn đã xây dựng một bộ phân loại và huấn luyện nó
để dự đoán liệu một cây nấm có độc hay không dựa trên một bức ảnh.
Giả sử bộ phân loại phát hiện nấm độc của chúng ta cho biết
xác suất mà
:numref:`fig_death_cap` cho thấy một cây mũ tử thần là 0.2.
Nói cách khác, bộ phân loại tin chắc 80\%
rằng cây nấm của chúng ta không phải là mũ tử thần.
Tuy vậy, bạn phải là một kẻ ngốc mới ăn nó.
Đó là vì lợi ích chắc chắn của một bữa tối ngon lành
không đáng để đánh đổi lấy 20\% nguy cơ chết vì nó.
Nói cách khác, tác động của rủi ro không chắc chắn
vượt xa lợi ích.
Do đó, để đưa ra quyết định có nên ăn cây nấm hay không,
chúng ta cần tính toán thiệt hại kỳ vọng
gắn với từng hành động,
phụ thuộc cả vào các kết quả có khả năng xảy ra
lẫn những lợi ích hoặc tác hại gắn với mỗi kết quả.
Trong trường hợp này, thiệt hại phát sinh
do ăn cây nấm
có thể là $0.2 \times \infty + 0.8 \times 0 = \infty$,
trong khi tổn thất do vứt bỏ nó
là $0.2 \times 0 + 0.8 \times 1 = 0.8$.
Sự thận trọng của chúng ta là có cơ sở:
như bất kỳ nhà nấm học nào cũng sẽ nói,
cây nấm trong :numref:`fig_death_cap`
thực sự là một cây mũ tử thần.

Phân loại có thể phức tạp hơn nhiều so với chỉ
phân loại nhị phân hoặc đa lớp.
Chẳng hạn, có một số biến thể phân loại
xử lý các lớp được cấu trúc theo thứ bậc.
Trong những trường hợp như vậy, không phải mọi lỗi đều như nhau---nếu
buộc phải mắc lỗi, chúng ta có thể muốn phân loại sai
sang một lớp có liên quan thay vì một lớp xa cách.
Thông thường, cách này được gọi là *phân loại phân cấp*.
Để lấy cảm hứng, bạn có thể nghĩ đến [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
người đã tổ chức hệ động vật theo một thứ bậc.

Trong trường hợp phân loại động vật,
nhầm một con poodle với một con schnauzer
có thể không quá nghiêm trọng,
nhưng mô hình của chúng ta sẽ chịu một hình phạt rất lớn
nếu nhầm một con poodle với một con khủng long.
Thứ bậc nào có liên quan có thể phụ thuộc
vào cách bạn dự định sử dụng mô hình.
Ví dụ, rắn đuôi chuông và rắn garter
có thể ở gần nhau trên cây phát sinh loài,
nhưng nhầm rắn đuôi chuông với rắn garter
có thể dẫn đến hậu quả chết người.

#### Gắn thẻ

Một số bài toán phân loại phù hợp hoàn toàn
với các thiết lập phân loại nhị phân hoặc đa lớp.
Ví dụ, chúng ta có thể huấn luyện một bộ phân loại nhị phân thông thường
để phân biệt mèo với chó.
Với tình trạng hiện tại của computer vision,
chúng ta có thể dễ dàng làm điều này bằng các công cụ có sẵn.
Tuy nhiên, dù mô hình của chúng ta có chính xác đến đâu,
chúng ta vẫn có thể gặp rắc rối khi bộ phân loại
gặp một hình ảnh về *Những nhạc công thành Bremen*,
một truyện cổ tích Đức nổi tiếng có bốn con vật
(:numref:`fig_stackedanimals`).

![Một con lừa, một con chó, một con mèo và một con gà trống.](../img/stackedanimals.png)
:width:`300px`
:label:`fig_stackedanimals`

Như bạn có thể thấy, bức ảnh có một con mèo,
một con gà trống, một con chó và một con lừa,
cùng một số cây ở phía sau.
Nếu dự đoán sẽ gặp những hình ảnh như vậy,
phân loại đa lớp có thể không phải là
cách xây dựng bài toán phù hợp.
Thay vào đó, chúng ta có thể muốn cho mô hình tùy chọn
nói rằng hình ảnh mô tả một con mèo, một con chó, một con lừa,
*và* một con gà trống.

Bài toán học cách dự đoán các lớp
không loại trừ lẫn nhau được gọi là *phân loại đa nhãn*.
Các bài toán tự động gắn thẻ thường được mô tả phù hợp nhất
dưới dạng phân loại đa nhãn.
Hãy nghĩ đến những thẻ mà mọi người có thể gắn
cho các bài đăng trên một blog kỹ thuật,
chẳng hạn như "machine learning", "technology", "gadgets",
"programming languages", "Linux", "cloud computing", "AWS".
Một bài viết điển hình có thể được gắn 5--10 thẻ.
Thông thường, các thẻ sẽ thể hiện một cấu trúc tương quan nhất định.
Các bài đăng về "cloud computing" có khả năng đề cập đến "AWS",
và các bài đăng về "machine learning" có khả năng đề cập đến "GPUs".

Đôi khi, các bài toán gắn thẻ như vậy
sử dụng những tập nhãn khổng lồ.
National Library of Medicine
sử dụng nhiều annotator chuyên nghiệp
để gán cho mỗi bài viết được lập chỉ mục trong PubMed
một tập thẻ được lấy từ ontology Medical Subject Headings (MeSH),
một tập hợp khoảng 28.000 thẻ.
Việc gắn thẻ chính xác cho các bài viết rất quan trọng
vì nó cho phép các nhà nghiên cứu tiến hành
những đánh giá toàn diện về tài liệu.
Đây là một quy trình tốn thời gian và thường có độ trễ một năm giữa việc lưu trữ và gắn thẻ.
Machine learning có thể cung cấp các thẻ tạm thời
cho đến khi mỗi bài viết được đánh giá thủ công đầy đủ.
Thực tế, trong nhiều năm, tổ chức BioASQ đã [tổ chức các cuộc thi](http://bioasq.org/)
cho nhiệm vụ này.

#### Tìm kiếm

Trong lĩnh vực truy hồi thông tin,
chúng ta thường áp dụng thứ hạng cho các tập hợp mục.
Lấy tìm kiếm trên web làm ví dụ.
Mục tiêu không phải là xác định *liệu*
một trang cụ thể có phù hợp với truy vấn hay không,
mà là trong số một tập hợp các kết quả phù hợp,
kết quả nào nên được hiển thị nổi bật nhất
cho một người dùng cụ thể.
Một cách thực hiện là
đầu tiên gán một điểm số
cho mọi phần tử trong tập hợp,
sau đó truy xuất các phần tử được xếp hạng cao nhất.
[PageRank](https://en.wikipedia.org/wiki/PageRank),
bí quyết ban đầu đằng sau công cụ tìm kiếm Google,
là một ví dụ sớm về hệ thống chấm điểm như vậy.
Điều kỳ lạ là điểm số do PageRank cung cấp
không phụ thuộc vào truy vấn thực tế.
Thay vào đó, họ dựa vào một bộ lọc mức độ phù hợp đơn giản
để xác định tập hợp các ứng viên phù hợp,
sau đó dùng PageRank để ưu tiên
những trang có thẩm quyền hơn.
Ngày nay, các công cụ tìm kiếm sử dụng machine learning và các mô hình hành vi
để thu được những điểm số mức độ phù hợp phụ thuộc vào truy vấn.
Có hẳn những hội nghị học thuật dành riêng cho chủ đề này.

#### Hệ thống đề xuất
:label:`subsec_recommender_systems`

Hệ thống đề xuất là một thiết lập bài toán khác
có liên quan đến tìm kiếm và xếp hạng.
Các bài toán tương tự nhau ở chỗ mục tiêu
là hiển thị một tập hợp các mục phù hợp với người dùng.
Điểm khác biệt chính là sự nhấn mạnh vào *cá nhân hóa*
cho từng người dùng cụ thể trong bối cảnh hệ thống đề xuất.
Ví dụ, đối với đề xuất phim,
trang kết quả dành cho một người hâm mộ khoa học viễn tưởng
và trang kết quả
dành cho một người sành các phim hài của Peter Sellers
có thể khác nhau đáng kể.
Các bài toán tương tự cũng xuất hiện trong những bối cảnh đề xuất khác,
chẳng hạn như đề xuất sản phẩm bán lẻ, âm nhạc và tin tức.

Trong một số trường hợp, khách hàng cung cấp phản hồi rõ ràng,
cho biết họ thích một sản phẩm cụ thể đến mức nào
(chẳng hạn như điểm đánh giá và nhận xét sản phẩm
trên Amazon, IMDb hoặc Goodreads).
Trong những trường hợp khác, họ cung cấp phản hồi ngầm,
chẳng hạn như bỏ qua các tiêu đề trong một playlist,
điều này có thể cho thấy
sự không hài lòng hoặc đơn giản là
cho thấy
bài hát đó không phù hợp với ngữ cảnh.
Trong những cách xây dựng đơn giản nhất,
các hệ thống này được huấn luyện
để ước tính một điểm số nào đó,
chẳng hạn như số sao dự kiến
hoặc xác suất một người dùng nhất định
sẽ mua một mục cụ thể.

Với một mô hình như vậy, đối với bất kỳ người dùng nào,
chúng ta có thể truy xuất tập hợp các đối tượng có điểm số cao nhất,
sau đó đề xuất chúng cho người dùng.
Các hệ thống production tiên tiến hơn đáng kể
và tính đến hoạt động chi tiết của người dùng cũng như đặc điểm của mục
khi tính toán những điểm số này.
:numref:`fig_deeplearning_amazon` hiển thị các sách deep learning
được Amazon đề xuất dựa trên các thuật toán cá nhân hóa
được tinh chỉnh để nắm bắt sở thích của Aston.

![Các sách deep learning được Amazon đề xuất.](../img/deeplearning-amazon.jpg)
:label:`fig_deeplearning_amazon`

Mặc dù có giá trị kinh tế to lớn,
các hệ thống gợi ý
được xây dựng một cách ngây thơ dựa trên các mô hình dự đoán
vẫn mắc phải một số khiếm khuyết nghiêm trọng về mặt khái niệm.
Trước hết, chúng ta chỉ quan sát được *phản hồi bị kiểm duyệt*:
người dùng có xu hướng đánh giá những bộ phim
mà họ có cảm nhận mạnh mẽ.
Ví dụ, trên thang điểm năm mức,
bạn có thể nhận thấy rằng các mục nhận được
nhiều đánh giá một sao và năm sao
nhưng lại có rất ít đánh giá ba sao một cách đáng chú ý.
Hơn nữa, thói quen mua hàng hiện tại thường là kết quả
của thuật toán gợi ý đang được sử dụng,
nhưng các thuật toán học không phải lúc nào cũng tính đến chi tiết này.
Do đó, các vòng lặp phản hồi có thể hình thành,
trong đó một hệ thống gợi ý ưu tiên đẩy một mục
và mục đó sau đó được xem là tốt hơn (do có nhiều lượt mua hơn),
từ đó lại được gợi ý thường xuyên hơn nữa.
Nhiều vấn đề trong số này---về
cách xử lý việc kiểm duyệt, các động lực khuyến khích và các vòng lặp phản hồi---
là những câu hỏi nghiên cứu quan trọng vẫn còn bỏ ngỏ.

#### Học theo chuỗi

Cho đến nay, chúng ta đã xem xét những bài toán trong đó
chúng ta có một số lượng đầu vào cố định và tạo ra một số lượng đầu ra cố định.
Ví dụ, chúng ta đã xét việc dự đoán giá nhà
dựa trên một tập đặc trưng cố định:
diện tích sàn, số phòng ngủ,
số phòng tắm và thời gian di chuyển đến trung tâm thành phố.
Chúng ta cũng đã thảo luận về việc ánh xạ từ một hình ảnh (có kích thước cố định)
sang các xác suất dự đoán rằng nó thuộc
vào từng lớp trong một số lượng lớp cố định
và dự đoán số sao gắn với các giao dịch mua
chỉ dựa trên ID người dùng và ID sản phẩm.
Trong những trường hợp này, sau khi mô hình được huấn luyện,
mỗi ví dụ kiểm thử được đưa vào mô hình,
nó sẽ ngay lập tức bị lãng quên.
Chúng ta giả định rằng các quan sát liên tiếp là độc lập
và do đó không cần lưu giữ ngữ cảnh này.

Nhưng chúng ta nên xử lý các đoạn video như thế nào?
Trong trường hợp này, mỗi đoạn có thể bao gồm một số lượng khung hình khác nhau.
Và phán đoán của chúng ta về những gì đang diễn ra trong mỗi khung hình có thể chính xác hơn nhiều
nếu tính đến các khung hình trước đó hoặc sau đó.
Điều tương tự cũng đúng với ngôn ngữ.
Ví dụ, một bài toán deep learning phổ biến là machine translation:
nhiệm vụ tiếp nhận các câu trong một ngôn ngữ nguồn
và dự đoán bản dịch của chúng sang một ngôn ngữ khác.

Những bài toán như vậy cũng xuất hiện trong y khoa.
Chúng ta có thể muốn một mô hình theo dõi bệnh nhân trong phòng chăm sóc đặc biệt
và phát cảnh báo bất cứ khi nào nguy cơ tử vong của họ trong 24 giờ tới
vượt quá một ngưỡng nhất định.
Ở đây, chúng ta sẽ không vứt bỏ mọi thứ
mà mình biết về bệnh sử của bệnh nhân sau mỗi giờ,
vì có thể chúng ta không muốn đưa ra dự đoán chỉ dựa trên
những phép đo gần đây nhất.

Những câu hỏi như vậy nằm trong số các ứng dụng
thú vị nhất của machine learning
và là những trường hợp của *học theo chuỗi*.
Chúng yêu cầu một mô hình hoặc tiếp nhận các chuỗi đầu vào
hoặc tạo ra các chuỗi đầu ra (hoặc cả hai).
Cụ thể, *học chuỗi-đến-chuỗi* xem xét các bài toán
trong đó cả đầu vào và đầu ra đều bao gồm các chuỗi có độ dài thay đổi.
Các ví dụ bao gồm machine translation
và phiên âm từ giọng nói thành văn bản.
Mặc dù không thể xem xét
tất cả các kiểu biến đổi chuỗi,
những trường hợp đặc biệt sau đây đáng được nhắc đến.

**Gắn thẻ và phân tích cú pháp**.
Việc này bao gồm chú thích một chuỗi văn bản bằng các thuộc tính.
Ở đây, đầu vào và đầu ra được *căn chỉnh*,
tức là chúng có cùng số lượng phần tử
và xuất hiện theo thứ tự tương ứng.
Chẳng hạn, trong *gắn thẻ từ loại (PoS)*,
chúng ta chú thích mỗi từ trong một câu
bằng từ loại tương ứng,
tức là "danh từ" hoặc "tân ngữ trực tiếp".
Ngoài ra, chúng ta có thể muốn biết
những nhóm từ liên tiếp nào đề cập đến các thực thể có tên,
chẳng hạn như *người*, *địa điểm* hoặc *tổ chức*.
Trong ví dụ đơn giản đến mức hoạt hình dưới đây,
chúng ta có thể chỉ muốn cho biết liệu có từ nào trong câu là một phần của thực thể có tên hay không (được gắn thẻ là "Ent").

```text
Tom has dinner in Washington with Sally
Ent  -    -    -     Ent      -    Ent
```


**Nhận dạng giọng nói tự động**.
Trong nhận dạng giọng nói, chuỗi đầu vào
là bản ghi âm của một người nói (:numref:`fig_speech`),
còn đầu ra là bản chép lại những gì người đó đã nói.
Thách thức nằm ở chỗ có nhiều khung âm thanh hơn rất nhiều
(âm thanh thường được lấy mẫu ở tần số 8kHz hoặc 16kHz)
so với văn bản, tức là không có sự tương ứng 1:1 giữa âm thanh và văn bản,
vì hàng nghìn mẫu có thể
tương ứng với một từ được nói.
Đây là những bài toán học chuỗi-đến-chuỗi,
trong đó đầu ra ngắn hơn nhiều so với đầu vào.
Mặc dù con người nhận dạng giọng nói rất giỏi,
ngay cả từ âm thanh chất lượng thấp,
việc khiến máy tính thực hiện được điều tương tự
là một thách thức vô cùng lớn.

![`-D-e-e-p- L-ea-r-ni-ng-` trong một bản ghi âm.](../img/speech.png)
:width:`700px`
:label:`fig_speech`

**Chuyển văn bản thành giọng nói**.
Đây là quá trình ngược lại của nhận dạng giọng nói tự động.
Ở đây, đầu vào là văn bản và đầu ra là một tệp âm thanh.
Trong trường hợp này, đầu ra dài hơn nhiều so với đầu vào.

**Machine Translation**.
Không giống trường hợp nhận dạng giọng nói,
trong đó đầu vào và đầu ra tương ứng
xuất hiện theo cùng một thứ tự,
trong machine translation,
dữ liệu không được căn chỉnh tạo ra một thách thức mới.
Ở đây, các chuỗi đầu vào và đầu ra
có thể có độ dài khác nhau,
và các vùng tương ứng
của hai chuỗi tương ứng
có thể xuất hiện theo thứ tự khác nhau.
Hãy xem ví dụ minh họa sau đây
về khuynh hướng đặc biệt của người Đức
là đặt động từ ở cuối câu:

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Have you already looked at this excellent textbook?
Wrong alignment:  Have you yourself already this excellent textbook looked at?
```


Nhiều bài toán liên quan cũng xuất hiện trong các nhiệm vụ học khác.
Chẳng hạn, việc xác định thứ tự mà người dùng
đọc một trang web là một bài toán phân tích bố cục hai chiều.
Các bài toán đối thoại thể hiện đủ loại phức tạp bổ sung,
trong đó việc xác định nên nói gì tiếp theo đòi hỏi phải tính đến
kiến thức về thế giới thực và trạng thái trước đó của cuộc hội thoại
qua những khoảng cách thời gian dài.
Những chủ đề như vậy là các lĩnh vực nghiên cứu đang được tích cực phát triển.

### Học không giám sát và tự giám sát

Các ví dụ trước tập trung vào học có giám sát,
trong đó chúng ta cung cấp cho mô hình một tập dữ liệu khổng lồ
chứa cả các đặc trưng và các giá trị nhãn tương ứng.
Bạn có thể hình dung người học có giám sát như một người
có công việc cực kỳ chuyên biệt và một ông sếp cực kỳ độc đoán.
Ông sếp đứng cạnh người học và bảo họ chính xác phải làm gì
trong mọi tình huống cho đến khi họ học được cách ánh xạ từ tình huống sang hành động.
Làm việc cho một ông sếp như vậy nghe thật tệ.
Mặt khác, làm hài lòng một ông sếp như vậy lại khá dễ.
Bạn chỉ cần nhận ra mẫu càng nhanh càng tốt
và bắt chước hành động của ông sếp.

Xét tình huống ngược lại,
làm việc cho một ông sếp
không biết mình muốn bạn làm gì
có thể khiến bạn thất vọng.
Tuy nhiên, nếu dự định trở thành một data scientist,
bạn nên làm quen với điều đó.
Ông sếp có thể chỉ đưa cho bạn một đống dữ liệu khổng lồ
và bảo bạn *hãy làm một chút data science với nó!*
Điều này nghe mơ hồ vì đúng là nó mơ hồ.
Chúng ta gọi nhóm bài toán này là *học không giám sát*,
và loại cũng như số lượng câu hỏi mà chúng ta có thể đặt ra
chỉ bị giới hạn bởi sức sáng tạo của mình.
Chúng ta sẽ đề cập đến các kỹ thuật học không giám sát
trong những chương sau.
Để khơi gợi sự hứng thú của bạn lúc này,
chúng ta mô tả một vài câu hỏi sau đây mà bạn có thể đặt ra.

* Chúng ta có thể tìm một số lượng nhỏ các nguyên mẫu
  để tóm tắt dữ liệu một cách chính xác không?
  Với một tập hợp ảnh, chúng ta có thể nhóm chúng thành ảnh phong cảnh,
  ảnh chó, trẻ sơ sinh, mèo và đỉnh núi không?
  Tương tự, với một tập hợp hoạt động duyệt web của người dùng,
  chúng ta có thể nhóm họ thành những người dùng có hành vi tương tự không?
  Bài toán này thường được gọi là *clustering*.
* Chúng ta có thể tìm một số lượng nhỏ các tham số
  để nắm bắt chính xác những thuộc tính liên quan của dữ liệu không?
  Quỹ đạo của một quả bóng được mô tả khá tốt
  bằng vận tốc, đường kính và khối lượng của quả bóng.
  Các thợ may đã phát triển một số lượng nhỏ các tham số
  mô tả khá chính xác hình dáng cơ thể người
  nhằm mục đích may quần áo vừa vặn.
  Những bài toán này được gọi là *ước lượng không gian con*.
  Nếu mối phụ thuộc là tuyến tính, nó được gọi là *phân tích thành phần chính*.
* Có một biểu diễn của các đối tượng (có cấu trúc tùy ý)
  trong không gian Euclid
  sao cho các thuộc tính ký hiệu có thể được đối sánh tốt không?
  Điều này có thể được sử dụng để mô tả các thực thể và mối quan hệ của chúng,
  chẳng hạn như "Rome" $-$ "Italy" $+$ "France" $=$ "Paris".
* Có một mô tả nào về các nguyên nhân gốc rễ
  của phần lớn dữ liệu mà chúng ta quan sát được không?
  Chẳng hạn, nếu có dữ liệu nhân khẩu học
  về giá nhà, ô nhiễm, tội phạm, vị trí,
  giáo dục và tiền lương, liệu chúng ta có thể khám phá
  mối quan hệ giữa chúng chỉ dựa trên dữ liệu thực nghiệm không?
  Các lĩnh vực liên quan đến *tính nhân quả* và
  *mô hình đồ thị xác suất* giải quyết những câu hỏi như vậy.
* Một bước phát triển quan trọng và thú vị khác gần đây trong học không giám sát
  là sự ra đời của *mô hình sinh sâu*.
  Các mô hình này ước lượng mật độ của dữ liệu,
  theo cách *tường minh* hoặc *ẩn*.
  Sau khi được huấn luyện, chúng ta có thể sử dụng một mô hình sinh
  để chấm điểm các ví dụ theo mức độ khả dĩ của chúng,
  hoặc lấy mẫu các ví dụ tổng hợp từ phân phối đã học.
  Những đột phá đầu tiên của deep learning trong mô hình hóa sinh
  đến cùng với phát minh về *bộ mã hóa tự động biến phân* :cite:`Kingma.Welling.2014,rezende2014stochastic`
  và tiếp tục với sự phát triển của *mạng đối nghịch sinh* :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
  Những tiến bộ gần đây hơn bao gồm normalizing flows :cite:`dinh2014nice,dinh2017density` và
  các mô hình khuếch tán :cite:`sohl2015deep,song2019generative,ho2020denoising,song2021score`.



Một bước phát triển tiếp theo trong học không giám sát
là sự trỗi dậy của *học tự giám sát*,
các kỹ thuật tận dụng một khía cạnh nào đó của dữ liệu chưa gán nhãn
để cung cấp sự giám sát.
Đối với văn bản, chúng ta có thể huấn luyện các mô hình
“điền vào chỗ trống”
bằng cách dự đoán các từ bị che ngẫu nhiên
dựa trên những từ xung quanh chúng (ngữ cảnh)
trong các kho ngữ liệu lớn mà không cần gán nhãn :cite:`Devlin.Chang.Lee.ea.2018`!
Đối với hình ảnh, chúng ta có thể huấn luyện các mô hình
để xác định vị trí tương đối
giữa hai vùng được cắt
của cùng một hình ảnh :cite:`Doersch.Gupta.Efros.2015`,
dự đoán phần bị che khuất của một hình ảnh
dựa trên các phần còn lại của hình ảnh,
hoặc dự đoán liệu hai mẫu
có phải là các phiên bản bị biến đổi của cùng một hình ảnh gốc hay không.
Các mô hình tự giám sát thường học được những biểu diễn
sau đó được tận dụng
bằng cách fine-tune các mô hình thu được
trên một tác vụ hạ nguồn cần quan tâm.


### Tương tác với một môi trường

Cho đến nay, chúng ta chưa thảo luận về nguồn gốc thực sự của dữ liệu,
hay điều gì thực sự xảy ra khi một mô hình machine learning tạo ra đầu ra.
Đó là vì học có giám sát và học không giám sát
không giải quyết những vấn đề này theo một cách thật sự tinh vi.
Trong mỗi trường hợp, chúng ta thu thập trước một khối lượng lớn dữ liệu,
sau đó khởi động các máy nhận dạng mẫu
mà không tương tác thêm với môi trường.
Vì toàn bộ quá trình học diễn ra
sau khi thuật toán bị ngắt kết nối khỏi môi trường,
điều này đôi khi được gọi là *học offline*.
Ví dụ, học có giám sát giả định một mẫu tương tác đơn giản
được minh họa trong :numref:`fig_data_collection`.

![Thu thập dữ liệu cho học có giám sát từ một môi trường.](../img/data-collection.svg)
:label:`fig_data_collection`

Sự đơn giản của học offline có những ưu điểm riêng.
Mặt tích cực là chúng ta có thể chỉ tập trung lo lắng
về việc nhận dạng mẫu,
mà không cần quan tâm đến những phức tạp phát sinh
từ các tương tác với một môi trường biến động.
Nhưng cách phát biểu bài toán này có những giới hạn.
Nếu bạn lớn lên cùng các tiểu thuyết Robot của Asimov,
thì có lẽ bạn hình dung những tác nhân trí tuệ nhân tạo
không chỉ có khả năng đưa ra dự đoán,
mà còn có thể thực hiện hành động trong thế giới.
Chúng ta muốn suy nghĩ về những *tác nhân* thông minh,
không chỉ về các mô hình dự đoán.
Điều này có nghĩa là chúng ta cần suy nghĩ về việc lựa chọn *hành động*,
không chỉ về việc đưa ra dự đoán.
Khác với những dự đoán đơn thuần,
các hành động thực sự tác động đến môi trường.
Nếu muốn huấn luyện một tác nhân thông minh,
chúng ta phải tính đến cách các hành động của tác nhân có thể
tác động đến những quan sát trong tương lai của nó, và vì vậy học offline không phù hợp.

Việc xem xét tương tác với một môi trường
mở ra một loạt câu hỏi mô hình hóa mới.
Sau đây chỉ là một vài ví dụ.

* Môi trường có ghi nhớ những gì chúng ta đã làm trước đó không?
* Môi trường có muốn giúp chúng ta không, chẳng hạn như người dùng đọc văn bản vào một bộ nhận dạng giọng nói?
* Môi trường có muốn đánh bại chúng ta không, chẳng hạn như những kẻ gửi spam điều chỉnh email để né bộ lọc spam?
* Môi trường có động lực thay đổi theo thời gian không? Ví dụ, dữ liệu trong tương lai có luôn giống với quá khứ không, hay các mẫu sẽ thay đổi theo thời gian, một cách tự nhiên hoặc để phản ứng với các công cụ tự động của chúng ta?

Những câu hỏi này nêu lên vấn đề *dịch chuyển phân phối*,
trong đó dữ liệu huấn luyện và dữ liệu kiểm tra khác nhau.
Một ví dụ mà nhiều người trong chúng ta có thể từng gặp là khi làm các bài thi do giảng viên biên soạn,
trong khi bài tập về nhà lại do các trợ giảng của họ soạn.
Tiếp theo, chúng ta sẽ mô tả ngắn gọn về học tăng cường,
một framework phong phú để đặt ra các bài toán học trong đó
một tác nhân tương tác với một môi trường.


### Học tăng cường

Nếu bạn quan tâm đến việc sử dụng machine learning
để phát triển một tác nhân tương tác với môi trường
và thực hiện các hành động, thì có lẽ bạn sẽ tập trung
vào *học tăng cường*.
Các ứng dụng có thể bao gồm robotics,
các hệ thống hội thoại,
và thậm chí cả việc phát triển trí tuệ nhân tạo (AI)
cho trò chơi điện tử.
*Học tăng cường sâu*, áp dụng
deep learning cho các bài toán học tăng cường,
đã trở nên phổ biến mạnh mẽ.
Deep Q-network mang tính đột phá, đã đánh bại con người
trong các trò chơi Atari chỉ bằng đầu vào hình ảnh :cite:`mnih2015human`,
và chương trình AlphaGo, đã lật đổ nhà vô địch thế giới
trong trò chơi cờ vây :cite:`Silver.Huang.Maddison.ea.2016`,
là hai ví dụ nổi bật.

Học tăng cường đưa ra một phát biểu rất tổng quát về một bài toán
trong đó một tác nhân tương tác với môi trường qua một chuỗi các bước thời gian.
Ở mỗi bước thời gian, tác nhân nhận được một *quan sát*
từ môi trường và phải chọn một *hành động*
sau đó được truyền trở lại môi trường
thông qua một cơ chế nào đó (đôi khi được gọi là *bộ chấp hành*), rồi sau mỗi vòng lặp,
tác nhân nhận được một phần thưởng từ môi trường.
Quá trình này được minh họa trong :numref:`fig_rl-environment`.
Sau đó tác nhân nhận một quan sát tiếp theo,
chọn một hành động tiếp theo, và cứ thế tiếp diễn.
Hành vi của một tác nhân học tăng cường được chi phối bởi một *policy*.
Nói ngắn gọn, một *policy* chỉ là một hàm ánh xạ
các quan sát về môi trường thành các hành động.
Mục tiêu của học tăng cường là tạo ra các policy tốt.

![Sự tương tác giữa học tăng cường và một môi trường.](../img/rl-environment.svg)
:label:`fig_rl-environment`

Không thể phóng đại mức độ tổng quát
của framework học tăng cường.
Ví dụ, học có giám sát
có thể được chuyển thành học tăng cường.
Giả sử chúng ta có một bài toán phân loại.
Chúng ta có thể tạo một tác nhân học tăng cường
với một hành động tương ứng với mỗi lớp.
Sau đó, chúng ta có thể tạo một môi trường cung cấp phần thưởng
chính xác bằng với hàm mất mát
của bài toán học có giám sát ban đầu.

Hơn nữa, học tăng cường
cũng có thể giải quyết nhiều bài toán
mà học có giám sát không thể.
Ví dụ, trong học có giám sát,
chúng ta luôn kỳ vọng đầu vào huấn luyện
được cung cấp cùng với nhãn chính xác.
Nhưng trong học tăng cường,
chúng ta không giả định rằng, với mỗi quan sát,
môi trường cho chúng ta biết hành động tối ưu.
Nhìn chung, chúng ta chỉ nhận được một phần thưởng.
Hơn nữa, môi trường thậm chí có thể không cho chúng ta biết
những hành động nào đã dẫn đến phần thưởng đó.

Hãy xem xét trò chơi cờ vua.
Tín hiệu phần thưởng thực sự duy nhất xuất hiện ở cuối ván đấu,
khi chúng ta thắng và nhận được phần thưởng, chẳng hạn, là $1$,
hoặc khi chúng ta thua và nhận được phần thưởng, chẳng hạn, là $-1$.
Vì vậy, các tác nhân học tăng cường phải xử lý
bài toán *gán công lao*:
xác định những hành động nào đáng được ghi công hoặc phải chịu trách nhiệm cho một kết quả.
Điều tương tự cũng xảy ra với một nhân viên
được thăng chức vào ngày 11 tháng 10.
Việc thăng chức đó có thể phản ánh một số
hành động được lựa chọn kỹ lưỡng trong năm trước.
Để được thăng chức trong tương lai, cần xác định
những hành động nào trên chặng đường đó đã dẫn đến các lần thăng chức trước đây.

Các tác nhân học tăng cường cũng có thể phải xử lý
bài toán quan sát một phần.
Nghĩa là, quan sát hiện tại có thể không
cho bạn biết mọi thứ về trạng thái hiện tại.
Giả sử robot dọn dẹp của bạn bị mắc kẹt
trong một trong nhiều tủ giống hệt nhau ở nhà.
Để giải cứu robot, cần suy luận
vị trí chính xác của nó, việc này có thể đòi hỏi xem xét các quan sát trước đó, trước khi nó đi vào tủ.

Cuối cùng, tại bất kỳ thời điểm nào, các tác nhân học tăng cường
có thể biết một policy tốt,
nhưng có thể còn nhiều policy khác tốt hơn
mà tác nhân chưa từng thử.
Tác nhân học tăng cường phải liên tục lựa chọn
giữa việc *khai thác* chiến lược tốt nhất (hiện đã biết) làm policy,
hoặc *khám phá* không gian các chiến lược,
có khả năng từ bỏ một phần phần thưởng ngắn hạn
để đổi lấy kiến thức.

Bài toán học tăng cường tổng quát
có một bối cảnh rất rộng.
Các hành động ảnh hưởng đến những quan sát tiếp theo.
Phần thưởng chỉ được quan sát khi chúng tương ứng với các hành động đã chọn.
Môi trường có thể được quan sát hoàn toàn hoặc một phần.
Việc tính đến toàn bộ sự phức tạp này cùng lúc có thể là đòi hỏi quá nhiều.
Hơn nữa, không phải mọi bài toán thực tế đều thể hiện toàn bộ sự phức tạp đó.
Do đó, các nhà nghiên cứu đã nghiên cứu một số
trường hợp đặc biệt của các bài toán học tăng cường.

Khi môi trường được quan sát hoàn toàn,
chúng ta gọi bài toán học tăng cường là một *quy trình quyết định Markov*.
Khi trạng thái không phụ thuộc vào các hành động trước đó,
chúng ta gọi đó là một *bài toán contextual bandit*.
Khi không có trạng thái, chỉ có một tập hợp các hành động khả dụng
với phần thưởng ban đầu chưa biết, chúng ta có *bài toán multi-armed bandit* kinh điển.

## Nguồn gốc

Chúng ta vừa điểm qua một tập hợp nhỏ các bài toán
mà machine learning có thể giải quyết.
Đối với một tập hợp đa dạng các bài toán machine learning,
deep learning cung cấp những công cụ mạnh mẽ để giải quyết chúng.
Mặc dù nhiều phương pháp deep learning là những phát minh gần đây,
những ý tưởng cốt lõi đằng sau việc học từ dữ liệu
đã được nghiên cứu trong nhiều thế kỷ.
Trên thực tế, con người đã mong muốn phân tích dữ liệu
và dự đoán các kết quả trong tương lai
từ rất lâu, và chính mong muốn này là nguồn gốc của phần lớn khoa học tự nhiên và toán học.
Hai ví dụ là phân phối Bernoulli, được đặt theo tên
[Jacob Bernoulli (1655--1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli),
và phân phối Gaussian được phát hiện
bởi [Carl Friedrich Gauss (1777--1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss).
Chẳng hạn, Gauss đã phát minh ra thuật toán bình phương tối thiểu,
vẫn được sử dụng ngày nay cho vô số bài toán,
từ tính toán bảo hiểm đến chẩn đoán y khoa.
Những công cụ như vậy đã thúc đẩy phương pháp thực nghiệm
trong khoa học tự nhiên---chẳng hạn, định luật Ohm
liên hệ cường độ dòng điện và điện áp trong một điện trở
được mô tả hoàn hảo bằng một mô hình tuyến tính.

Ngay cả vào thời Trung Cổ, các nhà toán học
đã có trực giác nhạy bén về việc ước lượng.
Chẳng hạn, cuốn sách hình học của [Jacob Köbel (1460--1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)
minh họa việc lấy trung bình độ dài bàn chân của 16 người đàn ông trưởng thành
để ước lượng độ dài bàn chân điển hình trong quần thể (:numref:`fig_koebel`).

![Ước lượng độ dài của một bàn chân.](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`


Khi một nhóm người rời khỏi nhà thờ,
16 người đàn ông trưởng thành được yêu cầu xếp thành một hàng
và đo bàn chân của họ.
Sau đó, tổng các số đo này được chia cho 16
để thu được ước lượng cho đơn vị mà ngày nay được gọi là một foot.
“Thuật toán” này sau đó được cải tiến
để xử lý những bàn chân dị dạng;
hai người có bàn chân ngắn nhất và dài nhất được yêu cầu rời đi,
chỉ lấy trung bình trên những người còn lại.
Đây là một trong những ví dụ sớm nhất
về ước lượng bằng trung bình cắt ngọn.

Statistics thực sự phát triển mạnh cùng với sự sẵn có và việc thu thập dữ liệu.
Một trong những người tiên phong của lĩnh vực này, [Ronald Fisher (1890--1962)](https://en.wikipedia.org/wiki/Ronald_Fisher),
đã có những đóng góp đáng kể cho lý thuyết của nó
cũng như các ứng dụng của nó trong di truyền học.
Nhiều thuật toán của ông (chẳng hạn như linear discriminant analysis)
và các khái niệm (chẳng hạn như Fisher information matrix)
vẫn giữ vị trí nổi bật
trong nền tảng của statistics hiện đại.
Ngay cả các tài nguyên dữ liệu của ông cũng có ảnh hưởng lâu dài.
Bộ dữ liệu Iris mà Fisher công bố vào năm 1936
vẫn đôi khi được dùng để minh họa
các thuật toán machine learning.
Fisher cũng là người ủng hộ thuyết ưu sinh,
điều này nhắc nhở chúng ta rằng việc sử dụng data science đáng ngờ về mặt đạo đức
có một lịch sử lâu dài và bền bỉ không kém việc sử dụng hữu ích của nó
trong công nghiệp và các ngành khoa học tự nhiên.


Các ảnh hưởng khác đối với machine learning
đến từ information theory của
[Claude Shannon (1916--2001)](https://en.wikipedia.org/wiki/Claude_Shannon)
và theory of computation do
[Alan Turing (1912--1954)](https://en.wikipedia.org/wiki/Alan_Turing) đề xuất.
Turing đặt ra câu hỏi “máy móc có thể suy nghĩ không?”
trong bài báo nổi tiếng *Computing Machinery and Intelligence* :cite:`Turing.1950`.
Mô tả điều ngày nay được gọi là bài kiểm tra Turing, ông đề xuất rằng một cỗ máy
có thể được xem là *thông minh* nếu một người đánh giá khó
phân biệt được các câu trả lời
của máy với câu trả lời của con người, chỉ dựa trên tương tác bằng văn bản.

Những ảnh hưởng tiếp theo đến từ neuroscience và psychology.
Xét cho cùng, con người rõ ràng thể hiện hành vi thông minh.
Nhiều học giả đã đặt câu hỏi liệu có thể giải thích
và, có lẽ, reverse engineer khả năng này hay không.
Một trong những thuật toán lấy cảm hứng từ sinh học đầu tiên
được xây dựng bởi [Donald Hebb (1904--1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).
Trong cuốn sách mang tính đột phá *The Organization of Behavior* :cite:`Hebb.1949`,
ông đưa ra giả thuyết rằng các neuron học thông qua reinforcement tích cực.
Điều này được biết đến với tên gọi quy tắc học Hebbian.
Những ý tưởng này đã truyền cảm hứng cho các công trình sau đó, chẳng hạn như
thuật toán học perceptron của Rosenblatt,
và đặt nền móng cho nhiều thuật toán stochastic gradient descent
làm nền tảng cho deep learning ngày nay:
củng cố hành vi mong muốn và làm giảm hành vi không mong muốn
để có được các thiết lập tốt cho những tham số trong một neural network.

Cảm hứng sinh học là nguồn gốc của tên gọi *neural networks*.
Trong hơn một thế kỷ (bắt đầu từ các mô hình của Alexander Bain, 1873,
và James Sherrington, 1890), các nhà nghiên cứu đã cố gắng lắp ráp
những mạch tính toán giống với các mạng neuron tương tác với nhau.
Theo thời gian, cách diễn giải sinh học đã trở nên ít sát nghĩa hơn,
nhưng tên gọi vẫn được giữ lại. Về cốt lõi, có một vài nguyên tắc then chốt
có thể tìm thấy trong hầu hết các mạng ngày nay:

* Sự xen kẽ của các đơn vị xử lý tuyến tính và phi tuyến, thường được gọi là *layers*.
* Việc sử dụng chain rule (còn được gọi là *backpropagation*) để điều chỉnh các tham số trong toàn bộ mạng cùng một lúc.

Sau những tiến bộ nhanh chóng ban đầu, nghiên cứu về neural networks
chững lại trong khoảng từ năm 1995 đến năm 2005.
Điều này chủ yếu là do hai nguyên nhân.
Thứ nhất, việc train một mạng rất tốn kém về mặt tính toán.
Mặc dù bộ nhớ truy cập ngẫu nhiên rất dồi dào vào cuối thế kỷ trước,
năng lực tính toán lại khan hiếm.
Thứ hai, các bộ dữ liệu tương đối nhỏ.
Trên thực tế, bộ dữ liệu Iris của Fisher từ năm 1936
vẫn là một công cụ phổ biến để kiểm thử hiệu quả của các thuật toán.
Bộ dữ liệu MNIST với 60.000 chữ số viết tay được xem là khổng lồ.

Do dữ liệu và năng lực tính toán khan hiếm,
các công cụ thống kê mạnh như kernel methods,
decision trees và graphical models
đã chứng minh được tính ưu việt về thực nghiệm trong nhiều ứng dụng.
Hơn nữa, không giống neural networks,
chúng không cần nhiều tuần để train
và cho kết quả có thể dự đoán được
cùng những bảo đảm lý thuyết vững chắc.


## Con đường dẫn đến Deep Learning

Phần lớn điều này đã thay đổi khi có sẵn
một lượng dữ liệu khổng lồ,
nhờ World Wide Web,
sự xuất hiện của các công ty phục vụ
hàng trăm triệu người dùng trực tuyến,
sự phổ biến của các cảm biến chất lượng cao với chi phí thấp,
việc lưu trữ dữ liệu giá rẻ (định luật Kryder),
và năng lực tính toán rẻ (định luật Moore).
Đặc biệt, bối cảnh tính toán trong deep learning
đã được cách mạng hóa nhờ những tiến bộ trong GPU, vốn ban đầu được chế tạo cho game máy tính.
Đột nhiên, các thuật toán và mô hình
từng có vẻ không khả thi về mặt tính toán
đã trở nên trong tầm với.
Điều này được minh họa rõ nhất trong :numref:`tab_intro_decade`.

:Tập dữ liệu so với bộ nhớ máy tính và năng lực tính toán
:label:`tab_intro_decade`

|Thập kỷ|Tập dữ liệu|Bộ nhớ|Số phép tính dấu phẩy động mỗi giây|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (giá nhà ở Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (nhận dạng ký tự quang học)|10 MB|10 MF (Intel 80486)|
|2000|10 M (trang web)|100 MB|1 GF (Intel Core)|
|2010|10 G (quảng cáo)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (mạng xã hội)|100 GB|1 PF (NVIDIA DGX-2)|


Lưu ý rằng bộ nhớ truy cập ngẫu nhiên đã không theo kịp tốc độ tăng trưởng của dữ liệu.
Đồng thời, sự gia tăng năng lực tính toán
đã vượt xa tốc độ tăng trưởng của các bộ dữ liệu.
Điều này có nghĩa là các mô hình thống kê
cần trở nên hiệu quả hơn về bộ nhớ,
nhờ đó chúng có thể dành nhiều chu kỳ máy tính hơn
để tối ưu hóa các tham số, nhờ
ngân sách tính toán được tăng lên.
Do đó, điểm tối ưu trong machine learning và statistics
đã chuyển từ các mô hình tuyến tính (tổng quát hóa) và kernel methods
sang deep neural networks.
Đây cũng là một trong những lý do khiến nhiều nền tảng chính
của deep learning, chẳng hạn như multilayer perceptrons
:cite:`McCulloch.Pitts.1943`, convolutional neural networks
:cite:`LeCun.Bottou.Bengio.ea.1998`, long short-term memory
:cite:`Hochreiter.Schmidhuber.1997`,
và Q-Learning :cite:`Watkins.Dayan.1992`,
về cơ bản đã được “khám phá lại” trong thập kỷ qua,
sau khi tương đối không hoạt động trong một thời gian đáng kể.

Những tiến bộ gần đây trong các mô hình, ứng dụng và thuật toán thống kê
đôi khi được ví với vụ nổ Cambri:
một thời điểm các loài tiến hóa nhanh chóng.
Quả thực, hiện trạng tiên tiến không chỉ đơn thuần là hệ quả
của các tài nguyên sẵn có được áp dụng cho những thuật toán có tuổi đời hàng thập kỷ.
Lưu ý rằng danh sách các ý tưởng dưới đây mới chỉ chạm đến bề mặt
của những gì đã giúp các nhà nghiên cứu đạt được những tiến bộ to lớn
trong thập kỷ qua.


* Các phương pháp mới để kiểm soát năng lực mô hình, chẳng hạn như *dropout*
  :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`,
  đã giúp giảm thiểu overfitting.
  Ở đây, nhiễu được đưa vào :cite:`Bishop.1995`
  khắp mạng neural trong quá trình huấn luyện.
* *Cơ chế attention* đã giải quyết một vấn đề thứ hai
  từng gây khó khăn cho ngành thống kê trong hơn một thế kỷ:
  làm thế nào để tăng bộ nhớ và độ phức tạp của một hệ thống
  mà không tăng số lượng tham số có thể học.
  Các nhà nghiên cứu đã tìm ra một lời giải thanh nhã
  bằng cách sử dụng thứ chỉ có thể được xem là
  một *cấu trúc con trỏ có thể học* :cite:`Bahdanau.Cho.Bengio.2014`.
  Thay vì phải ghi nhớ toàn bộ chuỗi văn bản, chẳng hạn
  trong dịch máy bằng một biểu diễn có số chiều cố định,
  tất cả những gì cần lưu trữ chỉ là một con trỏ tới trạng thái trung gian
  của quá trình dịch. Điều này cho phép tăng đáng kể độ chính xác
  đối với các chuỗi dài, vì mô hình
  không còn cần ghi nhớ toàn bộ chuỗi trước khi
  bắt đầu tạo ra một chuỗi mới.
* Được xây dựng hoàn toàn dựa trên các cơ chế attention,
  kiến trúc *Transformer* :cite:`Vaswani.Shazeer.Parmar.ea.2017` đã cho thấy khả năng *scaling* vượt trội: mô hình hoạt động tốt hơn khi tăng kích thước tập dữ liệu, kích thước mô hình và lượng compute dùng cho huấn luyện :cite:`kaplan2020scaling`. Kiến trúc này đã đạt được thành công nổi bật trong nhiều lĩnh vực,
  chẳng hạn như xử lý ngôn ngữ tự nhiên :cite:`Devlin.Chang.Lee.ea.2018,brown2020language`, thị giác máy tính :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`, nhận dạng giọng nói :cite:`gulati2020conformer`, học tăng cường :cite:`chen2021decision`, và mạng neural đồ thị :cite:`dwivedi2020generalization`. Ví dụ, một Transformer duy nhất được pretrain trên các modality
  đa dạng như văn bản, hình ảnh, mô-men xoắn khớp nối và thao tác nhấn nút
  có thể chơi Atari, chú thích hình ảnh, trò chuyện,
  và điều khiển robot :cite:`reed2022generalist`.
* Bằng cách mô hình hóa xác suất của các chuỗi văn bản, *language model* có thể dự đoán văn bản dựa trên văn bản khác. Việc mở rộng dữ liệu, mô hình và compute đã mở ra ngày càng nhiều khả năng để language model thực hiện các tác vụ mong muốn thông qua việc tạo văn bản giống con người dựa trên văn bản đầu vào :cite:`brown2020language,rae2021scaling,hoffmann2022training,chowdhery2022palm,openai2023gpt4,anil2023palm,touvron2023llama,touvron2023llama2`. Chẳng hạn, bằng cách căn chỉnh language model với ý định của con người :cite:`ouyang2022training`, OpenAI's [ChatGPT](https://chat.openai.com/) cho phép người dùng tương tác với nó theo cách hội thoại để giải quyết các vấn đề như debug code và viết sáng tạo.
* Các thiết kế nhiều giai đoạn, chẳng hạn thông qua các memory network
  :cite:`Sukhbaatar.Weston.Fergus.ea.2015`
  và neural programmer-interpreter :cite:`Reed.De-Freitas.2015`
  đã cho phép các nhà mô hình hóa thống kê mô tả những phương pháp tiếp cận suy luận lặp.
  Những công cụ này cho phép một trạng thái bên trong của deep neural network
  được sửa đổi nhiều lần,
  qua đó thực hiện các bước tiếp theo
  trong một chuỗi suy luận, giống như một processor
  có thể sửa đổi bộ nhớ để thực hiện một phép tính.
* Một bước phát triển then chốt trong *mô hình sinh sâu* là phát minh ra
  *generative adversarial network*
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
  Theo truyền thống, các phương pháp thống kê để ước lượng mật độ
  và các mô hình sinh tập trung vào việc tìm các phân phối xác suất phù hợp
  và các thuật toán (thường là xấp xỉ) để lấy mẫu từ chúng.
  Vì vậy, các thuật toán này phần lớn bị giới hạn bởi sự thiếu linh hoạt
  vốn có trong các mô hình thống kê.
  Đổi mới quan trọng trong generative adversarial network là thay sampler
  bằng một thuật toán tùy ý có các tham số khả vi.
  Sau đó, các tham số này được điều chỉnh sao cho discriminator
  (về bản chất là một phép kiểm định hai mẫu) không thể phân biệt dữ liệu giả với dữ liệu thật.
  Nhờ khả năng sử dụng các thuật toán tùy ý để sinh dữ liệu,
  việc ước lượng mật độ đã được mở rộng để áp dụng cho nhiều kỹ thuật khác nhau.
  Các ví dụ về những con ngựa vằn phi nước đại :cite:`Zhu.Park.Isola.ea.2017`
  và những khuôn mặt người nổi tiếng giả :cite:`Karras.Aila.Laine.ea.2017`
  đều là minh chứng cho tiến bộ này.
  Ngay cả những người chỉ vẽ nguệch ngoạc nghiệp dư cũng có thể tạo ra
  hình ảnh chân thực như ảnh chụp chỉ dựa trên các bản phác thảo mô tả bố cục của một cảnh :cite:`Park.Liu.Wang.ea.2019`. 
* Hơn nữa, trong khi quá trình diffusion dần dần thêm nhiễu ngẫu nhiên vào các mẫu dữ liệu, *diffusion model* :cite:`sohl2015deep,ho2020denoising` học quá trình khử nhiễu để dần dần xây dựng các mẫu dữ liệu từ nhiễu ngẫu nhiên, đảo ngược quá trình diffusion. Chúng đã bắt đầu thay thế generative adversarial network trong các mô hình sinh sâu gần đây hơn, chẳng hạn như DALL-E 2 :cite:`ramesh2022hierarchical` và Imagen :cite:`saharia2022photorealistic` để sáng tác nghệ thuật và tạo hình ảnh dựa trên mô tả bằng văn bản.
* Trong nhiều trường hợp, một GPU duy nhất không đủ để xử lý lượng dữ liệu lớn có sẵn cho việc huấn luyện.
  Trong thập kỷ qua, khả năng xây dựng các thuật toán huấn luyện song song và
  phân tán đã được cải thiện đáng kể.
  Một trong những thách thức then chốt khi thiết kế các thuật toán có khả năng mở rộng
  là công cụ chủ lực của việc tối ưu hóa deep learning,
  stochastic gradient descent, dựa vào việc xử lý các minibatch dữ liệu
  tương đối nhỏ.
  Đồng thời, các batch nhỏ làm hạn chế hiệu suất của GPU.
  Do đó, huấn luyện trên 1,024 GPU với kích thước minibatch là,
  chẳng hạn, 32 hình ảnh mỗi batch tương đương với một minibatch tổng hợp
  gồm khoảng 32.000 hình ảnh. Công trình đầu tiên của :citet:`Li.2017`
  và sau đó của :citet:`You.Gitman.Ginsburg.2017`
  và :citet:`Jia.Song.He.ea.2018` đã tăng kích thước lên 64.000 quan sát,
  giảm thời gian huấn luyện mô hình ResNet-50
  trên tập dữ liệu ImageNet xuống còn chưa đầy 7 phút.
  Để so sánh, thời gian huấn luyện ban đầu thường kéo dài hàng ngày.
* Khả năng song song hóa tính toán
  cũng góp phần thúc đẩy tiến bộ trong *học tăng cường*.
  Điều này đã dẫn đến những tiến bộ đáng kể trong việc giúp máy tính đạt
  hiệu suất vượt con người ở các tác vụ như Go, trò chơi Atari,
  Starcraft và mô phỏng vật lý (ví dụ: sử dụng MuJoCo),
  nơi có các simulator cho môi trường.
  Xem, chẳng hạn, :citet:`Silver.Huang.Maddison.ea.2016` để biết mô tả
  về những thành tựu như vậy trong AlphaGo. Tóm lại,
  học tăng cường hoạt động tốt nhất
  khi có sẵn rất nhiều bộ ba (state, action, reward).
  Mô phỏng cung cấp một phương thức như vậy.
* Các framework deep learning đã đóng vai trò then chốt
  trong việc phổ biến các ý tưởng.
  Thế hệ đầu tiên của các framework mã nguồn mở
  để mô hình hóa mạng neural bao gồm
  [Caffe](https://github.com/BVLC/caffe),
  [Torch](https://github.com/torch), và
  [Theano](https://github.com/Theano/Theano).
  Nhiều bài báo có ảnh hưởng nền tảng đã được viết bằng các công cụ này.
  Hiện nay, chúng đã được thay thế bởi
  [TensorFlow](https://github.com/tensorflow/tensorflow) (thường được sử dụng thông qua API cấp cao [Keras](https://github.com/keras-team/keras)), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), và [Apache MXNet](https://github.com/apache/incubator-mxnet).
  Thế hệ thứ ba của các framework bao gồm
  những công cụ được gọi là *imperative* dành cho deep learning,
  một xu hướng có thể nói là được khơi mào bởi [Chainer](https://github.com/chainer/chainer),
  sử dụng cú pháp tương tự Python NumPy để mô tả các mô hình.
  Ý tưởng này đã được cả [PyTorch](https://github.com/pytorch/pytorch),
  [Gluon API](https://github.com/apache/incubator-mxnet) của MXNet,
  và [JAX](https://github.com/google/jax) áp dụng.


Sự phân công lao động giữa các nhà nghiên cứu hệ thống xây dựng những công cụ tốt hơn
và các nhà mô hình hóa thống kê xây dựng những mạng neural tốt hơn
đã đơn giản hóa mọi thứ đáng kể. Chẳng hạn,
việc huấn luyện một mô hình hồi quy logistic tuyến tính
từng là một bài tập về nhà không hề đơn giản,
xứng đáng được giao cho các nghiên cứu sinh mới ngành machine learning
tại Carnegie Mellon University vào năm 2014.
Đến nay, nhiệm vụ này có thể được hoàn thành
với chưa đến 10 dòng code,
khiến nó hoàn toàn nằm trong khả năng của bất kỳ lập trình viên nào.


## Những câu chuyện thành công

Trí tuệ nhân tạo có một lịch sử lâu dài trong việc tạo ra những kết quả
mà nếu không có nó sẽ rất khó đạt được.
Chẳng hạn, các hệ thống phân loại thư
sử dụng nhận dạng ký tự quang học
đã được triển khai từ những năm 1990.
Xét cho cùng, đây chính là nguồn gốc
của tập dữ liệu MNIST nổi tiếng
về các chữ số viết tay.
Điều tương tự cũng áp dụng cho việc đọc séc gửi tiền ngân hàng và chấm điểm
mức độ tín nhiệm của người đăng ký.
Các giao dịch tài chính được tự động kiểm tra gian lận.
Đây là nền tảng của nhiều hệ thống thanh toán thương mại điện tử,
chẳng hạn như PayPal, Stripe, AliPay, WeChat, Apple, Visa và MasterCard.
Các chương trình máy tính chơi cờ đã có tính cạnh tranh trong nhiều thập kỷ.
Machine learning hỗ trợ hoạt động tìm kiếm, đề xuất, cá nhân hóa
và xếp hạng trên Internet.
Nói cách khác, machine learning hiện diện ở khắp nơi, dù thường bị che khuất khỏi tầm nhìn.

Chỉ mới gần đây AI
mới trở thành tâm điểm chú ý, phần lớn nhờ
các giải pháp cho những vấn đề
trước đây được xem là không thể giải quyết
và liên quan trực tiếp đến người tiêu dùng.
Nhiều tiến bộ trong số đó được quy cho deep learning.

* Các trợ lý thông minh, chẳng hạn như Siri của Apple,
  Alexa của Amazon và trợ lý của Google,
  có thể phản hồi các yêu cầu bằng giọng nói
  với độ chính xác khá cao.
  Điều này bao gồm những tác vụ đơn giản như bật công tắc đèn,
  cũng như các nhiệm vụ phức tạp hơn, chẳng hạn như sắp xếp lịch hẹn với thợ cắt tóc
  và cung cấp hội thoại hỗ trợ qua điện thoại.
  Đây có lẽ là dấu hiệu dễ nhận thấy nhất
  cho thấy AI đang tác động đến cuộc sống của chúng ta.
* Một thành phần then chốt của các trợ lý kỹ thuật số
  là khả năng nhận dạng giọng nói chính xác.
  Độ chính xác của những hệ thống như vậy đã dần dần
  tăng lên đến mức
  ngang bằng với con người
  trong một số ứng dụng nhất định :cite:`Xiong.Wu.Alleva.ea.2018`.
* Nhận dạng đối tượng cũng đã tiến bộ vượt bậc.
  Việc xác định đối tượng trong một bức ảnh
  là một nhiệm vụ khá thách thức vào năm 2010.
  Trên benchmark ImageNet, các nhà nghiên cứu từ NEC Labs
  và University of Illinois at Urbana-Champaign
  đã đạt tỷ lệ lỗi top-5 là 28% :cite:`Lin.Lv.Zhu.ea.2010`.
  Đến năm 2017, tỷ lệ lỗi này đã giảm xuống còn 2,25% :cite:`Hu.Shen.Sun.2018`.
  Tương tự, những kết quả đáng kinh ngạc cũng đã đạt được
  trong việc nhận dạng tiếng chim hót và chẩn đoán ung thư da.
* Năng lực chơi game từng được dùng
  làm thước đo khả năng của con người.
  Bắt đầu với TD-Gammon, một chương trình chơi backgammon
  sử dụng học tăng cường sai phân thời gian,
  những tiến bộ về thuật toán và năng lực tính toán
  đã dẫn đến các thuật toán cho nhiều ứng dụng khác nhau.
  So với backgammon, cờ vua có
  không gian trạng thái và tập hợp hành động phức tạp hơn nhiều.
  DeepBlue đã đánh bại Garry Kasparov bằng cách sử dụng tính song song quy mô lớn,
  phần cứng chuyên dụng và phương pháp tìm kiếm hiệu quả
  trong cây trò chơi :cite:`Campbell.Hoane-Jr.Hsu.2002`.
  Go còn khó hơn nữa do có không gian trạng thái khổng lồ.
  AlphaGo đạt mức ngang bằng với con người vào năm 2015,
  bằng cách kết hợp deep learning với lấy mẫu cây Monte Carlo :cite:`Silver.Huang.Maddison.ea.2016`.
  Thách thức trong Poker là không gian trạng thái rất lớn
  và chỉ được quan sát một phần
  (chúng ta không biết các lá bài của đối thủ).
  Libratus đã vượt qua năng lực của con người trong Poker
  bằng cách sử dụng các chiến lược được cấu trúc hiệu quả :cite:`Brown.Sandholm.2017`.
* Một dấu hiệu khác cho thấy AI đang tiến bộ
  là sự xuất hiện của các phương tiện tự lái.
  Mặc dù khả năng tự chủ hoàn toàn vẫn chưa đạt được,
  lĩnh vực này đã có những tiến bộ xuất sắc,
  với các công ty như Tesla, NVIDIA
  và Waymo đưa ra thị trường những sản phẩm
  cho phép tự chủ một phần.
  Điều khiến khả năng tự chủ hoàn toàn trở nên khó khăn
  là việc lái xe đúng cách đòi hỏi
  khả năng nhận thức, suy luận
  và đưa các quy tắc vào một hệ thống.
  Hiện tại, deep learning chủ yếu được sử dụng
  cho khía cạnh thị giác của những bài toán này.
  Phần còn lại được các kỹ sư tinh chỉnh rất nhiều.



Những điều này mới chỉ chạm đến bề mặt
của các ứng dụng quan trọng của machine learning.
Chẳng hạn, robot học, logistics, sinh học tính toán,
vật lý hạt và thiên văn học
đều có được một phần những tiến bộ ấn tượng nhất gần đây
nhờ machine learning; vì thế, machine learning đang trở thành
một công cụ phổ biến cho các kỹ sư và nhà khoa học.

Các bài viết không mang tính kỹ thuật thường xuyên đặt ra
những câu hỏi về ngày tận thế AI sắp xảy ra
và tính khả thi của một *singularity*.
Nỗi lo là bằng cách nào đó, các hệ thống machine learning
sẽ trở nên có tri giác và đưa ra quyết định,
độc lập với những người lập trình chúng,
những quyết định trực tiếp tác động đến cuộc sống của con người.
Ở một mức độ nào đó, AI đã tác động
đến sinh kế của con người theo những cách trực tiếp:
khả năng tín dụng được đánh giá tự động,
autopilot phần lớn điều hướng phương tiện, các quyết định
về việc có cho phép bảo lãnh hay không sử dụng dữ liệu thống kê làm đầu vào.
Ở khía cạnh ít nghiêm trọng hơn, chúng ta có thể yêu cầu Alexa bật máy pha cà phê.

May mắn thay, chúng ta vẫn còn cách rất xa một hệ thống AI có tri giác
có thể cố ý thao túng những người tạo ra nó.
Thứ nhất, các hệ thống AI được thiết kế,
huấn luyện và triển khai
theo một cách cụ thể, hướng đến mục tiêu.
Mặc dù hành vi của chúng có thể tạo ảo giác
về trí thông minh tổng quát, nền tảng của thiết kế này là sự kết hợp giữa các quy tắc, heuristic
và các mô hình thống kê.
Thứ hai, hiện tại đơn giản là chưa có công cụ nào cho *artificial general intelligence*
có khả năng tự cải thiện,
suy luận về chính mình, đồng thời có thể sửa đổi,
mở rộng và cải thiện kiến trúc của chính chúng
trong khi cố gắng giải quyết các nhiệm vụ tổng quát.

Mối lo cấp bách hơn nhiều là cách AI đang được sử dụng trong đời sống hằng ngày.
Nhiều nhiệm vụ thường nhật hiện do con người đảm nhiệm có thể và sẽ được tự động hóa.
Các robot nông nghiệp có khả năng sẽ làm giảm chi phí cho những người nông dân hữu cơ,
nhưng đồng thời cũng sẽ tự động hóa các hoạt động thu hoạch.
Giai đoạn này của cuộc cách mạng công nghiệp
có thể gây ra những hệ quả sâu sắc đối với những bộ phận lớn trong xã hội,
vì các công việc đơn giản tạo ra nhiều việc làm
ở nhiều quốc gia.
Hơn nữa, các mô hình thống kê, khi được áp dụng thiếu thận trọng,
có thể dẫn đến thiên lệch về chủng tộc, giới tính hoặc độ tuổi, đồng thời làm dấy lên
những lo ngại chính đáng về tính công bằng trong quy trình
nếu được tự động hóa để đưa ra các quyết định có hậu quả đáng kể.
Điều quan trọng là phải đảm bảo các thuật toán này được sử dụng cẩn trọng.
Với những gì chúng ta biết ngày nay, đây dường như là mối lo cấp bách hơn nhiều
so với nguy cơ siêu trí tuệ độc ác hủy diệt nhân loại.


## Bản chất của Deep Learning

Cho đến nay, chúng ta đã nói một cách khái quát về machine learning.
Deep learning là một phân nhánh của machine learning
tập trung vào các mô hình dựa trên mạng neural nhiều lớp.
Nó là *deep* chính xác theo nghĩa các mô hình của nó
học được nhiều *lớp* biến đổi.
Mặc dù điều này có vẻ hẹp,
deep learning đã tạo ra
một loạt mô hình, kỹ thuật,
cách xây dựng bài toán và ứng dụng vô cùng đa dạng.
Nhiều trực giác đã được phát triển
để giải thích lợi ích của độ sâu.
Có thể nói, mọi machine learning
đều có nhiều lớp tính toán,
trong đó lớp đầu tiên bao gồm các bước xử lý đặc trưng.
Điểm khác biệt của deep learning là
các phép toán được học ở mỗi lớp trong số nhiều lớp
biểu diễn được học đồng thời từ dữ liệu.

Những bài toán mà chúng ta đã thảo luận từ trước đến nay,
chẳng hạn như học từ tín hiệu âm thanh thô,
các giá trị pixel thô của hình ảnh,
hoặc ánh xạ giữa những câu có độ dài tùy ý và
các câu tương ứng bằng ngôn ngữ nước ngoài,
là những bài toán mà deep learning phát huy hiệu quả
còn các phương pháp truyền thống gặp khó khăn.
Hóa ra những mô hình nhiều lớp này
có khả năng xử lý dữ liệu cảm nhận cấp thấp
theo cách mà các công cụ trước đây không thể thực hiện.
Có thể nói, điểm chung quan trọng nhất
trong các phương pháp deep learning là *end-to-end training*.
Tức là, thay vì lắp ráp một hệ thống
dựa trên các thành phần được tinh chỉnh riêng lẻ,
người ta xây dựng hệ thống rồi tinh chỉnh hiệu năng của chúng đồng thời.
Chẳng hạn, trong computer vision, các nhà khoa học
từng tách riêng quy trình *feature engineering*
khỏi quy trình xây dựng các mô hình machine learning.
Bộ phát hiện biên Canny :cite:`Canny.1987`
và bộ trích xuất đặc trưng SIFT của Lowe :cite:`Lowe.2004`
đã thống trị hơn một thập kỷ với tư cách là các thuật toán
ánh xạ hình ảnh thành các vector đặc trưng.
Ngày trước, phần cốt lõi của việc áp dụng machine learning vào những bài toán này
là nghĩ ra những cách được thiết kế thủ công
để biến đổi dữ liệu thành một dạng phù hợp với các mô hình nông.
Đáng tiếc là khả năng con người có thể đạt được nhờ sự khéo léo
vẫn có giới hạn, nếu so với việc đánh giá nhất quán
hàng triệu lựa chọn do một thuật toán tự động thực hiện.
Khi deep learning lên ngôi,
các bộ trích xuất đặc trưng này được thay thế
bằng những bộ lọc được tự động tinh chỉnh, cho độ chính xác cao hơn.

Vì vậy, một ưu điểm then chốt của deep learning là nó thay thế
không chỉ các mô hình nông ở cuối các pipeline học tập truyền thống,
mà còn cả quy trình feature engineering tốn nhiều công sức.
Hơn nữa, bằng cách thay thế phần lớn quy trình tiền xử lý đặc thù theo lĩnh vực,
deep learning đã xóa bỏ nhiều ranh giới
từng ngăn cách computer vision, nhận dạng giọng nói,
xử lý ngôn ngữ tự nhiên, tin học y tế và các lĩnh vực ứng dụng khác,
qua đó cung cấp một bộ công cụ thống nhất để giải quyết những bài toán đa dạng.

Bên cạnh end-to-end training, chúng ta đang chứng kiến sự chuyển đổi
từ các mô tả thống kê tham số sang các mô hình hoàn toàn phi tham số.
Khi dữ liệu khan hiếm, cần dựa vào những giả định đơn giản hóa về thực tại
để có được các mô hình hữu ích.
Khi dữ liệu dồi dào, những giả định này có thể được thay thế
bằng các mô hình phi tham số phù hợp với dữ liệu hơn.
Ở một mức độ nào đó, điều này phản ánh những tiến bộ
mà vật lý đã trải qua vào giữa thế kỷ trước
khi máy tính trở nên phổ biến.
Thay vì giải bằng tay các xấp xỉ tham số về cách electron hoạt động,
giờ đây người ta có thể sử dụng các mô phỏng số
của những phương trình vi phân riêng phần liên quan.
Điều này đã dẫn đến các mô hình chính xác hơn nhiều,
dù thường phải đánh đổi bằng khả năng diễn giải.

Một điểm khác biệt so với các công trình trước đây là sự chấp nhận những nghiệm chưa tối ưu,
việc xử lý các bài toán tối ưu phi tuyến không lồi
và sự sẵn sàng thử nghiệm trước khi chứng minh.
Chủ nghĩa kinh nghiệm mới mẻ này trong việc xử lý các bài toán thống kê,
kết hợp với dòng nhân tài đổ vào nhanh chóng, đã dẫn đến
những tiến bộ nhanh chóng trong việc phát triển các thuật toán thực tiễn,
dù trong nhiều trường hợp phải đánh đổi bằng việc sửa đổi
và phát minh lại những công cụ đã tồn tại hàng thập kỷ.

Sau cùng, cộng đồng deep learning tự hào
về việc chia sẻ các công cụ vượt qua ranh giới học thuật và doanh nghiệp,
phát hành nhiều thư viện, mô hình thống kê
và mạng đã huấn luyện xuất sắc dưới dạng mã nguồn mở.
Chính theo tinh thần đó, các notebook tạo nên cuốn sách này
được cung cấp tự do để phân phối và sử dụng.
Chúng tôi đã nỗ lực để giảm các rào cản tiếp cận
cho bất kỳ ai muốn tìm hiểu về deep learning
và hy vọng độc giả sẽ được hưởng lợi từ điều này.


## Tóm tắt

Machine learning nghiên cứu cách các hệ thống máy tính có thể tận dụng kinh nghiệm (thường là dữ liệu) để cải thiện hiệu suất trong các tác vụ cụ thể. Nó kết hợp các ý tưởng từ thống kê, khai phá dữ liệu và tối ưu hóa. Thông thường, machine learning được sử dụng như một phương tiện để triển khai các giải pháp AI. Là một nhánh của machine learning, học biểu diễn tập trung vào cách tự động tìm ra phương thức phù hợp để biểu diễn dữ liệu. Được xem là học biểu diễn đa cấp thông qua việc học nhiều lớp biến đổi, deep learning không chỉ thay thế các mô hình nông ở cuối các pipeline machine learning truyền thống, mà còn thay thế quy trình tốn nhiều công sức là feature engineering. Phần lớn những tiến bộ gần đây trong deep learning được thúc đẩy bởi sự dồi dào của dữ liệu bắt nguồn từ các cảm biến giá rẻ và những ứng dụng có quy mô Internet, cũng như bởi những tiến bộ đáng kể trong tính toán, chủ yếu thông qua GPU. Hơn nữa, sự sẵn có của các framework deep learning hiệu quả đã giúp việc thiết kế và triển khai tối ưu hóa toàn bộ hệ thống trở nên dễ dàng hơn đáng kể; đây là một thành phần then chốt để đạt được hiệu suất cao.

## Bài tập

1. Những phần nào trong code mà bạn đang viết hiện nay có thể được “học”, tức là được cải thiện bằng cách học và tự động xác định các lựa chọn thiết kế được đưa ra trong code của bạn? Code của bạn có bao gồm các lựa chọn thiết kế dựa trên heuristic không? Bạn có thể cần những dữ liệu nào để học hành vi mong muốn?
1. Những vấn đề nào mà bạn gặp phải có nhiều ví dụ về cách giải quyết nhưng chưa có phương pháp cụ thể để tự động hóa? Đây có thể là những ứng viên hàng đầu cho việc sử dụng deep learning.
1. Hãy mô tả mối quan hệ giữa các thuật toán, dữ liệu và năng lực tính toán. Đặc điểm của dữ liệu và các tài nguyên tính toán hiện có ảnh hưởng như thế nào đến tính phù hợp của các thuật toán khác nhau?
1. Hãy nêu một số bối cảnh mà việc huấn luyện end-to-end hiện chưa phải là phương pháp mặc định nhưng có thể hữu ích.

[Thảo luận](https://discuss.d2l.ai/t/22)