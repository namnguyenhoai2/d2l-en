# Khả năng khái quát hóa trong Phân loại

:label:`chap_classification_generalization`



Cho đến nay, chúng ta đã tập trung vào cách giải quyết các bài toán phân loại đa lớp
bằng cách huấn luyện các mạng nơ-ron (tuyến tính) với nhiều đầu ra và các hàm softmax.
Diễn giải đầu ra của mô hình dưới dạng các dự đoán xác suất,
chúng ta đã giải thích động cơ và suy ra hàm mất mát cross-entropy,
hàm này tính negative log likelihood
mà mô hình của chúng ta (với một tập tham số cố định)
gán cho các nhãn thực tế.
Cuối cùng, chúng ta đã áp dụng các công cụ này vào thực tế
bằng cách khớp mô hình với tập huấn luyện.
Tuy nhiên, như thường lệ, mục tiêu của chúng ta là học các *mẫu tổng quát*,
được đánh giá thực nghiệm trên dữ liệu chưa từng thấy trước đó (tập kiểm tra).
Độ chính xác cao trên tập huấn luyện không có ý nghĩa gì.
Bất cứ khi nào mỗi đầu vào của chúng ta là duy nhất
(và thực tế điều này đúng với hầu hết các tập dữ liệu nhiều chiều),
chúng ta có thể đạt được độ chính xác hoàn hảo trên tập huấn luyện
chỉ bằng cách ghi nhớ tập dữ liệu trong epoch huấn luyện đầu tiên,
sau đó tra cứu nhãn bất cứ khi nào thấy một ảnh mới.
Tuy nhiên, việc ghi nhớ chính xác các nhãn
gắn với chính xác các ví dụ huấn luyện
không cho chúng ta biết cách phân loại các ví dụ mới.
Nếu không có thêm hướng dẫn, chúng ta có thể phải quay lại
đoán ngẫu nhiên bất cứ khi nào gặp các ví dụ mới.

Một số câu hỏi cấp thiết cần được trả lời ngay:

1. Chúng ta cần bao nhiêu ví dụ kiểm tra để ước lượng tốt độ chính xác của các classifier trên quần thể cơ bản?
1. Điều gì xảy ra nếu chúng ta liên tục đánh giá các mô hình trên cùng một tập kiểm tra?
1. Tại sao chúng ta nên kỳ vọng rằng việc khớp các mô hình tuyến tính với tập huấn luyện
   sẽ hoạt động tốt hơn phương án ghi nhớ ngây thơ của chúng ta?


Trong khi :numref:`sec_generalization_basics` đã giới thiệu
các kiến thức cơ bản về overfitting và khả năng khái quát hóa
trong bối cảnh hồi quy tuyến tính,
chương này sẽ đi sâu hơn một chút,
giới thiệu một số ý tưởng nền tảng
của lý thuyết học thống kê.
Hóa ra chúng ta thường có thể đảm bảo khả năng khái quát hóa *a priori*:
với nhiều mô hình,
và với bất kỳ cận trên mong muốn nào
cho khoảng cách khái quát hóa $\epsilon$,
chúng ta thường có thể xác định một số lượng mẫu cần thiết $n$
sao cho nếu tập huấn luyện chứa ít nhất $n$
mẫu, lỗi thực nghiệm của chúng ta sẽ nằm
trong phạm vi $\epsilon$ so với lỗi thực,
*đối với bất kỳ phân phối sinh dữ liệu nào*.
Đáng tiếc, cũng hóa ra rằng
mặc dù các dạng đảm bảo này cung cấp
một tập hợp nền tảng tri thức sâu sắc,
chúng lại có tiện ích thực tế hạn chế
đối với người thực hành deep learning.
Tóm lại, những đảm bảo này gợi ý
rằng để đảm bảo khả năng khái quát hóa
của các mạng nơ-ron sâu *a priori*
cần một số lượng ví dụ phi lý
(có lẽ hàng nghìn tỷ hoặc hơn),
ngay cả khi chúng ta nhận thấy rằng, trên các tác vụ mà mình quan tâm,
các mạng nơ-ron sâu thường khái quát hóa
rất tốt với ít ví dụ hơn nhiều (hàng nghìn).
Do đó, người thực hành deep learning thường từ bỏ hoàn toàn
các đảm bảo *a priori*,
thay vào đó sử dụng các phương pháp
đã khái quát hóa tốt
trên các bài toán tương tự trong quá khứ,
và chứng nhận khả năng khái quát hóa *post hoc*
thông qua các đánh giá thực nghiệm.
Khi đến :numref:`chap_perceptrons`,
chúng ta sẽ xem lại khả năng khái quát hóa
và giới thiệu ngắn gọn
về kho tài liệu khoa học đồ sộ
đã xuất hiện nhằm giải thích lý do
các mạng nơ-ron sâu có thể khái quát hóa trong thực tế.

## Tập kiểm tra

Vì chúng ta đã bắt đầu dựa vào các tập kiểm tra
như phương pháp tiêu chuẩn vàng
để đánh giá lỗi khái quát hóa,
hãy bắt đầu bằng việc thảo luận
các thuộc tính của những ước lượng lỗi như vậy.
Hãy tập trung vào một classifier cố định $f$,
mà không cần bận tâm nó được tạo ra như thế nào.
Ngoài ra, giả sử rằng chúng ta có
một tập dữ liệu *mới* gồm các ví dụ $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$
chưa được dùng để huấn luyện classifier $f$.
*Lỗi thực nghiệm* của classifier $f$ trên $\mathcal{D}$
đơn giản là tỷ lệ các trường hợp
mà dự đoán $f(\mathbf{x}^{(i)})$
không khớp với nhãn thực $y^{(i)}$,
và được cho bởi biểu thức sau:

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

Ngược lại, *lỗi quần thể*
là tỷ lệ *kỳ vọng*
của các ví dụ trong quần thể cơ bản
(một phân phối $P(X,Y)$ được đặc trưng
bởi hàm mật độ xác suất $p(\mathbf{x},y)$)
mà classifier của chúng ta không khớp
với nhãn thực:

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Mặc dù $\epsilon(f)$ là đại lượng mà chúng ta thực sự quan tâm,
chúng ta không thể quan sát trực tiếp nó,
cũng như không thể trực tiếp
quan sát chiều cao trung bình trong một quần thể lớn
mà không đo từng người một.
Chúng ta chỉ có thể ước lượng đại lượng này dựa trên các mẫu.
Vì tập kiểm tra $\mathcal{D}$
đại diện về mặt thống kê
cho quần thể cơ bản,
chúng ta có thể xem $\epsilon_\mathcal{D}(f)$ là một ước lượng thống kê
của lỗi quần thể $\epsilon(f)$.
Hơn nữa, vì đại lượng quan tâm $\epsilon(f)$
là một kỳ vọng (của biến ngẫu nhiên $\mathbf{1}(f(X) \neq Y)$)
và ước lượng tương ứng $\epsilon_\mathcal{D}(f)$
là trung bình mẫu,
việc ước lượng lỗi quần thể
đơn giản là bài toán ước lượng trung bình cổ điển,
mà bạn có thể nhớ từ :numref:`sec_prob`.

Một kết quả cổ điển quan trọng trong lý thuyết xác suất
được gọi là *định lý giới hạn trung tâm* đảm bảo
rằng bất cứ khi nào chúng ta có $n$ mẫu ngẫu nhiên $a_1, ..., a_n$
được rút ra từ bất kỳ phân phối nào có trung bình $\mu$ và độ lệch chuẩn $\sigma$,
thì khi số lượng mẫu $n$ tiến đến vô hạn,
trung bình mẫu $\hat{\mu}$ xấp xỉ
tiến tới một phân phối chuẩn có tâm
tại trung bình thực và có độ lệch chuẩn $\sigma/\sqrt{n}$.
Điều này đã cho chúng ta biết một điều quan trọng:
khi số lượng ví dụ trở nên lớn,
lỗi kiểm tra $\epsilon_\mathcal{D}(f)$
sẽ tiến gần lỗi thực $\epsilon(f)$
với tốc độ $\mathcal{O}(1/\sqrt{n})$.
Vì vậy, để ước lượng lỗi kiểm tra chính xác gấp đôi,
chúng ta phải thu thập tập kiểm tra lớn gấp bốn lần.
Để giảm lỗi kiểm tra đi một hệ số một trăm,
chúng ta phải thu thập tập kiểm tra lớn gấp mười nghìn lần.
Nói chung, tốc độ $\mathcal{O}(1/\sqrt{n})$
như vậy thường là điều tốt nhất mà chúng ta có thể kỳ vọng trong thống kê.

Giờ đây, khi đã biết điều gì đó về tốc độ tiệm cận
mà lỗi kiểm tra $\epsilon_\mathcal{D}(f)$ hội tụ về lỗi thực $\epsilon(f)$,
chúng ta có thể xem xét kỹ hơn một số chi tiết quan trọng.
Hãy nhớ rằng biến ngẫu nhiên đang xét
$\mathbf{1}(f(X) \neq Y)$
chỉ có thể nhận các giá trị $0$ và $1$
và do đó là một biến ngẫu nhiên Bernoulli,
được đặc trưng bởi một tham số
cho biết xác suất nó nhận giá trị $1$.
Ở đây, $1$ có nghĩa là classifier của chúng ta đã mắc lỗi,
nên tham số của biến ngẫu nhiên thực chất
là tỷ lệ lỗi thực $\epsilon(f)$.
Phương sai $\sigma^2$ của một biến Bernoulli
phụ thuộc vào tham số của nó (ở đây là $\epsilon(f)$)
theo biểu thức $\epsilon(f)(1-\epsilon(f))$.
Mặc dù ban đầu chưa biết $\epsilon(f)$,
chúng ta biết rằng nó không thể lớn hơn $1$.
Khảo sát một chút hàm này
cho thấy phương sai của chúng ta đạt cao nhất
khi tỷ lệ lỗi thực gần $0.5$
và có thể thấp hơn nhiều khi nó
gần $0$ hoặc gần $1$.
Điều này cho chúng ta biết rằng độ lệch chuẩn tiệm cận
của ước lượng $\epsilon_\mathcal{D}(f)$ cho lỗi $\epsilon(f)$
(trên lựa chọn $n$ mẫu kiểm tra)
không thể lớn hơn $\sqrt{0.25/n}$.

Nếu bỏ qua thực tế rằng tốc độ này đặc trưng cho
hành vi khi kích thước tập kiểm tra tiến đến vô hạn
thay vì khi chúng ta có số lượng mẫu hữu hạn,
điều này cho biết rằng nếu muốn lỗi kiểm tra $\epsilon_\mathcal{D}(f)$
xấp xỉ lỗi quần thể $\epsilon(f)$
sao cho một độ lệch chuẩn tương ứng
với khoảng $\pm 0.01$,
thì chúng ta nên thu thập xấp xỉ 2500 mẫu.
Nếu muốn đặt hai độ lệch chuẩn
trong phạm vi đó và do đó có độ tin cậy 95%
rằng $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$,
thì chúng ta sẽ cần 10.000 mẫu!

Đây chính là kích thước của các tập kiểm tra
đối với nhiều benchmark phổ biến trong machine learning.
Bạn có thể ngạc nhiên khi biết rằng mỗi năm có hàng nghìn
bài báo deep learning ứng dụng được xuất bản
và nhấn mạnh các cải thiện về tỷ lệ lỗi chỉ $0.01$ hoặc ít hơn.
Tất nhiên, khi các tỷ lệ lỗi gần $0$ hơn nhiều,
thì một cải thiện $0.01$ thực sự có thể là một vấn đề lớn.


Một đặc điểm gây khó chịu trong phân tích của chúng ta cho đến đây
là nó thực sự chỉ cho biết về tính tiệm cận,
nghĩa là mối quan hệ giữa $\epsilon_\mathcal{D}$ và $\epsilon$
phát triển như thế nào khi kích thước mẫu tiến đến vô hạn.
May mắn thay, vì biến ngẫu nhiên của chúng ta bị chặn,
chúng ta có thể thu được các cận hợp lệ cho mẫu hữu hạn
bằng cách áp dụng một bất đẳng thức của Hoeffding (1963):

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Giải kích thước tập dữ liệu nhỏ nhất
cho phép chúng ta kết luận
với độ tin cậy 95% rằng khoảng cách $t$
giữa ước lượng $\epsilon_\mathcal{D}(f)$
và tỷ lệ lỗi thực $\epsilon(f)$
không vượt quá $0.01$,
bạn sẽ thấy rằng cần khoảng 15.000 ví dụ
so với 10.000 ví dụ được gợi ý
bởi phân tích tiệm cận ở trên.
Nếu tìm hiểu sâu hơn về thống kê,
bạn sẽ thấy rằng xu hướng này nhìn chung vẫn đúng.
Các đảm bảo vẫn đúng ngay cả với mẫu hữu hạn
thường có phần thận trọng hơn một chút.
Lưu ý rằng xét trên tổng thể,
các con số này không quá chênh lệch,
phản ánh tính hữu ích chung
của phân tích tiệm cận trong việc cung cấp
cho chúng ta các con số ước lượng sơ bộ, ngay cả khi chúng không phải là
những đảm bảo mà ta có thể dựa vào trước pháp luật.

## Tái sử dụng Tập kiểm tra

Theo một nghĩa nào đó, giờ đây bạn đã sẵn sàng
để thành công trong việc thực hiện nghiên cứu machine learning thực nghiệm.
Hầu như mọi mô hình thực tế đều được phát triển
và xác thực dựa trên hiệu năng của tập kiểm tra,
và giờ đây bạn đã là bậc thầy về tập kiểm tra.
Với bất kỳ classifier cố định $f$ nào,
bạn biết cách đánh giá lỗi kiểm tra $\epsilon_\mathcal{D}(f)$ của nó,
và biết chính xác điều gì có thể (và không thể)
được nói về lỗi quần thể $\epsilon(f)$ của nó.

Vậy giả sử bạn vận dụng kiến thức này
và chuẩn bị huấn luyện mô hình đầu tiên $f_1$.
Biết chính xác mức độ tự tin cần có
đối với tỷ lệ lỗi của classifier,
bạn áp dụng phân tích ở trên để xác định
số lượng ví dụ phù hợp
cần dành riêng cho tập kiểm tra.
Hơn nữa, giả sử rằng bạn đã ghi nhớ các bài học từ
:numref:`sec_generalization_basics`
và đảm bảo duy trì tính bất khả xâm phạm của tập kiểm tra
bằng cách thực hiện toàn bộ phân tích sơ bộ,
tuning hyperparameter, và thậm chí cả việc lựa chọn
giữa nhiều kiến trúc mô hình cạnh tranh
trên một tập validation.
Cuối cùng, bạn đánh giá mô hình $f_1$
trên tập kiểm tra và báo cáo một ước lượng không chệch
của lỗi quần thể
kèm theo một khoảng tin cậy.

Cho đến đây, mọi thứ dường như đang diễn ra tốt đẹp.
Tuy nhiên, đêm đó bạn tỉnh dậy lúc 3 giờ sáng
với một ý tưởng xuất sắc cho một phương pháp mô hình hóa mới.
Ngày hôm sau, bạn code mô hình mới,
tune các hyperparameter trên tập validation,
và không những bạn làm cho mô hình mới $f_2$ hoạt động
mà tỷ lệ lỗi của nó còn có vẻ thấp hơn nhiều so với $f_1$.
Tuy nhiên, cảm giác phấn khích khi khám phá đột nhiên phai nhạt
khi bạn chuẩn bị cho bước đánh giá cuối cùng.
Bạn không có tập kiểm tra!

Mặc dù tập kiểm tra ban đầu $\mathcal{D}$
vẫn nằm trên server của bạn,
giờ đây bạn đối mặt với hai vấn đề đáng gờm.
Thứ nhất, khi thu thập tập kiểm tra,
bạn đã xác định mức độ chính xác cần thiết
dựa trên giả định rằng mình đang đánh giá
một classifier duy nhất $f$.
Tuy nhiên, nếu bạn bắt đầu
đánh giá nhiều classifier $f_1, ..., f_k$
trên cùng một tập kiểm tra,
bạn phải xem xét vấn đề phát hiện sai.
Trước đây, bạn có thể chắc chắn 95%
rằng $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$
đối với một classifier duy nhất $f$,
và do đó xác suất nhận được một kết quả gây hiểu lầm
chỉ là 5%.
Khi có $k$ classifier trong cuộc,
có thể khó đảm bảo
rằng không có dù chỉ một classifier trong số đó
có hiệu năng trên tập kiểm tra gây hiểu lầm.
Với 20 classifier đang được xem xét,
bạn có thể hoàn toàn không đủ khả năng
để loại trừ khả năng
rằng ít nhất một classifier trong số chúng
nhận được một điểm số gây hiểu lầm.
Vấn đề này liên quan đến kiểm định nhiều giả thuyết,
và mặc dù có một kho tài liệu đồ sộ trong thống kê,
nó vẫn là một vấn đề dai dẳng gây khó khăn cho nghiên cứu khoa học.


Nếu điều đó vẫn chưa đủ khiến bạn lo lắng,
còn có một lý do đặc biệt để không tin tưởng
vào các kết quả nhận được trong những lần đánh giá sau.
Hãy nhớ rằng phân tích của chúng ta về hiệu năng tập kiểm tra
dựa trên giả định rằng classifier
được chọn mà không hề tiếp xúc với tập kiểm tra,
và do đó chúng ta có thể xem tập kiểm tra
là được rút ngẫu nhiên từ quần thể nền tảng.
Ở đây, không chỉ bạn kiểm tra nhiều hàm,
mà hàm tiếp theo $f_2$ còn được chọn
sau khi bạn đã quan sát hiệu năng trên tập kiểm tra của $f_1$.
Một khi thông tin từ tập kiểm tra đã bị rò rỉ đến người xây dựng mô hình,
nó không bao giờ có thể lại là một tập kiểm tra thực sự theo nghĩa chặt chẽ nhất.
Vấn đề này được gọi là *overfitting thích nghi (adaptive overfitting)* và gần đây đã nổi lên
như một chủ đề được các nhà lý thuyết học máy và nhà thống kê đặc biệt quan tâm
:cite:`dwork2015preserving`.
May mắn thay, mặc dù có thể
làm rò rỉ toàn bộ thông tin từ một tập holdout,
và các kịch bản xấu nhất về mặt lý thuyết khá ảm đạm,
những phân tích này có thể quá thận trọng.
Trong thực tế, hãy cẩn thận tạo ra các tập kiểm tra thực sự,
tham khảo chúng ít thường xuyên nhất có thể,
tính đến kiểm định nhiều giả thuyết
khi báo cáo các khoảng tin cậy,
và tăng cường mức độ cảnh giác quyết liệt hơn
khi mức độ rủi ro cao và kích thước tập dữ liệu nhỏ.
Khi tổ chức một loạt thử thách benchmark,
thông lệ tốt thường là duy trì
nhiều tập kiểm tra để sau mỗi vòng,
tập kiểm tra cũ có thể được hạ xuống thành tập validation.





## Lý thuyết Học Thống kê

Nói đơn giản, *tập kiểm tra là tất cả những gì chúng ta thực sự có*,
nhưng thực tế này lại có vẻ không thỏa đáng một cách kỳ lạ.
Trước hết, chúng ta hiếm khi sở hữu một *tập kiểm tra thực sự*---trừ khi
chúng ta là người tạo ra tập dữ liệu,
có lẽ người khác đã đánh giá
classifier của chính họ trên "tập kiểm tra" bề ngoài của chúng ta.
Và ngay cả khi chúng ta được quyền sử dụng đầu tiên,
chúng ta nhanh chóng cảm thấy thất vọng, mong ước có thể
đánh giá các nỗ lực mô hình hóa tiếp theo
mà không có cảm giác day dứt
rằng mình không thể tin tưởng các con số.
Hơn nữa, ngay cả một tập kiểm tra thực sự cũng chỉ có thể cho chúng ta biết *sau thực tế*
liệu một classifier trên thực tế có khái quát hóa cho quần thể hay không,
chứ không phải liệu chúng ta có bất kỳ lý do nào để kỳ vọng *trước thực tế*
rằng nó sẽ khái quát hóa hay không.

Với những băn khoăn này,
giờ đây bạn có thể đã được chuẩn bị đủ
để thấy được sức hấp dẫn của *lý thuyết học thống kê*,
phân ngành toán học của machine learning
mà những người thực hành hướng đến việc làm sáng tỏ
các nguyên lý cơ bản giải thích
vì sao/khi nào các mô hình được huấn luyện trên dữ liệu thực nghiệm
có thể/sẽ khái quát hóa sang dữ liệu chưa quan sát.
Một trong các mục tiêu chính
của các nhà nghiên cứu học thống kê
là chặn trên khoảng cách khái quát hóa,
liên hệ các thuộc tính của lớp mô hình
với số lượng mẫu trong tập dữ liệu.

Các nhà lý thuyết học tập hướng đến việc chặn trên sự khác biệt
giữa *lỗi thực nghiệm* $\epsilon_\mathcal{S}(f_\mathcal{S})$
của một classifier đã học $f_\mathcal{S}$,
vừa được huấn luyện vừa được đánh giá
trên tập huấn luyện $\mathcal{S}$,
và lỗi thực $\epsilon(f_\mathcal{S})$
của chính classifier đó trên quần thể nền tảng.
Điều này có thể trông giống vấn đề đánh giá
mà chúng ta vừa đề cập, nhưng có một khác biệt lớn.
Trước đó, classifier $f$ là cố định
và chúng ta chỉ cần một tập dữ liệu
cho mục đích đánh giá.
Thật vậy, bất kỳ classifier cố định nào cũng khái quát hóa:
lỗi của nó trên một tập dữ liệu (chưa từng thấy trước đó)
là một ước lượng không chệch của lỗi quần thể.
Nhưng chúng ta có thể nói gì khi một classifier
được huấn luyện và đánh giá trên cùng một tập dữ liệu?
Liệu chúng ta có thể tự tin rằng lỗi huấn luyện
sẽ gần với lỗi kiểm tra không?


Giả sử classifier đã học $f_\mathcal{S}$ của chúng ta phải được chọn
từ một tập hàm được chỉ định trước nào đó $\mathcal{F}$.
Hãy nhớ từ phần thảo luận về tập kiểm tra
rằng mặc dù dễ dàng ước lượng
lỗi của một classifier duy nhất,
mọi thứ trở nên phức tạp khi chúng ta bắt đầu
xem xét các tập hợp classifier.
Ngay cả khi lỗi thực nghiệm
của bất kỳ classifier (cố định) nào
sẽ gần với lỗi thực của nó
với xác suất cao,
một khi xem xét một tập hợp classifier,
chúng ta cần lo về khả năng
rằng *chỉ một* classifier trong số đó
nhận được một ước lượng lỗi rất kém.
Điều đáng lo là chúng ta có thể chọn classifier như vậy
và do đó đánh giá thấp nghiêm trọng
lỗi quần thể.
Hơn nữa, ngay cả với các mô hình tuyến tính,
vì tham số của chúng có giá trị liên tục,
chúng ta thường chọn từ
một lớp hàm vô hạn ($|\mathcal{F}| = \infty$).

Một giải pháp đầy tham vọng cho vấn đề này
là phát triển các công cụ phân tích
để chứng minh hội tụ đều, tức là,
với xác suất cao,
tỷ lệ lỗi thực nghiệm của mọi classifier trong lớp $f\in\mathcal{F}$
sẽ hội tụ *đồng thời* về tỷ lệ lỗi thực của nó.
Nói cách khác, chúng ta tìm kiếm một nguyên lý lý thuyết
cho phép phát biểu rằng
với xác suất ít nhất là $1-\delta$
(với một $\delta$ nhỏ nào đó),
tỷ lệ lỗi $\epsilon(f)$ của không classifier nào
(trong số tất cả classifier thuộc lớp $\mathcal{F}$)
sẽ bị ước lượng sai vượt quá
một lượng nhỏ nào đó $\alpha$.
Rõ ràng, chúng ta không thể đưa ra những phát biểu như vậy
cho mọi lớp mô hình $\mathcal{F}$.
Hãy nhớ lớp các máy ghi nhớ
luôn đạt lỗi thực nghiệm bằng $0$
nhưng không bao giờ tốt hơn việc đoán ngẫu nhiên
trên quần thể nền tảng.

Theo một nghĩa nào đó, lớp các mô hình ghi nhớ quá linh hoạt.
Không thể có bất kỳ kết quả hội tụ đều nào như vậy.
Mặt khác, một classifier cố định thì vô dụng---nó
khái quát hóa hoàn hảo, nhưng không khớp cả
dữ liệu huấn luyện lẫn dữ liệu kiểm tra.
Do đó, câu hỏi trung tâm của học tập
trong lịch sử được định khung như một sự đánh đổi
giữa các lớp mô hình linh hoạt hơn (phương sai cao hơn)
khớp dữ liệu huấn luyện tốt hơn nhưng có nguy cơ overfitting,
và các lớp mô hình cứng nhắc hơn (độ chệch cao hơn)
khái quát hóa tốt nhưng có nguy cơ underfitting.
Một câu hỏi trung tâm trong lý thuyết học tập
là phát triển phân tích toán học phù hợp
để định lượng
vị trí của một mô hình trên phổ này,
và cung cấp các đảm bảo tương ứng.

Trong một loạt bài báo mang tính nền tảng,
Vapnik và Chervonenkis đã mở rộng
lý thuyết về sự hội tụ
của các tần suất tương đối
sang các lớp hàm tổng quát hơn
:cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91`.
Một trong những đóng góp chủ chốt của hướng nghiên cứu này
là chiều Vapnik--Chervonenkis (VC),
đo lường (một khái niệm về)
độ phức tạp (tính linh hoạt) của một lớp mô hình.
Hơn nữa, một trong các kết quả quan trọng của họ chặn
sự khác biệt giữa lỗi thực nghiệm
và lỗi tổng thể theo một hàm
của chiều VC và số lượng mẫu:

$$P\left(R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta
\ \textrm{ for }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$

Ở đây, $\delta > 0$ là xác suất mà cận bị vi phạm,
$\alpha$ là cận trên của khoảng cách tổng quát hóa,
và $n$ là kích thước tập dữ liệu.
Cuối cùng, $c > 0$ là một hằng số chỉ phụ thuộc
vào thang đo của loss có thể phát sinh.
Một cách sử dụng cận này là thay các giá trị mong muốn
của $\delta$ và $\alpha$
để xác định cần thu thập bao nhiêu mẫu.
Chiều VC định lượng số lượng lớn nhất
các điểm dữ liệu mà ta có thể gán
bất kỳ nhãn (nhị phân) tùy ý nào,
và với mỗi cách gán nhãn, tìm được một mô hình $f$ trong lớp
phù hợp với cách gán nhãn đó.
Ví dụ, các mô hình tuyến tính trên đầu vào $d$ chiều
có chiều VC là $d+1$.
Dễ thấy rằng một đường thẳng có thể gán
mọi cách gán nhãn có thể cho ba điểm trong hai chiều,
nhưng không thể cho bốn điểm.
Đáng tiếc là lý thuyết này thường
quá bi quan đối với các mô hình phức tạp hơn,
và để đạt được đảm bảo này thường cần
nhiều ví dụ hơn rất nhiều so với thực tế cần thiết
để đạt tỷ lệ lỗi mong muốn.
Cũng lưu ý rằng khi cố định lớp mô hình và $\delta$,
tỷ lệ lỗi của chúng ta một lần nữa giảm
với tốc độ thông thường $\mathcal{O}(1/\sqrt{n})$.
Dường như khó có thể làm tốt hơn theo $n$.
Tuy nhiên, khi ta thay đổi lớp mô hình,
chiều VC có thể đưa ra
một bức tranh bi quan
về khoảng cách tổng quát hóa.





## Tóm tắt

Cách trực tiếp nhất để đánh giá một mô hình
là tham khảo một tập kiểm tra gồm dữ liệu chưa từng thấy trước đó.
Các đánh giá trên tập kiểm tra cung cấp một ước lượng không chệch của lỗi thực
và hội tụ với tốc độ $\mathcal{O}(1/\sqrt{n})$ mong muốn khi tập kiểm tra tăng lên.
Ta có thể đưa ra các khoảng tin cậy gần đúng
dựa trên các phân phối tiệm cận chính xác
hoặc các khoảng tin cậy hợp lệ với mẫu hữu hạn
dựa trên các đảm bảo mẫu hữu hạn (bảo thủ hơn).
Thực tế, đánh giá trên tập kiểm tra là nền tảng
của nghiên cứu machine learning hiện đại.
Tuy nhiên, các tập kiểm tra hiếm khi là tập kiểm tra thực sự
(được nhiều nhà nghiên cứu sử dụng lặp đi lặp lại).
Khi cùng một tập kiểm tra được dùng
để đánh giá nhiều mô hình,
việc kiểm soát false discovery có thể khó khăn.
Điều này có thể gây ra những vấn đề rất lớn trong lý thuyết.
Trong thực tế, mức độ nghiêm trọng của vấn đề
phụ thuộc vào kích thước của các tập holdout đang xét
và việc chúng chỉ được dùng để chọn hyperparameter
hay đang làm rò rỉ thông tin trực tiếp hơn.
Tuy nhiên, thực hành tốt là xây dựng các tập kiểm tra thực sự (hoặc nhiều tập)
và bảo thủ tối đa về tần suất sử dụng chúng.


Với hy vọng đưa ra một giải pháp thỏa đáng hơn,
các nhà lý thuyết học thống kê đã phát triển các phương pháp
để đảm bảo hội tụ đồng đều trên một lớp mô hình.
Nếu thực sự lỗi thực nghiệm của mọi mô hình đồng thời
hội tụ về lỗi thực của nó,
thì ta tự do chọn mô hình có hiệu năng
tốt nhất, bằng cách tối thiểu hóa lỗi huấn luyện,
và biết rằng nó cũng sẽ hoạt động tốt tương tự
trên dữ liệu holdout.
Điều then chốt là bất kỳ kết quả nào như vậy cũng phải phụ thuộc
vào một số thuộc tính của lớp mô hình.
Vladimir Vapnik và Alexey Chernovenkis
đã giới thiệu chiều VC,
trình bày các kết quả hội tụ đồng đều
áp dụng cho mọi mô hình trong một lớp VC.
Lỗi huấn luyện của tất cả mô hình trong lớp
được đảm bảo (đồng thời)
là gần với lỗi thực của chúng,
và được đảm bảo sẽ ngày càng gần hơn
với tốc độ $\mathcal{O}(1/\sqrt{n})$.
Sau phát hiện mang tính cách mạng về chiều VC,
nhiều thước đo độ phức tạp thay thế đã được đề xuất,
mỗi thước đo hỗ trợ một đảm bảo tổng quát hóa tương tự.
Xem :citet:`boucheron2005theory` để có thảo luận chi tiết
về một số cách nâng cao để đo độ phức tạp của hàm.
Đáng tiếc là, mặc dù các thước đo độ phức tạp này
đã trở thành những công cụ hữu ích rộng rãi trong lý thuyết thống kê,
chúng lại tỏ ra bất lực
(khi được áp dụng trực tiếp)
trong việc giải thích tại sao các mạng nơ-ron sâu có thể tổng quát hóa.
Các mạng nơ-ron sâu thường có hàng triệu tham số (hoặc nhiều hơn),
và có thể dễ dàng gán nhãn ngẫu nhiên cho các tập hợp điểm lớn.
Tuy nhiên, chúng tổng quát hóa tốt trên các bài toán thực tế
và, đáng ngạc nhiên, chúng thường tổng quát hóa tốt hơn
khi lớn hơn và sâu hơn,
mặc dù có chiều VC cao hơn.
Trong chương tiếp theo, chúng ta sẽ xem xét lại vấn đề tổng quát hóa
trong bối cảnh deep learning.

## Bài tập

1. Nếu muốn ước lượng lỗi của một mô hình cố định $f$
   trong phạm vi $0.0001$ với xác suất lớn hơn 99.9%,
   ta cần bao nhiêu mẫu?
1. Giả sử một người khác sở hữu một tập kiểm tra có nhãn
   $\mathcal{D}$ và chỉ cung cấp các đầu vào (feature) không có nhãn.
   Bây giờ, giả sử bạn chỉ có thể truy cập các nhãn của tập kiểm tra
   bằng cách chạy một mô hình $f$ (không có hạn chế nào đối với lớp mô hình)
   trên từng đầu vào không có nhãn
   và nhận lỗi tương ứng $\epsilon_\mathcal{D}(f)$.
   Bạn cần đánh giá bao nhiêu mô hình
   trước khi làm rò rỉ toàn bộ tập kiểm tra
   và do đó có thể trông như có lỗi $0$,
   bất kể lỗi thực của bạn là bao nhiêu?
1. Chiều VC của lớp đa thức bậc năm là bao nhiêu?
1. Chiều VC của các hình chữ nhật song song với trục trên dữ liệu hai chiều là bao nhiêu?

[Thảo luận](https://discuss.d2l.ai/t/6829)