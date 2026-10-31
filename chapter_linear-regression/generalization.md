# Khả năng khái quát hóa
:label:`sec_generalization_basics`

Hãy xem xét hai sinh viên đại học đang chăm chỉ
chuẩn bị cho kỳ thi cuối kỳ.
Thông thường, quá trình chuẩn bị này sẽ bao gồm
việc luyện tập và kiểm tra năng lực
bằng cách làm các kỳ thi được tổ chức trong những năm trước.
Tuy nhiên, làm tốt các kỳ thi trước đây không đảm bảo
rằng họ sẽ đạt kết quả xuất sắc khi điều đó thực sự quan trọng.
Chẳng hạn, hãy tưởng tượng một sinh viên tên Extraordinary Ellie,
người chỉ chuẩn bị bằng cách
ghi nhớ đáp án
cho các câu hỏi thi của những năm trước.
Ngay cả khi Ellie có
một trí nhớ phi thường,
và do đó có thể nhớ chính xác đáp án
cho bất kỳ câu hỏi nào *đã từng thấy trước đây*,
cô ấy vẫn có thể bị khựng lại
khi gặp một câu hỏi mới (*chưa từng thấy trước đây*).
Ngược lại, hãy tưởng tượng một sinh viên khác,
Inductive Irene, có khả năng
ghi nhớ kém tương tự,
nhưng lại rất giỏi nhận ra các mẫu.
Lưu ý rằng nếu kỳ thi thực sự bao gồm
các câu hỏi được dùng lại từ một năm trước,
Ellie sẽ dễ dàng vượt qua Irene.
Ngay cả khi các mẫu mà Irene suy ra
cho dự đoán chính xác 90%,
chúng cũng không thể cạnh tranh với
khả năng nhớ lại 100% của Ellie.
Tuy nhiên, ngay cả khi kỳ thi bao gồm
hoàn toàn các câu hỏi mới,
Irene vẫn có thể duy trì điểm trung bình 90%.

Với tư cách là các nhà khoa học machine learning,
mục tiêu của chúng ta là phát hiện *các mẫu*.
Nhưng làm thế nào để chắc chắn rằng chúng ta
đã thực sự phát hiện ra một *mẫu tổng quát*
chứ không chỉ ghi nhớ dữ liệu?
Phần lớn thời gian, các dự đoán của chúng ta chỉ hữu ích
nếu model phát hiện ra một mẫu như vậy.
Chúng ta không muốn dự đoán giá cổ phiếu của ngày hôm qua, mà là của ngày mai.
Chúng ta không cần nhận diện
các bệnh đã được chẩn đoán
ở những bệnh nhân đã từng gặp,
mà cần nhận diện
các bệnh chưa được chẩn đoán
ở những bệnh nhân chưa từng gặp.
Bài toán này---làm thế nào để phát hiện các mẫu có *khả năng khái quát hóa*---là
bài toán nền tảng của machine learning,
và có thể nói là của toàn bộ lĩnh vực thống kê.
Chúng ta có thể xem bài toán này như một lát cắt
của một câu hỏi lớn hơn rất nhiều
bao trùm toàn bộ khoa học:
khi nào chúng ta thực sự có cơ sở
để thực hiện bước nhảy từ những quan sát cụ thể
đến các phát biểu tổng quát hơn?


Trong đời thực, chúng ta phải fit model
bằng một tập dữ liệu hữu hạn.
Quy mô điển hình của dữ liệu đó
khác nhau rất lớn giữa các lĩnh vực.
Đối với nhiều bài toán y khoa quan trọng,
chúng ta chỉ có thể tiếp cận vài nghìn điểm dữ liệu.
Khi nghiên cứu các bệnh hiếm,
chúng ta có thể may mắn tiếp cận được vài trăm điểm.
Ngược lại, các dataset công khai lớn nhất
bao gồm các ảnh được gán nhãn,
ví dụ như ImageNet :cite:`Deng.Dong.Socher.ea.2009`,
chứa hàng triệu hình ảnh.
Và một số tập hợp ảnh chưa gán nhãn
như dataset Flickr YFC100M
thậm chí còn lớn hơn, chứa
hơn 100 triệu hình ảnh :cite:`thomee2016yfcc100m`.
Tuy nhiên, ngay cả ở quy mô cực lớn này,
số lượng điểm dữ liệu hiện có
vẫn nhỏ đến vô cùng
so với không gian của mọi hình ảnh có thể có
ở độ phân giải megapixel.
Bất cứ khi nào làm việc với các mẫu hữu hạn,
chúng ta phải ghi nhớ rủi ro
rằng mình có thể fit dữ liệu training,
nhưng rồi phát hiện ra rằng mình đã không
khám phá được một mẫu có khả năng khái quát hóa.

Hiện tượng fit sát với dữ liệu training hơn
so với phân phối nền tảng được gọi là *overfitting*,
và các kỹ thuật chống overfitting
thường được gọi là các phương pháp *regularization*.
Mặc dù không thể thay thế cho phần nhập môn đầy đủ
về lý thuyết học thống kê (xem :citet:`Vapnik98,boucheron2005theory`),
chúng tôi sẽ cung cấp vừa đủ trực giác để bạn bắt đầu.
Chúng ta sẽ quay lại vấn đề khả năng khái quát hóa trong nhiều chương
xuyên suốt cuốn sách,
khám phá cả những gì đã biết về
các nguyên lý nền tảng của khả năng khái quát hóa
trong nhiều model khác nhau,
cũng như các kỹ thuật heuristic
đã được chứng minh (qua thực nghiệm)
là giúp cải thiện khả năng khái quát hóa
trên những tác vụ có ý nghĩa thực tiễn.



## Sai số Training và Sai số Khái quát hóa


Trong bối cảnh supervised learning tiêu chuẩn,
chúng ta giả định rằng dữ liệu training và dữ liệu test
được lấy mẫu *độc lập* từ các phân phối *giống hệt nhau*.
Điều này thường được gọi là *giả định IID*.
Mặc dù đây là một giả định mạnh,
đáng lưu ý rằng nếu không có giả định như vậy,
chúng ta sẽ hoàn toàn không có cơ sở để tiếp tục.
Tại sao chúng ta lại tin rằng dữ liệu training
được lấy mẫu từ phân phối $P(X,Y)$
có thể cho chúng ta biết cách dự đoán trên
dữ liệu test được tạo ra bởi một *phân phối khác* $Q(X,Y)$?
Hóa ra, việc thực hiện những bước nhảy như vậy đòi hỏi
các giả định mạnh về mối quan hệ giữa $P$ và $Q$.
Sau này chúng ta sẽ thảo luận về một số giả định
cho phép phân phối thay đổi,
nhưng trước hết chúng ta cần hiểu trường hợp IID,
trong đó $P(\cdot) = Q(\cdot)$.

Trước tiên, chúng ta cần phân biệt giữa
*sai số training* $R_\textrm{emp}$,
là một *thống kê*
được tính trên dataset training,
và *sai số khái quát hóa* $R$,
là một *kỳ vọng* được lấy
theo phân phối nền tảng.
Bạn có thể hình dung sai số khái quát hóa là
điều bạn sẽ quan sát được nếu áp dụng model
cho một luồng vô hạn các ví dụ dữ liệu bổ sung
được lấy mẫu từ cùng một phân phối dữ liệu nền tảng.
Về mặt hình thức, sai số training được biểu diễn dưới dạng một *tổng* (với cùng ký hiệu như :numref:`sec_linear_regression`):

$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$


trong khi sai số khái quát hóa được biểu diễn dưới dạng một tích phân:

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] =
\int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Vấn đề là chúng ta không bao giờ có thể tính chính xác
sai số khái quát hóa $R$.
Không ai cho chúng ta biết dạng chính xác
của hàm mật độ $p(\mathbf{x}, y)$.
Hơn nữa, chúng ta không thể lấy mẫu một luồng vô hạn các điểm dữ liệu.
Do đó, trong thực tế, chúng ta phải *ước lượng* sai số khái quát hóa
bằng cách áp dụng model vào một tập test độc lập
được tạo thành từ một lựa chọn ngẫu nhiên các ví dụ
$\mathbf{X}'$ và nhãn $\mathbf{y}'$
đã được giữ lại, không đưa vào tập training.
Việc này bao gồm áp dụng cùng công thức
đã được dùng để tính sai số training thực nghiệm
nhưng trên một tập test $\mathbf{X}', \mathbf{y}'$.


Điều quan trọng là khi đánh giá classifier trên tập test,
chúng ta đang làm việc với một classifier *cố định*
(nó không phụ thuộc vào mẫu của tập test),
và do đó việc ước lượng sai số của nó
chỉ đơn giản là bài toán ước lượng trung bình.
Tuy nhiên, điều tương tự không đúng
với tập training.
Lưu ý rằng model cuối cùng chúng ta có được
phụ thuộc rõ ràng vào việc lựa chọn tập training
và do đó, nhìn chung, sai số training sẽ là
một ước lượng thiên lệch của sai số thực
trên tổng thể nền tảng.
Khi đó, câu hỏi trọng tâm của khả năng khái quát hóa
là khi nào chúng ta có thể kỳ vọng sai số training
gần với sai số trên tổng thể
(và do đó gần với sai số khái quát hóa).

### Độ phức tạp của Model

Trong lý thuyết cổ điển, khi chúng ta có
các model đơn giản và dữ liệu dồi dào,
sai số training và sai số khái quát hóa có xu hướng gần nhau.
Tuy nhiên, khi làm việc với
các model phức tạp hơn và/hoặc ít ví dụ hơn,
chúng ta kỳ vọng sai số training giảm xuống
nhưng khoảng cách khái quát hóa tăng lên.
Điều này không có gì đáng ngạc nhiên.
Hãy tưởng tượng một lớp model có tính biểu đạt cao đến mức
với bất kỳ dataset nào gồm $n$ ví dụ,
chúng ta đều có thể tìm được một tập tham số
fit hoàn hảo các nhãn tùy ý,
ngay cả khi chúng được gán ngẫu nhiên.
Trong trường hợp này, ngay cả khi chúng ta fit hoàn hảo dữ liệu training,
làm thế nào chúng ta có thể kết luận bất cứ điều gì về sai số khái quát hóa?
Theo những gì chúng ta biết, sai số khái quát hóa của chúng ta
có thể không tốt hơn việc đoán ngẫu nhiên.

Nói chung, nếu không có bất kỳ hạn chế nào đối với lớp model,
chúng ta không thể kết luận, chỉ dựa trên việc fit dữ liệu training,
rằng model của mình đã phát hiện ra bất kỳ mẫu nào có khả năng khái quát hóa :cite:`vapnik1994measuring`.
Mặt khác, nếu lớp model của chúng ta
không có khả năng fit các nhãn tùy ý,
thì nó hẳn đã phát hiện ra một mẫu.
Các ý tưởng lý thuyết học về độ phức tạp của model
được truyền cảm hứng phần nào từ các ý tưởng
của Karl Popper, một triết gia khoa học có ảnh hưởng,
người đã hệ thống hóa tiêu chí về khả năng bị bác bỏ.
Theo Popper, một lý thuyết
có thể giải thích mọi và bất kỳ quan sát nào
thì hoàn toàn không phải là một lý thuyết khoa học!
Sau cùng, nó đã cho chúng ta biết gì về thế giới
nếu nó không loại trừ bất kỳ khả năng nào?
Tóm lại, điều chúng ta muốn là một giả thuyết
*không thể* giải thích bất kỳ quan sát nào
mà chúng ta có thể hình dung sẽ thực hiện
nhưng dù vậy lại tương thích
với những quan sát mà *thực tế* chúng ta thực hiện.

Vậy chính xác thì điều gì tạo nên
một khái niệm phù hợp về độ phức tạp của model
là một vấn đề phức tạp.
Thông thường, các model có nhiều tham số hơn
có thể fit được số lượng lớn hơn
các nhãn được gán tùy ý.
Tuy nhiên, điều này không phải lúc nào cũng đúng.
Chẳng hạn, các phương pháp kernel hoạt động trong những không gian
có số lượng tham số vô hạn,
nhưng độ phức tạp của chúng được kiểm soát
bằng các phương thức khác :cite:`Scholkopf.Smola.2002`.
Một khái niệm về độ phức tạp thường tỏ ra hữu ích
là miền giá trị mà các tham số có thể nhận.
Ở đây, một model có các tham số được phép
nhận các giá trị tùy ý
sẽ phức tạp hơn.
Chúng ta sẽ quay lại ý tưởng này trong phần tiếp theo,
khi giới thiệu *weight decay*,
kỹ thuật regularization thực tiễn đầu tiên của bạn.
Đáng chú ý là việc so sánh
độ phức tạp giữa các thành viên thuộc các lớp model khác biệt đáng kể
(ví dụ, decision tree so với neural network)
có thể rất khó.


Ở thời điểm này, chúng ta phải nhấn mạnh một điểm quan trọng khác
mà chúng ta sẽ quay lại khi giới thiệu các mạng neural sâu.
Khi một mô hình có khả năng khớp với các nhãn tùy ý,
sai số huấn luyện thấp không nhất thiết
ngụ ý sai số tổng quát hóa thấp.
*그러 Tuy nhiên, điều đó cũng không nhất thiết
ngụ ý sai số tổng quát hóa cao!*
Điều duy nhất chúng ta có thể khẳng định chắc chắn là
chỉ riêng sai số huấn luyện thấp thì chưa đủ
để chứng nhận sai số tổng quát hóa thấp.
Các mạng neural sâu hóa ra chính là những mô hình như vậy:
mặc dù chúng tổng quát hóa tốt trong thực tế,
chúng quá mạnh để cho phép chúng ta kết luận
nhiều điều chỉ dựa trên sai số huấn luyện.
Trong những trường hợp này, chúng ta phải phụ thuộc nhiều hơn
vào dữ liệu holdout để chứng nhận khả năng tổng quát hóa
sau khi mô hình đã được huấn luyện.
Sai số trên dữ liệu holdout, tức validation set,
được gọi là *sai số validation*.

## Underfitting hay Overfitting?

Khi so sánh sai số huấn luyện và sai số validation,
chúng ta cần lưu ý đến hai tình huống phổ biến.
Trước hết, chúng ta cần đề phòng những trường hợp
mà cả sai số huấn luyện lẫn sai số validation đều đáng kể
nhưng khoảng cách giữa chúng lại nhỏ.
Nếu mô hình không thể giảm sai số huấn luyện,
điều đó có thể có nghĩa là mô hình của chúng ta quá đơn giản
(tức là không đủ khả năng biểu đạt)
để nắm bắt mẫu hình mà chúng ta đang cố gắng mô hình hóa.
Hơn nữa, vì *khoảng cách tổng quát hóa* ($R_\textrm{emp} - R$)
giữa sai số huấn luyện và sai số tổng quát hóa của chúng ta là nhỏ,
chúng ta có lý do để tin rằng mình có thể sử dụng một mô hình phức tạp hơn.
Hiện tượng này được gọi là *underfitting*.

Mặt khác, như đã thảo luận ở trên,
chúng ta cần đề phòng những trường hợp
khi sai số huấn luyện thấp hơn đáng kể
so với sai số validation, cho thấy hiện tượng *overfitting* nghiêm trọng.
Lưu ý rằng overfitting không phải lúc nào cũng là điều xấu.
Đặc biệt trong deep learning,
những mô hình dự đoán tốt nhất thường đạt kết quả
tốt hơn rất nhiều trên dữ liệu huấn luyện so với dữ liệu holdout.
Sau cùng, điều chúng ta thường quan tâm là
giảm sai số tổng quát hóa,
và chỉ quan tâm đến khoảng cách này trong chừng mực
nó trở thành trở ngại cho mục tiêu đó.
Lưu ý rằng nếu sai số huấn luyện bằng 0,
thì khoảng cách tổng quát hóa chính xác bằng sai số tổng quát hóa
và chúng ta chỉ có thể tiến bộ bằng cách giảm khoảng cách này.

### Khớp đường cong đa thức
:label:`subsec_polynomial-curve-fitting`

Để minh họa một số trực giác kinh điển
về overfitting và độ phức tạp của mô hình,
hãy xét trường hợp sau:
cho dữ liệu huấn luyện gồm một đặc trưng duy nhất $x$
và nhãn thực tương ứng $y$,
chúng ta cố gắng tìm đa thức bậc $d$

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

để ước lượng nhãn $y$.
Đây chỉ là một bài toán linear regression,
trong đó các đặc trưng của chúng ta được cho bởi các lũy thừa của $x$,
các trọng số của mô hình được cho bởi $w_i$,
và bias được cho bởi $w_0$ vì $x^0 = 1$ với mọi $x$.
Vì đây chỉ là một bài toán linear regression,
chúng ta có thể sử dụng squared error làm hàm loss.

Một hàm đa thức bậc cao phức tạp hơn
một hàm đa thức bậc thấp,
vì đa thức bậc cao có nhiều tham số hơn
và phạm vi lựa chọn của hàm mô hình rộng hơn.
Với tập dữ liệu huấn luyện cố định,
các hàm đa thức bậc cao hơn luôn phải đạt được
sai số huấn luyện thấp hơn (hoặc tệ nhất là bằng)
so với các đa thức bậc thấp hơn.
Trên thực tế, bất cứ khi nào mỗi mẫu dữ liệu
có một giá trị $x$ riêng biệt,
một hàm đa thức có bậc
bằng số lượng mẫu dữ liệu
có thể khớp hoàn hảo với tập huấn luyện.
Chúng ta so sánh mối quan hệ giữa bậc đa thức (độ phức tạp của mô hình)
và cả underfitting lẫn overfitting trong :numref:`fig_capacity_vs_error`.

![Ảnh hưởng của độ phức tạp mô hình đến underfitting và overfitting.](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`


### Kích thước tập dữ liệu

Như giới hạn ở trên đã cho thấy,
một yếu tố quan trọng khác
cần ghi nhớ là kích thước tập dữ liệu.
Với mô hình cố định, càng ít mẫu
trong tập dữ liệu huấn luyện,
chúng ta càng có khả năng (và càng dễ bị nghiêm trọng hơn)
gặp phải overfitting.
Khi tăng lượng dữ liệu huấn luyện,
sai số tổng quát hóa thường giảm.
Hơn nữa, nhìn chung, nhiều dữ liệu hơn không bao giờ gây hại.
Với một tác vụ và phân phối dữ liệu cố định,
độ phức tạp của mô hình không nên tăng
nhanh hơn lượng dữ liệu.
Khi có nhiều dữ liệu hơn, chúng ta có thể cố gắng
khớp một mô hình phức tạp hơn.
Nếu không có đủ dữ liệu, các mô hình đơn giản hơn
có thể khó bị vượt qua hơn.
Đối với nhiều tác vụ, deep learning
chỉ vượt trội so với các mô hình tuyến tính
khi có sẵn hàng nghìn mẫu huấn luyện.
Một phần, thành công hiện tại của deep learning
đáng kể nhờ sự dồi dào của các tập dữ liệu khổng lồ
bắt nguồn từ các công ty Internet, lưu trữ giá rẻ,
các thiết bị được kết nối và quá trình số hóa rộng khắp nền kinh tế.

## Lựa chọn mô hình
:label:`subsec_generalization-model-selection`

Thông thường, chúng ta chỉ chọn mô hình cuối cùng
sau khi đánh giá nhiều mô hình
khác nhau theo nhiều cách
(các kiến trúc khác nhau, mục tiêu huấn luyện,
các đặc trưng được chọn, tiền xử lý dữ liệu,
learning rate, v.v.).
Việc lựa chọn giữa nhiều mô hình được gọi một cách thích hợp là
*lựa chọn mô hình*.

Về nguyên tắc, chúng ta không nên đụng đến test set
cho đến sau khi đã chọn xong tất cả hyperparameter.
Nếu sử dụng dữ liệu kiểm tra trong quá trình lựa chọn mô hình,
có nguy cơ chúng ta sẽ overfit dữ liệu kiểm tra.
Khi đó chúng ta sẽ gặp rắc rối nghiêm trọng.
Nếu overfit dữ liệu huấn luyện,
chúng ta vẫn luôn có kết quả đánh giá trên dữ liệu kiểm tra để giữ cho mình khách quan.
Nhưng nếu overfit dữ liệu kiểm tra, làm sao chúng ta có thể biết được?
Xem :citet:`ong2005learning` để biết một ví dụ về việc
điều này có thể dẫn đến những kết quả vô lý ngay cả với các mô hình
mà độ phức tạp có thể được kiểm soát chặt chẽ.

Vì vậy, chúng ta không bao giờ nên dựa vào dữ liệu kiểm tra để lựa chọn mô hình.
Tuy nhiên, chúng ta cũng không thể chỉ dựa vào dữ liệu huấn luyện
để lựa chọn mô hình, vì
chúng ta không thể ước lượng sai số tổng quát hóa
trên chính dữ liệu được dùng để huấn luyện mô hình.


Trong các ứng dụng thực tế, vấn đề trở nên phức tạp hơn.
Mặc dù lý tưởng nhất là chúng ta chỉ chạm đến dữ liệu kiểm tra một lần,
để đánh giá mô hình tốt nhất hoặc so sánh
một số ít mô hình với nhau,
dữ liệu kiểm tra trong thế giới thực hiếm khi bị loại bỏ sau một lần sử dụng.
Chúng ta hiếm khi có đủ khả năng tạo một test set mới cho mỗi vòng thử nghiệm.
Trên thực tế, việc tái sử dụng dữ liệu benchmark trong nhiều thập kỷ
có thể tác động đáng kể đến
quá trình phát triển các thuật toán,
ví dụ như [phân loại hình ảnh](https://paperswithcode.com/sota/image-classification-on-imagenet)
và [nhận dạng ký tự quang học](https://paperswithcode.com/sota/image-classification-on-mnist).

Cách làm phổ biến để giải quyết vấn đề *huấn luyện trên test set*
là chia dữ liệu thành ba phần,
bổ sung một *validation set*
bên cạnh tập dữ liệu huấn luyện và kiểm tra.
Kết quả là một tình huống không rõ ràng, trong đó ranh giới
giữa dữ liệu validation và dữ liệu kiểm tra mơ hồ một cách đáng lo ngại.
Trừ khi được nói rõ khác đi, trong các thí nghiệm của cuốn sách này
chúng ta thực sự đang làm việc với những gì đúng ra nên được gọi là
dữ liệu huấn luyện và dữ liệu validation, không có test set thực sự.
Do đó, độ chính xác được báo cáo trong mỗi thí nghiệm của cuốn sách thực chất là
độ chính xác validation chứ không phải độ chính xác trên test set thực sự.

### Cross-Validation

Khi dữ liệu huấn luyện khan hiếm,
chúng ta thậm chí có thể không đủ khả năng giữ lại
đủ dữ liệu để tạo thành một validation set phù hợp.
Một giải pháp phổ biến cho vấn đề này là sử dụng
$K$*-fold cross-validation*.
Ở đây, dữ liệu huấn luyện ban đầu được chia thành $K$ tập con không chồng lấn.
Sau đó, quá trình huấn luyện và validation mô hình được thực hiện $K$ lần,
mỗi lần huấn luyện trên $K-1$ tập con và validation
trên một tập con khác (tập con không được dùng để huấn luyện trong vòng đó).
Cuối cùng, sai số huấn luyện và validation được ước lượng
bằng cách lấy trung bình kết quả từ $K$ thí nghiệm.



## Tóm tắt

Phần này đã tìm hiểu một số nền tảng
của khả năng tổng quát hóa trong machine learning.
Một số ý tưởng này trở nên phức tạp
và phản trực giác khi chúng ta chuyển sang các mô hình sâu hơn; ở đây, các mô hình có khả năng overfit dữ liệu nghiêm trọng,
và những khái niệm liên quan đến độ phức tạp
có thể vừa ẩn vừa phản trực giác
(ví dụ: các kiến trúc lớn hơn với nhiều tham số hơn
lại tổng quát hóa tốt hơn).
Dưới đây là một số quy tắc kinh nghiệm:

1. Sử dụng validation set (hoặc $K$*-fold cross-validation*) để lựa chọn mô hình;
1. Các mô hình phức tạp hơn thường cần nhiều dữ liệu hơn;
1. Những khái niệm liên quan đến độ phức tạp bao gồm cả số lượng tham số và phạm vi các giá trị mà chúng được phép nhận;
1. Khi mọi yếu tố khác không đổi, nhiều dữ liệu hơn hầu như luôn dẫn đến khả năng tổng quát hóa tốt hơn;
1. Toàn bộ cuộc thảo luận này về khả năng tổng quát hóa đều dựa trên giả định IID. Nếu nới lỏng giả định này, cho phép các phân phối thay đổi giữa giai đoạn huấn luyện và kiểm tra, thì chúng ta không thể nói gì về khả năng tổng quát hóa nếu không có thêm một giả định (có thể là nhẹ hơn).


## Bài tập

1. Khi nào bạn có thể giải chính xác bài toán hồi quy đa thức?
1. Hãy đưa ra ít nhất năm ví dụ trong đó các biến ngẫu nhiên phụ thuộc khiến việc xem bài toán là dữ liệu IID trở nên không phù hợp.
1. Bạn có bao giờ có thể kỳ vọng thấy sai số huấn luyện bằng 0 không? Trong những trường hợp nào bạn sẽ thấy sai số tổng quát hóa bằng 0?
1. Tại sao $K$-fold cross-validation lại rất tốn kém khi tính toán?
1. Tại sao ước lượng sai số của $K$-fold cross-validation lại bị chệch?
1. VC dimension được định nghĩa là số điểm lớn nhất có thể được phân loại với các nhãn tùy ý $\{\pm 1\}$ bởi một hàm thuộc một lớp các hàm. Tại sao đây có thể không phải là một ý tưởng tốt để đo độ phức tạp của lớp hàm? Gợi ý: hãy xét độ lớn của các hàm.
1. Người quản lý giao cho bạn một tập dữ liệu khó, trên đó thuật toán hiện tại của bạn hoạt động không tốt. Bạn sẽ biện minh thế nào với người đó rằng mình cần thêm dữ liệu? Gợi ý: bạn không thể tăng lượng dữ liệu nhưng có thể giảm lượng dữ liệu.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/234)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17978)
:end_tab: