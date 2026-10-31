# Môi trường và Dịch chuyển Phân phối
:label:`sec_environment-and-distribution-shift`

Trong các phần trước, chúng ta đã thực hành
nhiều ứng dụng thực tế của machine learning,
huấn luyện mô hình trên nhiều tập dữ liệu khác nhau.
Tuy nhiên, chúng ta chưa từng dừng lại để suy nghĩ
về việc dữ liệu ban đầu đến từ đâu
hoặc cuối cùng chúng ta dự định làm gì
với đầu ra từ các mô hình của mình.
Quá thường xuyên, các nhà phát triển machine learning
sở hữu dữ liệu đã vội vàng phát triển mô hình
mà không dừng lại để cân nhắc những vấn đề cơ bản này.

Nhiều lần deploy machine learning thất bại
có thể được truy nguyên về sự thiếu sót này.
Đôi khi các mô hình dường như hoạt động xuất sắc
khi được đo bằng độ chính xác trên tập kiểm tra
nhưng lại thất bại nghiêm trọng khi deploy
khi phân phối dữ liệu đột ngột thay đổi.
Nguy hiểm hơn, đôi khi chính việc deploy một mô hình
có thể là tác nhân làm xáo trộn phân phối dữ liệu.
Ví dụ, giả sử chúng ta huấn luyện một mô hình
để dự đoán ai sẽ trả nợ thay vì vỡ nợ,
và nhận thấy rằng lựa chọn giày dép của người nộp đơn
có liên quan đến rủi ro vỡ nợ
(giày Oxford biểu thị khả năng trả nợ, giày sneaker biểu thị vỡ nợ).
Sau đó, chúng ta có thể có xu hướng
cấp khoản vay cho bất kỳ người nộp đơn nào đi giày Oxford
và từ chối tất cả người nộp đơn đi giày sneaker.

Trong trường hợp này, bước nhảy thiếu cân nhắc của chúng ta từ
nhận dạng mẫu sang ra quyết định
và việc không xem xét môi trường một cách phản biện
có thể dẫn đến những hậu quả tai hại.
Trước hết, ngay khi chúng ta bắt đầu
đưa ra quyết định dựa trên giày dép,
khách hàng sẽ nhận ra và thay đổi hành vi của họ.
Chẳng bao lâu sau, mọi người nộp đơn đều sẽ đi giày Oxford,
mà không có bất kỳ sự cải thiện tương ứng nào về mức độ đáng tin cậy tín dụng.
Hãy dành một phút để suy ngẫm điều này vì các vấn đề tương tự xuất hiện rất nhiều
trong nhiều ứng dụng machine learning:
bằng cách đưa các quyết định dựa trên mô hình vào môi trường,
chúng ta có thể làm hỏng mô hình.

Dù không thể trình bày đầy đủ các chủ đề này
trong một phần duy nhất,
mục tiêu của chúng ta ở đây là nêu ra một số mối quan ngại phổ biến,
và thúc đẩy tư duy phản biện
cần thiết để phát hiện sớm các tình huống như vậy,
giảm thiểu thiệt hại, và sử dụng machine learning một cách có trách nhiệm.
Một số giải pháp đơn giản
(hỏi dữ liệu "đúng"),
một số khác khó về mặt kỹ thuật
(implement một hệ thống reinforcement learning),
và những giải pháp khác yêu cầu chúng ta bước ra khỏi phạm vi
của dự đoán thống kê hoàn toàn và
đối mặt với những câu hỏi triết học khó khăn
liên quan đến việc áp dụng thuật toán một cách có đạo đức.

## Các loại Dịch chuyển Phân phối

Để bắt đầu, chúng ta giới hạn ở bối cảnh dự đoán thụ động,
xem xét các cách khác nhau mà phân phối dữ liệu có thể dịch chuyển
và những việc có thể thực hiện để cứu vãn hiệu năng mô hình.
Trong một thiết lập cổ điển, chúng ta giả định rằng dữ liệu huấn luyện
được lấy mẫu từ một phân phối $p_S(\mathbf{x},y)$ nào đó,
nhưng dữ liệu kiểm tra sẽ bao gồm
các ví dụ không có nhãn được lấy từ
một phân phối khác $p_T(\mathbf{x},y)$.
Ngay từ đây, chúng ta đã phải đối mặt với một thực tế đáng lo ngại.
Nếu không có bất kỳ giả định nào về cách $p_S$
và $p_T$ liên hệ với nhau,
thì không thể học được một bộ phân loại robust.

Hãy xét một bài toán phân loại nhị phân,
trong đó chúng ta muốn phân biệt chó và mèo.
Nếu phân phối có thể dịch chuyển theo những cách tùy ý,
thì thiết lập của chúng ta cho phép trường hợp bệnh lý
mà phân phối trên đầu vào vẫn giữ nguyên:
$p_S(\mathbf{x}) = p_T(\mathbf{x})$,
nhưng tất cả nhãn đều bị đảo:
$p_S(y \mid \mathbf{x}) = 1 - p_T(y \mid \mathbf{x})$.
Nói cách khác, nếu Chúa có thể đột ngột quyết định
rằng trong tương lai mọi "con mèo" giờ là chó
và thứ chúng ta trước đây gọi là "chó" giờ là mèo---mà không có
bất kỳ thay đổi nào trong phân phối đầu vào $p(\mathbf{x})$,
thì chúng ta hoàn toàn không thể phân biệt được thiết lập này
với một thiết lập mà phân phối hoàn toàn không thay đổi.

May mắn thay, dưới một số giả định giới hạn
về những cách dữ liệu của chúng ta có thể thay đổi trong tương lai,
các thuật toán có nguyên tắc có thể phát hiện sự dịch chuyển
và đôi khi thậm chí thích nghi ngay lập tức,
cải thiện độ chính xác của bộ phân loại ban đầu.

### Dịch chuyển Hiệp biến

Trong các loại dịch chuyển phân phối,
dịch chuyển hiệp biến có lẽ được nghiên cứu rộng rãi nhất.
Ở đây, chúng ta giả định rằng mặc dù phân phối đầu vào
có thể thay đổi theo thời gian, hàm gán nhãn,
tức là phân phối có điều kiện
$P(y \mid \mathbf{x})$ không thay đổi.
Các nhà thống kê gọi đây là *dịch chuyển hiệp biến*
vì vấn đề phát sinh do sự
dịch chuyển trong phân phối của các hiệp biến (đặc trưng).
Mặc dù đôi khi chúng ta có thể lập luận về dịch chuyển phân phối
mà không cần viện đến quan hệ nhân quả, hãy lưu ý rằng dịch chuyển hiệp biến
là giả định tự nhiên trong các bối cảnh
mà chúng ta tin rằng $\mathbf{x}$ gây ra $y$.

Hãy xem xét thách thức phân biệt mèo và chó.
Dữ liệu huấn luyện của chúng ta có thể gồm các hình ảnh như trong :numref:`fig_cat-dog-train`.

![Dữ liệu huấn luyện để phân biệt mèo và chó (minh họa: Lafeez Hossain / 500px / Getty Images; ilkermetinkursova / iStock / Getty Images Plus; GlobalP / iStock / Getty Images Plus; Musthafa Aboobakuru / 500px / Getty Images).](../img/cat-dog-train.png)
:label:`fig_cat-dog-train`


Tại thời điểm kiểm tra, chúng ta được yêu cầu phân loại các hình ảnh trong :numref:`fig_cat-dog-test`.

![Dữ liệu kiểm tra để phân biệt mèo và chó (minh họa: SIBAS_minich / iStock / Getty Images Plus; Ghrzuzudu / iStock / Getty Images Plus; id-work / DigitalVision Vectors / Getty Images; Yime / iStock / Getty Images Plus).](../img/cat-dog-test.png)
:label:`fig_cat-dog-test`

Tập huấn luyện gồm các ảnh chụp,
trong khi tập kiểm tra chỉ chứa hình hoạt họa.
Huấn luyện trên một tập dữ liệu có các đặc tính khác biệt đáng kể
so với tập kiểm tra
có thể gây ra vấn đề nếu không có một kế hoạch rõ ràng
để thích nghi với domain mới.

### Dịch chuyển Nhãn

*Dịch chuyển nhãn* mô tả vấn đề ngược lại.
Ở đây, chúng ta giả định rằng biên của nhãn $P(y)$
có thể thay đổi
nhưng phân phối có điều kiện theo lớp
$P(\mathbf{x} \mid y)$ vẫn cố định giữa các domain.
Dịch chuyển nhãn là một giả định hợp lý
khi chúng ta tin rằng $y$ gây ra $\mathbf{x}$.
Ví dụ, chúng ta có thể muốn dự đoán chẩn đoán
dựa trên các triệu chứng (hoặc các biểu hiện khác),
ngay cả khi mức độ phổ biến tương đối của các chẩn đoán
thay đổi theo thời gian.
Dịch chuyển nhãn là giả định phù hợp ở đây
vì bệnh gây ra triệu chứng.
Trong một số trường hợp suy biến, các giả định dịch chuyển nhãn
và dịch chuyển hiệp biến có thể đồng thời đúng.
Ví dụ, khi nhãn mang tính xác định,
giả định dịch chuyển hiệp biến sẽ được thỏa mãn,
ngay cả khi $y$ gây ra $\mathbf{x}$.
Điều thú vị là trong những trường hợp này,
thường sẽ có lợi khi làm việc với các phương pháp
xuất phát từ giả định dịch chuyển nhãn.
Đó là vì các phương pháp này thường
thao tác trên những đối tượng giống nhãn (thường có số chiều thấp),
trái với các đối tượng giống đầu vào,
vốn thường có số chiều cao trong deep learning.

### Dịch chuyển Khái niệm

Chúng ta cũng có thể gặp vấn đề liên quan là *dịch chuyển khái niệm*,
xảy ra khi chính định nghĩa của các nhãn có thể thay đổi.
Điều này nghe có vẻ lạ---một *con mèo* vẫn là một *con mèo*, phải không?
Tuy nhiên, các danh mục khác có thể thay đổi cách sử dụng theo thời gian.
Tiêu chí chẩn đoán bệnh tâm thần,
những gì được coi là thời thượng, và chức danh công việc,
đều chịu những mức độ dịch chuyển khái niệm
đáng kể.
Hóa ra rằng nếu chúng ta di chuyển khắp Hoa Kỳ,
thay đổi nguồn dữ liệu theo địa lý,
chúng ta sẽ thấy sự dịch chuyển khái niệm đáng kể liên quan đến
phân phối các tên gọi của *nước ngọt có ga*
như minh họa trong :numref:`fig_popvssoda`.

![Dịch chuyển khái niệm về tên gọi nước ngọt có ga tại Hoa Kỳ (CC-BY: Alan McConchie, PopVsSoda.com).](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Nếu chúng ta xây dựng một hệ thống machine translation,
phân phối $P(y \mid \mathbf{x})$ có thể khác nhau
tùy thuộc vào vị trí của chúng ta.
Vấn đề này có thể khó nhận ra.
Chúng ta có thể hy vọng khai thác kiến thức
rằng sự dịch chuyển chỉ diễn ra dần dần
theo thời gian hoặc địa lý.

## Ví dụ về Dịch chuyển Phân phối

Trước khi đi sâu vào tính hình thức và các thuật toán,
chúng ta có thể thảo luận một số tình huống cụ thể
mà dịch chuyển hiệp biến hoặc dịch chuyển khái niệm có thể không rõ ràng.


### Chẩn đoán Y khoa

Hãy tưởng tượng rằng bạn muốn thiết kế một thuật toán phát hiện ung thư.
Bạn thu thập dữ liệu từ người khỏe mạnh và người bệnh
rồi huấn luyện thuật toán của mình.
Nó hoạt động tốt, mang lại độ chính xác cao
và bạn kết luận rằng mình đã sẵn sàng
cho một sự nghiệp thành công trong chẩn đoán y khoa.
*Chưa nhanh vậy.*

Các phân phối tạo ra dữ liệu huấn luyện
và những phân phối bạn sẽ gặp trong thực tế có thể khác biệt đáng kể.
Điều này đã xảy ra với một startup không may mắn
mà một số tác giả chúng tôi từng hợp tác cách đây nhiều năm.
Họ đang phát triển một xét nghiệm máu cho một căn bệnh
chủ yếu ảnh hưởng đến nam giới lớn tuổi
và hy vọng nghiên cứu nó bằng các mẫu máu
mà họ đã thu thập từ bệnh nhân.
Tuy nhiên, việc lấy mẫu máu từ nam giới khỏe mạnh
khó hơn đáng kể so với từ các bệnh nhân đã mắc bệnh trong hệ thống.
Để bù đắp, startup này đã kêu gọi
sinh viên tại một khuôn viên đại học hiến máu
để làm nhóm đối chứng khỏe mạnh trong quá trình phát triển xét nghiệm.
Sau đó, họ hỏi liệu chúng tôi có thể giúp họ
xây dựng một bộ phân loại để phát hiện căn bệnh này hay không.

Như chúng tôi đã giải thích với họ,
thực sự sẽ dễ dàng phân biệt
giữa các nhóm khỏe mạnh và mắc bệnh
với độ chính xác gần như hoàn hảo.
Tuy nhiên, đó là vì các đối tượng kiểm tra
khác nhau về độ tuổi, mức hormone,
hoạt động thể chất, chế độ ăn, mức tiêu thụ rượu,
và nhiều yếu tố khác không liên quan đến căn bệnh.
Điều này khó có thể đúng với bệnh nhân thực tế.
Do quy trình lấy mẫu của họ,
chúng tôi có thể kỳ vọng gặp phải covariate shift cực đoan.
Hơn nữa, trường hợp này khó có thể
được khắc phục bằng các phương pháp thông thường.
Tóm lại, họ đã lãng phí một khoản tiền đáng kể.



### Xe tự lái

Giả sử một công ty muốn tận dụng machine learning
để phát triển xe tự lái.
Một thành phần then chốt ở đây là bộ phát hiện lề đường.
Vì dữ liệu thực đã được gán nhãn rất tốn kém để có được,
họ có ý tưởng (thông minh nhưng cũng đáng nghi ngờ)
là sử dụng dữ liệu tổng hợp từ một game rendering engine
làm dữ liệu huấn luyện bổ sung.
Điều này hoạt động rất tốt trên "dữ liệu kiểm tra"
được lấy từ rendering engine.
Than ôi, bên trong một chiếc xe thật thì đó là thảm họa.
Hóa ra, lề đường đã được render
bằng một texture rất đơn giản.
Quan trọng hơn, *toàn bộ* lề đường đã được render
bằng *cùng một* texture và bộ phát hiện lề đường
đã học "đặc trưng" này rất nhanh.

Một điều tương tự đã xảy ra với Quân đội Mỹ
khi họ lần đầu thử phát hiện xe tăng trong rừng.
Họ chụp ảnh trên không của khu rừng không có xe tăng,
sau đó lái xe tăng vào rừng
và chụp một bộ ảnh khác.
Classifier dường như hoạt động *hoàn hảo*.
Không may, nó chỉ học được
cách phân biệt cây có bóng
với cây không có bóng---bộ ảnh đầu tiên
được chụp vào sáng sớm,
bộ ảnh thứ hai được chụp vào buổi trưa.

### Phân phối không dừng

Một tình huống tinh tế hơn nhiều nảy sinh
khi phân phối thay đổi chậm
(còn được gọi là *phân phối không dừng*)
và mô hình không được cập nhật đầy đủ.
Dưới đây là một số trường hợp điển hình.

* Chúng ta huấn luyện một mô hình quảng cáo tính toán rồi không cập nhật nó thường xuyên (ví dụ: chúng ta quên bổ sung thông tin rằng một thiết bị mới ít người biết đến có tên iPad vừa được ra mắt).
* Chúng ta xây dựng một bộ lọc spam. Nó hoạt động tốt trong việc phát hiện mọi spam mà chúng ta từng thấy cho đến nay. Nhưng rồi những kẻ gửi spam trở nên khôn ngoan hơn và tạo ra các thư mới không giống bất cứ thứ gì chúng ta từng thấy trước đây.
* Chúng ta xây dựng một hệ thống gợi ý sản phẩm. Nó hoạt động suốt mùa đông nhưng sau Giáng sinh rất lâu vẫn tiếp tục gợi ý mũ ông già Noel.

### Thêm các giai thoại

* Chúng ta xây dựng một bộ phát hiện khuôn mặt. Nó hoạt động tốt trên mọi benchmark. Không may, nó thất bại trên dữ liệu kiểm tra---các ví dụ gây lỗi là ảnh cận cảnh mà khuôn mặt chiếm toàn bộ ảnh (không có dữ liệu như vậy trong tập huấn luyện).
* Chúng ta xây dựng một web search engine cho thị trường Mỹ và muốn deploy nó tại Vương quốc Anh.
* Chúng ta huấn luyện một image classifier bằng cách tổng hợp một dataset lớn, trong đó mỗi lớp trong một tập lớn các lớp được biểu diễn đồng đều trong dataset, chẳng hạn 1000 danh mục, mỗi danh mục được biểu diễn bởi 1000 ảnh. Sau đó, chúng ta deploy hệ thống trong thế giới thực, nơi phân phối nhãn thực tế của ảnh rõ ràng là không đồng đều.






## Hiệu chỉnh Distribution Shift

Như chúng ta đã thảo luận, có nhiều trường hợp
mà các phân phối huấn luyện và kiểm tra
$P(\mathbf{x}, y)$ khác nhau.
Trong một số trường hợp, chúng ta may mắn và các mô hình vẫn hoạt động
bất chấp covariate shift, label shift hoặc concept shift.
Trong những trường hợp khác, chúng ta có thể làm tốt hơn bằng cách áp dụng
các chiến lược có cơ sở để ứng phó với shift.
Phần còn lại của mục này mang tính kỹ thuật hơn đáng kể.
Người đọc thiếu kiên nhẫn có thể tiếp tục sang mục tiếp theo
vì nội dung này không phải điều kiện tiên quyết cho các khái niệm sau đó.

### Rủi ro thực nghiệm và Rủi ro
:label:`subsec_empirical-risk-and-risk`

Trước tiên, hãy xem xét chính xác
điều gì diễn ra trong quá trình huấn luyện mô hình:
chúng ta lặp qua các đặc trưng và nhãn liên kết
của dữ liệu huấn luyện
$\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$
và cập nhật các tham số của mô hình $f$ sau mỗi minibatch.
Để đơn giản, chúng ta không xét regularization,
vì vậy về cơ bản chúng ta tối thiểu hóa loss trên dữ liệu huấn luyện:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

trong đó $l$ là loss function
đo lường dự đoán $f(\mathbf{x}_i)$ "tệ đến mức nào" với nhãn liên kết $y_i$.
Các nhà thống kê gọi số hạng trong :eqref:`eq_empirical-risk-min` là *rủi ro thực nghiệm*.
*Rủi ro thực nghiệm* là loss trung bình trên dữ liệu huấn luyện
để xấp xỉ *rủi ro*,
tức là
kỳ vọng của loss trên toàn bộ quần thể dữ liệu được lấy từ phân phối thực của chúng
$p(\mathbf{x},y)$:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

Tuy nhiên, trên thực tế, chúng ta thường không thể thu được toàn bộ quần thể dữ liệu.
Vì vậy, *tối thiểu hóa rủi ro thực nghiệm*,
tức tối thiểu hóa rủi ro thực nghiệm trong :eqref:`eq_empirical-risk-min`,
là một chiến lược thực tiễn cho machine learning,
với hy vọng xấp xỉ
tối thiểu hóa rủi ro.



### Hiệu chỉnh Covariate Shift
:label:`subsec_covariate-shift-correction`

Giả sử chúng ta muốn ước lượng
một quan hệ phụ thuộc $P(y \mid \mathbf{x})$
mà chúng ta có dữ liệu đã gán nhãn $(\mathbf{x}_i, y_i)$.
Không may, các quan sát $\mathbf{x}_i$ được lấy
từ một *phân phối nguồn* $q(\mathbf{x})$
thay vì *phân phối đích* $p(\mathbf{x})$.
May mắn thay,
giả định về quan hệ phụ thuộc có nghĩa
là phân phối có điều kiện không thay đổi: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$.
Nếu phân phối nguồn $q(\mathbf{x})$ là "sai",
chúng ta có thể hiệu chỉnh điều đó bằng cách sử dụng đẳng thức đơn giản sau trong rủi ro:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Nói cách khác, chúng ta cần gán lại trọng số cho mỗi ví dụ dữ liệu
theo tỷ lệ giữa
xác suất
mà nó được lấy từ phân phối đúng và xác suất từ phân phối sai:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Thay trọng số $\beta_i$ vào
mỗi ví dụ dữ liệu $(\mathbf{x}_i, y_i)$,
chúng ta có thể huấn luyện mô hình bằng
*tối thiểu hóa rủi ro thực nghiệm có trọng số*:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`



Than ôi, chúng ta không biết tỷ lệ đó,
vì vậy trước khi có thể làm bất cứ điều gì hữu ích, chúng ta cần ước lượng nó.
Có nhiều phương pháp khả dụng,
bao gồm một số phương pháp lý thuyết toán tử tinh vi
cố gắng hiệu chỉnh lại trực tiếp toán tử kỳ vọng
bằng nguyên lý chuẩn tối thiểu hoặc entropy cực đại.
Lưu ý rằng với bất kỳ phương pháp nào như vậy, chúng ta cần các mẫu
được lấy từ cả hai phân phối---$p$ "thực", ví dụ
thông qua quyền truy cập vào dữ liệu kiểm tra, và phân phối
được dùng để tạo tập huấn luyện $q$ (phân phối sau hiển nhiên là có sẵn).
Tuy nhiên, lưu ý rằng chúng ta chỉ cần các đặc trưng $\mathbf{x} \sim p(\mathbf{x})$;
chúng ta không cần truy cập các nhãn $y \sim p(y)$.

Trong trường hợp này, có một phương pháp rất hiệu quả
cho kết quả gần tốt như phương pháp ban đầu: đó là logistic regression,
một trường hợp đặc biệt của softmax regression (xem :numref:`sec_softmax`)
cho phân loại nhị phân.
Đây là tất cả những gì cần để tính các tỷ lệ xác suất ước lượng.
Chúng ta học một classifier để phân biệt
giữa dữ liệu được lấy từ $p(\mathbf{x})$
và dữ liệu được lấy từ $q(\mathbf{x})$.
Nếu không thể phân biệt
giữa hai phân phối,
thì điều đó có nghĩa là các instance liên quan
có xác suất như nhau đến từ
một trong hai phân phối đó.
Mặt khác, mọi instance
có thể được phân biệt tốt
nên được tăng hoặc giảm trọng số
đáng kể tương ứng.

Để đơn giản, giả sử chúng ta có
số lượng instance bằng nhau từ hai phân phối
$p(\mathbf{x})$
và $q(\mathbf{x})$, tương ứng.
Bây giờ, ký hiệu $z$ là các nhãn bằng $1$
cho dữ liệu được lấy từ $p$ và $-1$ cho dữ liệu được lấy từ $q$.
Khi đó, xác suất trong một dataset trộn được cho bởi

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \textrm{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Do đó, nếu chúng ta sử dụng phương pháp logistic regression,
trong đó $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ là một hàm được tham số hóa),
thì suy ra rằng

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

Kết quả là, chúng ta cần giải quyết hai vấn đề:
thứ nhất, phân biệt giữa
dữ liệu được lấy từ cả hai phân phối,
sau đó là bài toán tối thiểu hóa rủi ro thực nghiệm có trọng số
trong :eqref:`eq_weighted-empirical-risk-min`
mà chúng ta gán trọng số cho các số hạng bằng $\beta_i$.

Bây giờ chúng ta đã sẵn sàng mô tả một thuật toán hiệu chỉnh.
Giả sử rằng chúng ta có một tập huấn luyện $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ và một tập kiểm tra chưa gán nhãn $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$.
Đối với dịch chuyển biến đồng biến,
chúng ta giả định rằng $\mathbf{x}_i$ với mọi $1 \leq i \leq n$ được lấy mẫu từ một phân phối nguồn nào đó
và $\mathbf{u}_i$ với mọi $1 \leq i \leq m$
được lấy mẫu từ phân phối đích.
Đây là một thuật toán điển hình
để hiệu chỉnh dịch chuyển biến đồng biến:

1. Tạo một tập huấn luyện phân loại nhị phân: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Huấn luyện một bộ phân loại nhị phân bằng logistic regression để thu được hàm $h$.
1. Gán trọng số cho dữ liệu huấn luyện bằng $\beta_i = \exp(h(\mathbf{x}_i))$ hoặc tốt hơn là $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ với một hằng số $c$ nào đó.
1. Sử dụng các trọng số $\beta_i$ để huấn luyện trên $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ trong :eqref:`eq_weighted-empirical-risk-min`.

Lưu ý rằng thuật toán trên dựa vào một giả định then chốt.
Để phương pháp này hoạt động, chúng ta cần mọi ví dụ dữ liệu
trong phân phối đích (ví dụ, tại thời điểm kiểm tra)
đều có xác suất khác không xuất hiện tại thời điểm huấn luyện.
Nếu chúng ta tìm thấy một điểm mà $p(\mathbf{x}) > 0$ nhưng $q(\mathbf{x}) = 0$,
thì trọng số importance tương ứng phải là vô hạn.






### Hiệu chỉnh dịch chuyển nhãn

Giả sử rằng chúng ta đang xử lý một
bài toán phân loại với $k$ danh mục.
Sử dụng cùng ký hiệu trong :numref:`subsec_covariate-shift-correction`,
$q$ và $p$ lần lượt là phân phối nguồn (ví dụ, thời điểm huấn luyện) và phân phối đích (ví dụ, thời điểm kiểm tra).
Giả sử rằng phân phối của các nhãn thay đổi theo thời gian:
$q(y) \neq p(y)$, nhưng phân phối có điều kiện theo lớp
vẫn giữ nguyên: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$.
Nếu phân phối nguồn $q(y)$ là "không đúng",
chúng ta có thể hiệu chỉnh điều đó
theo
đồng nhất thức sau trong rủi ro
như được định nghĩa trong
:eqref:`eq_true-risk`:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$



Ở đây, các trọng số importance của chúng ta sẽ tương ứng với
các tỷ số likelihood của nhãn:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

Một điểm hay của dịch chuyển nhãn là
nếu chúng ta có một mô hình đủ tốt
trên phân phối nguồn,
thì chúng ta có thể có được các ước lượng nhất quán của những trọng số này
mà không bao giờ phải xử lý chiều không gian dữ liệu.
Trong deep learning, đầu vào thường
là các đối tượng có số chiều cao như hình ảnh,
trong khi nhãn thường là các đối tượng đơn giản hơn như danh mục.

Để ước lượng phân phối nhãn đích,
trước tiên chúng ta lấy bộ phân loại có sẵn đủ tốt của mình
(thường được huấn luyện trên dữ liệu huấn luyện)
và tính ma trận "nhầm lẫn" của nó bằng tập validation
(cũng thuộc phân phối huấn luyện).
*Ma trận nhầm lẫn*, $\mathbf{C}$, đơn giản là một ma trận $k \times k$,
trong đó mỗi cột tương ứng với danh mục nhãn (nhãn thực)
và mỗi hàng tương ứng với danh mục do mô hình của chúng ta dự đoán.
Giá trị của mỗi ô $c_{ij}$ là tỷ lệ trên tổng số dự đoán trong tập validation
mà nhãn thực là $j$ và mô hình của chúng ta dự đoán là $i$.

Tuy nhiên, chúng ta không thể tính ma trận nhầm lẫn
trực tiếp trên dữ liệu đích
vì chúng ta không được xem các nhãn của những ví dụ
mà mình gặp trong thực tế,
trừ khi đầu tư vào một pipeline gán nhãn thời gian thực phức tạp.
Dù vậy, điều chúng ta có thể làm là lấy trung bình tất cả các dự đoán của mô hình
tại thời điểm kiểm tra, thu được đầu ra trung bình của mô hình $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$,
trong đó phần tử thứ $i^\textrm{th}$ $\mu(\hat{y}_i)$
là tỷ lệ trên tổng số dự đoán trong tập kiểm tra
mà mô hình của chúng ta dự đoán là $i$.

Hóa ra rằng dưới một số điều kiện tương đối nhẹ---nếu
bộ phân loại của chúng ta vốn đã có độ chính xác hợp lý,
và nếu dữ liệu đích chỉ chứa các danh mục
mà chúng ta đã từng thấy trước đó,
và nếu giả định dịch chuyển nhãn ngay từ đầu là đúng
(đây là giả định mạnh nhất)---chúng ta có thể ước lượng phân phối nhãn của tập kiểm tra
bằng cách giải một hệ tuyến tính đơn giản

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

vì ước lượng $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ đúng với mọi $1 \leq i \leq k$,
trong đó $p(y_j)$ là phần tử thứ $j^\textrm{th}$ của vector phân phối nhãn $k$ chiều $p(\mathbf{y})$.
Nếu ngay từ đầu bộ phân loại của chúng ta đủ chính xác,
thì ma trận nhầm lẫn $\mathbf{C}$ sẽ khả nghịch,
và chúng ta có nghiệm $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.

Vì chúng ta quan sát được các nhãn trong dữ liệu nguồn,
việc ước lượng phân phối $q(y)$ là dễ dàng.
Sau đó, với bất kỳ ví dụ huấn luyện $i$ nào có nhãn $y_i$,
chúng ta có thể lấy tỷ số $p(y_i)/q(y_i)$ đã ước lượng
để tính trọng số $\beta_i$,
và đưa giá trị này vào quá trình tối thiểu hóa rủi ro thực nghiệm có trọng số
trong :eqref:`eq_weighted-empirical-risk-min`.


### Hiệu chỉnh dịch chuyển khái niệm

Dịch chuyển khái niệm khó khắc phục một cách có nguyên tắc hơn nhiều.
Chẳng hạn, trong tình huống mà bài toán đột ngột thay đổi
từ phân biệt mèo với chó sang
phân biệt động vật trắng với đen,
sẽ không hợp lý khi giả định
rằng chúng ta có thể làm tốt hơn nhiều so với chỉ thu thập nhãn mới
và huấn luyện lại từ đầu.
May mắn thay, trong thực tế, những dịch chuyển cực đoan như vậy rất hiếm.
Thay vào đó, điều thường xảy ra là nhiệm vụ tiếp tục thay đổi chậm rãi.
Để làm rõ hơn, đây là một số ví dụ:

* Trong quảng cáo tính toán, các sản phẩm mới được ra mắt,
các sản phẩm cũ trở nên ít phổ biến hơn. Điều này có nghĩa là phân phối của quảng cáo và mức độ phổ biến của chúng thay đổi dần dần, và mọi bộ dự đoán tỷ lệ nhấp chuột đều cần thay đổi dần theo nó.
* Thấu kính camera giao thông xuống cấp dần do hao mòn từ môi trường, làm ảnh hưởng đến chất lượng hình ảnh theo thời gian.
* Nội dung tin tức thay đổi dần dần (tức là phần lớn tin tức vẫn không thay đổi nhưng các câu chuyện mới xuất hiện).

Trong những trường hợp như vậy, chúng ta có thể sử dụng cùng cách tiếp cận đã dùng khi huấn luyện network để làm cho chúng thích nghi với thay đổi trong dữ liệu. Nói cách khác, chúng ta sử dụng các trọng số network hiện có và chỉ thực hiện vài bước cập nhật với dữ liệu mới thay vì huấn luyện lại từ đầu.


## Phân loại các bài toán học máy

Với kiến thức về cách xử lý các thay đổi trong phân phối, giờ đây chúng ta có thể xem xét một số khía cạnh khác của việc xây dựng bài toán học máy.


### Học theo batch

Trong *học theo batch*, chúng ta có quyền truy cập vào các đặc trưng và nhãn huấn luyện $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, được sử dụng để huấn luyện một mô hình $f(\mathbf{x})$. Sau đó, chúng ta deploy mô hình này để đánh giá dữ liệu mới $(\mathbf{x}, y)$ được lấy mẫu từ cùng một phân phối. Đây là giả định mặc định cho bất kỳ bài toán nào mà chúng ta thảo luận ở đây. Chẳng hạn, chúng ta có thể huấn luyện một bộ phát hiện mèo dựa trên rất nhiều hình ảnh mèo và chó. Sau khi huấn luyện xong, chúng ta đưa nó vào như một phần của hệ thống computer vision cửa cho mèo thông minh chỉ cho phép mèo đi vào. Sau đó, hệ thống này được lắp đặt tại nhà khách hàng và không bao giờ được cập nhật lại (trừ những trường hợp cực đoan).


### Học trực tuyến

Bây giờ hãy tưởng tượng rằng dữ liệu $(\mathbf{x}_i, y_i)$ đến từng mẫu một. Cụ thể hơn, giả sử rằng trước tiên chúng ta quan sát $\mathbf{x}_i$, sau đó cần đưa ra một ước lượng $f(\mathbf{x}_i)$. Chỉ sau khi làm điều này, chúng ta mới quan sát $y_i$ và do đó nhận được phần thưởng hoặc chịu tổn thất, tùy theo quyết định của mình.
Nhiều bài toán thực tế thuộc loại này. Ví dụ, chúng ta cần dự đoán giá cổ phiếu vào ngày mai, điều này cho phép chúng ta giao dịch dựa trên ước lượng đó, và vào cuối ngày chúng ta biết được liệu ước lượng của mình có mang lại lợi nhuận hay không. Nói cách khác, trong *học trực tuyến*, chúng ta có chu trình sau, trong đó liên tục cải thiện mô hình dựa trên các quan sát mới:

$$\begin{aligned}&\textrm{model } f_t \longrightarrow \textrm{data }  \mathbf{x}_t \longrightarrow \textrm{estimate } f_t(\mathbf{x}_t) \longrightarrow\\ \textrm{obs}&\textrm{ervation } y_t \longrightarrow \textrm{loss } l(y_t, f_t(\mathbf{x}_t)) \longrightarrow \textrm{model } f_{t+1}\end{aligned}$$

### Bandit

*Bandit* là một trường hợp đặc biệt của bài toán ở trên. Trong khi ở hầu hết các bài toán học máy, chúng ta có một hàm $f$ được tham số hóa liên tục mà các tham số của nó cần được học (ví dụ, một deep network), thì trong một bài toán *bandit*, chúng ta chỉ có một số hữu hạn các cần gạt (arm) có thể chọn, tức là một số hữu hạn hành động có thể thực hiện. Không quá ngạc nhiên khi với bài toán đơn giản hơn này, ta có thể đạt được những đảm bảo lý thuyết mạnh hơn về tính tối ưu. Chúng tôi chủ yếu liệt kê nó vì bài toán này thường bị xem (một cách gây nhầm lẫn) như thể nó là một bối cảnh học máy riêng biệt.


### Điều khiển

Trong nhiều trường hợp, môi trường ghi nhớ những gì chúng ta đã làm. Không nhất thiết theo cách đối kháng, nhưng nó vẫn sẽ ghi nhớ và phản hồi sẽ phụ thuộc vào những gì đã xảy ra trước đó. Chẳng hạn, một bộ điều khiển nồi đun cà phê sẽ quan sát các nhiệt độ khác nhau tùy theo việc trước đó nó có đang làm nóng nồi hay không. Các thuật toán điều khiển PID (proportional-integral-derivative) là một lựa chọn phổ biến trong trường hợp này.
Tương tự, hành vi của người dùng trên một trang tin tức sẽ phụ thuộc vào những gì chúng ta đã hiển thị cho họ trước đó (ví dụ, họ thường chỉ đọc hầu hết tin tức một lần). Nhiều thuật toán như vậy xây dựng một mô hình về môi trường mà chúng hoạt động để khiến các quyết định của chúng có vẻ ít ngẫu nhiên hơn.
Gần đây,
lý thuyết điều khiển (ví dụ, các biến thể PID) cũng đã được sử dụng
để tự động điều chỉnh các hyperparameter
nhằm đạt được chất lượng tách biệt biểu diễn và tái tạo tốt hơn,
cũng như cải thiện tính đa dạng của văn bản được sinh ra và chất lượng tái tạo của hình ảnh được sinh ra :cite:`Shao.Yao.Sun.ea.2020`.




### Reinforcement Learning

Trong trường hợp tổng quát hơn về một môi trường có bộ nhớ, chúng ta có thể gặp các tình huống mà môi trường cố gắng hợp tác với chúng ta (các trò chơi hợp tác, đặc biệt là đối với trò chơi không có tổng bằng không), hoặc những tình huống khác mà môi trường sẽ cố gắng chiến thắng. Cờ vua, Go, Backgammon hoặc StarCraft là một số trường hợp trong *reinforcement learning*. Tương tự, chúng ta có thể muốn xây dựng một bộ điều khiển tốt cho xe tự hành. Những xe khác có khả năng sẽ phản hồi theo những cách không tầm thường trước phong cách lái xe của xe tự hành, ví dụ như cố gắng tránh nó, cố gắng gây tai nạn, hoặc cố gắng hợp tác với nó.

### Xem xét Môi trường

Một điểm khác biệt then chốt giữa các tình huống khác nhau ở trên là một chiến lược có thể luôn hiệu quả trong trường hợp môi trường tĩnh lại có thể không luôn hiệu quả trong một môi trường có thể thích nghi. Chẳng hạn, một cơ hội arbitrage được nhà giao dịch phát hiện có khả năng sẽ biến mất khi nó được khai thác. Tốc độ và cách thức mà môi trường thay đổi quyết định phần lớn loại thuật toán mà chúng ta có thể áp dụng. Ví dụ, nếu biết rằng mọi thứ chỉ có thể thay đổi chậm, chúng ta cũng có thể buộc mọi ước lượng chỉ thay đổi chậm. Nếu biết rằng môi trường có thể thay đổi tức thời nhưng chỉ rất hiếm khi, chúng ta có thể dự liệu điều đó. Những loại kiến thức này rất quan trọng đối với một data scientist đang phát triển kỹ năng khi xử lý concept shift, tức là khi bài toán đang được giải có thể thay đổi theo thời gian.




## Tính Công bằng, Trách nhiệm giải trình và Tính Minh bạch trong Machine Learning

Cuối cùng, điều quan trọng cần nhớ
là khi bạn deploy các hệ thống machine learning,
bạn không chỉ đơn thuần tối ưu hóa một mô hình dự đoán---bạn
thường đang cung cấp một công cụ sẽ
được sử dụng để tự động hóa (một phần hoặc hoàn toàn) các quyết định.
Các hệ thống kỹ thuật này có thể tác động đến cuộc sống
của những cá nhân chịu ảnh hưởng bởi các quyết định tạo ra.
Bước chuyển từ việc xem xét dự đoán sang đưa ra quyết định
không chỉ đặt ra các câu hỏi kỹ thuật mới,
mà còn đặt ra hàng loạt câu hỏi đạo đức
cần được cân nhắc cẩn thận.
Nếu chúng ta deploy một hệ thống chẩn đoán y tế,
chúng ta cần biết hệ thống đó có thể hoạt động với những nhóm dân số nào
và có thể không hoạt động với những nhóm nào.
Việc bỏ qua các rủi ro có thể lường trước đối với phúc lợi của
một nhóm dân số con có thể khiến chúng ta cung cấp dịch vụ chăm sóc kém hơn.
Hơn nữa, khi xem xét các hệ thống ra quyết định,
chúng ta phải lùi lại và xem xét lại cách đánh giá công nghệ của mình.
Trong số các hệ quả khác của sự thay đổi phạm vi này,
chúng ta sẽ nhận thấy rằng *accuracy* hiếm khi là thước đo phù hợp.
Chẳng hạn, khi chuyển các dự đoán thành hành động,
chúng ta thường sẽ muốn tính đến
chi phí tiềm ẩn khác nhau của các kiểu mắc lỗi.
Nếu một cách phân loại sai hình ảnh
có thể bị xem là một sự xúc phạm ngầm về chủng tộc,
trong khi phân loại sai sang một hạng mục khác
sẽ vô hại, thì chúng ta có thể muốn điều chỉnh
các ngưỡng của mình cho phù hợp, có xét đến các giá trị xã hội
khi thiết kế giao thức ra quyết định.
Chúng ta cũng cần cẩn thận về
cách các hệ thống dự đoán có thể dẫn đến các feedback loop.
Ví dụ, hãy xem xét các hệ thống cảnh sát dự đoán,
vốn phân bổ các sĩ quan tuần tra
đến những khu vực có dự báo tội phạm cao.
Có thể dễ dàng thấy một mô hình đáng lo ngại có thể xuất hiện như thế nào:

 1. Các khu dân cư có nhiều tội phạm hơn sẽ có nhiều tuần tra hơn.
 1. Hệ quả là, nhiều tội phạm hơn được phát hiện tại các khu dân cư này, đi vào dữ liệu huấn luyện có sẵn cho các lần lặp sau.
 1. Khi tiếp xúc với nhiều mẫu dương tính hơn, mô hình dự đoán còn nhiều tội phạm hơn tại các khu dân cư này.
 1. Trong lần lặp tiếp theo, mô hình đã cập nhật nhắm đến cùng khu dân cư đó với mức độ còn cao hơn, dẫn đến việc phát hiện thêm nhiều tội phạm hơn, v.v.

Thông thường, các cơ chế khác nhau khiến
dự đoán của một mô hình gắn kết với dữ liệu huấn luyện của nó
không được tính đến trong quá trình mô hình hóa.
Điều này có thể dẫn đến cái mà các nhà nghiên cứu gọi là *runaway feedback loops*.
Ngoài ra, chúng ta cần cẩn thận về
liệu ngay từ đầu chúng ta có đang giải quyết đúng vấn đề hay không.
Các thuật toán dự đoán hiện nay đóng một vai trò quá lớn
trong việc chi phối sự phổ biến thông tin.
Liệu tin tức mà một cá nhân tiếp cận
có nên được quyết định bởi tập hợp các trang Facebook mà họ đã *Liked*?
Đây chỉ là một vài trong số rất nhiều tình huống nan giải về đạo đức cấp bách
mà bạn có thể gặp phải trong sự nghiệp machine learning.


## Tóm tắt

Trong nhiều trường hợp, tập huấn luyện và tập kiểm tra không đến từ cùng một phân phối. Điều này được gọi là distribution shift.
Risk là kỳ vọng của loss trên toàn bộ quần thể dữ liệu được lấy từ phân phối thực của chúng. Tuy nhiên, toàn bộ quần thể này thường không khả dụng. Empirical risk là loss trung bình trên dữ liệu huấn luyện để xấp xỉ risk. Trong thực tế, chúng ta thực hiện empirical risk minimization.

Với các giả định tương ứng, covariate shift và label shift có thể được phát hiện và hiệu chỉnh tại thời điểm kiểm tra. Việc không tính đến độ chệch này có thể trở thành vấn đề tại thời điểm kiểm tra.
Trong một số trường hợp, môi trường có thể ghi nhớ các hành động tự động và phản hồi theo những cách đáng ngạc nhiên. Chúng ta phải tính đến khả năng này khi xây dựng mô hình và tiếp tục giám sát các hệ thống đang vận hành, sẵn sàng với khả năng các mô hình và môi trường của chúng ta sẽ đan xen với nhau theo những cách không lường trước.

## Bài tập

1. Điều gì có thể xảy ra khi chúng ta thay đổi hành vi của một search engine? Người dùng có thể làm gì? Còn các nhà quảng cáo thì sao?
1. Hãy triển khai một bộ phát hiện covariate shift. Gợi ý: xây dựng một classifier.
1. Hãy triển khai một bộ hiệu chỉnh covariate shift.
1. Ngoài distribution shift, điều gì khác có thể ảnh hưởng đến cách empirical risk xấp xỉ risk?


[Thảo luận](https://discuss.d2l.ai/t/105)