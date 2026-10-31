# Khái quát hóa trong Học sâu


Trong :numref:`chap_regression` và :numref:`chap_classification`,
chúng ta đã giải quyết các bài toán hồi quy và phân loại
bằng cách khớp các mô hình tuyến tính với dữ liệu huấn luyện.
Trong cả hai trường hợp, chúng ta đã cung cấp các thuật toán thực tiễn
để tìm các tham số tối đa hóa
khả năng xảy ra của các nhãn huấn luyện quan sát được.
Và sau đó, ở gần cuối mỗi chương,
chúng ta đã nhắc lại rằng việc khớp dữ liệu huấn luyện
chỉ là một mục tiêu trung gian.
Mục tiêu thực sự của chúng ta từ đầu đến cuối là khám phá các *mẫu tổng quát*
làm cơ sở để đưa ra các dự đoán chính xác
ngay cả trên các ví dụ mới được rút ra từ cùng một quần thể nền tảng.
Các nhà nghiên cứu machine learning là *người sử dụng* các thuật toán tối ưu hóa.
Đôi khi, chúng ta thậm chí phải phát triển các thuật toán tối ưu hóa mới.
Nhưng xét cho cùng, tối ưu hóa chỉ là phương tiện để đạt được mục đích.
Về cốt lõi, machine learning là một lĩnh vực thống kê
và chúng ta chỉ muốn tối ưu hóa loss huấn luyện trong chừng mực
một nguyên lý thống kê nào đó (đã biết hoặc chưa biết)
dẫn các mô hình thu được đến khả năng khái quát hóa vượt ra ngoài tập huấn luyện.


Điểm tích cực là, các mạng nơ-ron sâu
được huấn luyện bằng stochastic gradient descent khái quát hóa tốt một cách đáng kinh ngạc
trên vô số bài toán dự đoán, bao gồm thị giác máy tính;
xử lý ngôn ngữ tự nhiên; dữ liệu chuỗi thời gian; hệ thống gợi ý;
hồ sơ sức khỏe điện tử; gấp cuộn protein;
xấp xỉ hàm giá trị trong trò chơi điện tử
và trò chơi bàn cờ; cùng nhiều lĩnh vực khác.
Điểm tiêu cực là, nếu bạn đang tìm kiếm
một lời giải thích đơn giản
cho câu chuyện tối ưu hóa
(tại sao chúng ta có thể khớp chúng với dữ liệu huấn luyện)
hoặc câu chuyện khái quát hóa
(tại sao các mô hình thu được lại khái quát hóa đến các ví dụ chưa từng thấy),
thì có lẽ bạn nên rót cho mình một ly.
Trong khi các quy trình tối ưu hóa mô hình tuyến tính của chúng ta
và các tính chất thống kê của nghiệm
đều được mô tả tốt bởi một hệ thống lý thuyết toàn diện,
hiểu biết của chúng ta về học sâu
vẫn giống miền viễn Tây ở cả hai phương diện.

Cả lý thuyết lẫn thực hành của học sâu
đều đang phát triển nhanh chóng,
với các nhà lý thuyết áp dụng những chiến lược mới
để giải thích điều đang diễn ra,
trong khi các học viên vẫn tiếp tục
đổi mới với tốc độ chóng mặt,
xây dựng kho vũ khí các heuristic để huấn luyện mạng sâu
cùng một hệ thống trực giác và tri thức kinh nghiệm
giúp định hướng cho việc quyết định
kỹ thuật nào nên được áp dụng trong tình huống nào.

Tóm tắt tình hình hiện tại là lý thuyết học sâu
đã đưa ra các hướng tiếp cận đầy hứa hẹn và những kết quả thú vị rải rác,
nhưng dường như vẫn còn rất xa mới có được lời giải thích toàn diện
cho cả (i) lý do chúng ta có thể tối ưu hóa mạng nơ-ron
và (ii) cách các mô hình được học bằng gradient descent
có thể khái quát hóa tốt đến vậy, ngay cả trên các tác vụ nhiều chiều.
Tuy nhiên, trong thực tế, (i) hiếm khi là vấn đề
(chúng ta luôn có thể tìm được các tham số khớp với toàn bộ dữ liệu huấn luyện)
và do đó việc hiểu khả năng khái quát hóa mới là vấn đề lớn hơn nhiều.
Mặt khác, ngay cả khi không có sự an tâm từ một lý thuyết khoa học mạch lạc,
các học viên đã phát triển một bộ sưu tập lớn các kỹ thuật
có thể giúp bạn tạo ra những mô hình khái quát hóa tốt trong thực tế.
Mặc dù không một bản tóm tắt ngắn gọn nào có thể làm trọn vẹn công bằng
cho chủ đề rộng lớn về khái quát hóa trong học sâu,
và mặc dù trạng thái nghiên cứu tổng thể còn lâu mới được giải quyết,
chúng tôi hy vọng trong phần này sẽ trình bày một cái nhìn tổng quan
về trạng thái của nghiên cứu và thực hành.


## Xem lại Overfitting và Regularization

Theo định lý "no free lunch" của :citet:`wolpert1995no`,
bất kỳ thuật toán học nào cũng khái quát hóa tốt hơn trên dữ liệu có một số phân phối nhất định, và kém hơn với các phân phối khác.
Do đó, với một tập huấn luyện hữu hạn,
một mô hình dựa vào các giả định nhất định:
để đạt được hiệu năng ở mức con người
có thể hữu ích khi xác định các *inductive bias*
phản ánh cách con người suy nghĩ về thế giới.
Các inductive bias như vậy thể hiện sự ưu tiên
đối với các nghiệm có những đặc tính nhất định.
Ví dụ,
một MLP sâu có inductive bias
hướng đến việc xây dựng một hàm phức tạp bằng cách hợp thành các hàm đơn giản hơn.

Với các mô hình machine learning mã hóa inductive bias,
cách tiếp cận huấn luyện chúng ta
thường gồm hai giai đoạn: (i) khớp dữ liệu huấn luyện;
và (ii) ước lượng *lỗi khái quát hóa*
(lỗi thực trên quần thể nền tảng)
bằng cách đánh giá mô hình trên dữ liệu holdout.
Sự khác biệt giữa mức độ khớp trên dữ liệu huấn luyện
và mức độ khớp trên dữ liệu kiểm tra được gọi là *khoảng cách khái quát hóa*, và khi khoảng cách này lớn,
chúng ta nói rằng các mô hình *overfit* với dữ liệu huấn luyện.
Trong các trường hợp overfitting cực đoan,
chúng ta có thể khớp chính xác dữ liệu huấn luyện,
ngay cả khi lỗi kiểm tra vẫn đáng kể.
Và theo quan điểm cổ điển,
cách diễn giải là các mô hình của chúng ta quá phức tạp,
đòi hỏi chúng ta phải giảm số lượng đặc trưng,
số lượng tham số khác 0 được học,
hoặc độ lớn của các tham số theo một thước đo định lượng.
Hãy nhớ lại biểu đồ về độ phức tạp mô hình so với loss
(:numref:`fig_capacity_vs_error`)
trong :numref:`sec_generalization_basics`.


Tuy nhiên, học sâu làm phức tạp bức tranh này theo những cách phản trực giác.
Trước hết, đối với các bài toán phân loại,
các mô hình của chúng ta thường đủ khả năng biểu đạt
để khớp hoàn hảo mọi ví dụ huấn luyện,
ngay cả trong các bộ dữ liệu gồm hàng triệu mẫu
:cite:`zhang2021understanding`.
Theo bức tranh cổ điển, chúng ta có thể cho rằng
bối cảnh này nằm ở cực phải
của trục độ phức tạp mô hình,
và mọi cải thiện về lỗi khái quát hóa
đều phải đến từ regularization,
hoặc bằng cách giảm độ phức tạp của lớp mô hình,
hoặc bằng cách áp dụng một hình phạt, hạn chế nghiêm ngặt
tập giá trị mà các tham số có thể nhận.
Nhưng đó là lúc mọi thứ bắt đầu trở nên kỳ lạ.

Điều lạ lùng là, đối với nhiều tác vụ học sâu
(ví dụ: nhận dạng ảnh và phân loại văn bản)
chúng ta thường lựa chọn giữa các kiến trúc mô hình,
mà tất cả đều có thể đạt loss huấn luyện thấp tùy ý
(và lỗi huấn luyện bằng 0).
Bởi vì mọi mô hình đang được xem xét đều đạt lỗi huấn luyện bằng 0,
*lựa chọn duy nhất để tiếp tục cải thiện là giảm overfitting*.
Kỳ lạ hơn nữa, thường xảy ra trường hợp
mặc dù khớp hoàn hảo dữ liệu huấn luyện,
chúng ta thực sự có thể *giảm lỗi khái quát hóa*
thêm nữa bằng cách làm mô hình *có khả năng biểu đạt cao hơn nữa*,
ví dụ, thêm layer, node hoặc huấn luyện
trong số epoch lớn hơn.
Kỳ lạ hơn nữa, mô hình liên hệ khoảng cách khái quát hóa
với *độ phức tạp* của mô hình (chẳng hạn được thể hiện qua độ sâu hoặc chiều rộng của mạng)
có thể không đơn điệu,
trong đó độ phức tạp lớn hơn ban đầu gây hại
nhưng sau đó lại có ích theo mô hình được gọi là "double-descent"
:cite:`nakkiran2021deep`.
Vì vậy, người thực hành học sâu sở hữu một túi mẹo,
trong đó một số dường như hạn chế mô hình theo cách nào đó
và số khác dường như khiến mô hình có khả năng biểu đạt cao hơn nữa,
và tất cả, theo một nghĩa nào đó, đều được áp dụng để giảm thiểu overfitting.

Làm mọi thứ phức tạp hơn nữa,
mặc dù các đảm bảo do lý thuyết học cổ điển đưa ra
có thể bảo thủ ngay cả đối với các mô hình cổ điển,
chúng dường như bất lực trong việc giải thích tại sao
mạng nơ-ron sâu lại khái quát hóa ngay từ đầu.
Bởi vì mạng nơ-ron sâu có khả năng khớp
các nhãn tùy ý ngay cả với các bộ dữ liệu lớn,
và mặc dù sử dụng các phương pháp quen thuộc như regularization $\ell_2$,
các cận khái quát hóa dựa trên độ phức tạp truyền thống,
ví dụ các cận dựa trên VC dimension
hoặc độ phức tạp Rademacher của một lớp giả thuyết
không thể giải thích tại sao mạng nơ-ron khái quát hóa.

## Cảm hứng từ Nonparametrics

Khi tiếp cận học sâu lần đầu,
ta dễ bị cám dỗ xem chúng là các mô hình tham số.
Rốt cuộc, các mô hình *thực sự* có hàng triệu tham số.
Khi cập nhật các mô hình, chúng ta cập nhật các tham số của chúng.
Khi lưu các mô hình, chúng ta ghi các tham số của chúng vào đĩa.
Tuy nhiên, toán học và khoa học máy tính đầy rẫy
những thay đổi góc nhìn phản trực giác,
và các đẳng cấu đáng ngạc nhiên giữa những bài toán tưởng chừng khác biệt.
Mặc dù mạng nơ-ron rõ ràng *có* tham số,
theo một số cách, sẽ hiệu quả hơn
nếu xem chúng hoạt động giống các mô hình phi tham số.
Vậy chính xác điều gì khiến một mô hình là phi tham số?
Mặc dù tên gọi này bao gồm một tập hợp phương pháp đa dạng,
một chủ đề chung là các phương pháp phi tham số
có xu hướng có mức độ phức tạp tăng lên
khi lượng dữ liệu sẵn có tăng lên.

Có lẽ ví dụ đơn giản nhất về một mô hình phi tham số
là thuật toán $k$-láng giềng gần nhất (chúng ta sẽ trình bày thêm các mô hình phi tham số sau này, ví dụ trong :numref:`sec_attention-pooling`).
Ở đây, tại thời điểm huấn luyện,
người học chỉ đơn giản ghi nhớ bộ dữ liệu.
Sau đó, tại thời điểm dự đoán,
khi đối mặt với một điểm mới $\mathbf{x}$,
người học tìm $k$ láng giềng gần nhất
($k$ điểm $\mathbf{x}_i'$ tối thiểu hóa
một khoảng cách nào đó $d(\mathbf{x}, \mathbf{x}_i')$).
Khi $k=1$, thuật toán này được gọi là $1$-láng giềng gần nhất,
và thuật toán sẽ luôn đạt lỗi huấn luyện bằng 0.
Tuy nhiên, điều đó không có nghĩa là thuật toán sẽ không khái quát hóa.
Trên thực tế, hóa ra trong một số điều kiện nhẹ,
thuật toán 1-láng giềng gần nhất là nhất quán
(cuối cùng hội tụ về bộ dự đoán tối ưu).


Lưu ý rằng $1$-láng giềng gần nhất yêu cầu chúng ta chỉ định
một hàm khoảng cách $d$, hay tương đương,
chỉ định một hàm cơ sở có giá trị vector $\phi(\mathbf{x})$
để trích xuất đặc trưng cho dữ liệu.
Với bất kỳ lựa chọn nào của metric khoảng cách,
chúng ta sẽ đạt lỗi huấn luyện bằng 0
và cuối cùng đạt được một bộ dự đoán tối ưu,
nhưng các metric khoảng cách $d$ khác nhau
mã hóa các inductive bias khác nhau
và với lượng dữ liệu sẵn có hữu hạn
sẽ tạo ra các bộ dự đoán khác nhau.
Các lựa chọn khác nhau cho metric khoảng cách $d$
đại diện cho các giả định khác nhau về các mẫu nền tảng
và hiệu năng của các bộ dự đoán khác nhau
sẽ phụ thuộc vào mức độ tương thích của các giả định
với dữ liệu quan sát được.

Theo một nghĩa nào đó, vì các neural network được over-parametrized,
sở hữu số lượng parameter nhiều hơn rất nhiều so với mức cần thiết để khớp dữ liệu huấn luyện,
chúng có xu hướng *nội suy (interpolate)* dữ liệu huấn luyện (khớp hoàn hảo)
và do đó, ở một số khía cạnh, hoạt động giống các mô hình phi tham số hơn.
Các nghiên cứu lý thuyết gần đây đã thiết lập
mối liên hệ sâu sắc giữa các neural network lớn
và các phương pháp phi tham số, đáng chú ý là kernel methods.
Cụ thể, :citet:`Jacot.Grabriel.Hongler.2018`
đã chứng minh rằng trong giới hạn, khi các multilayer perceptron
với weight được khởi tạo ngẫu nhiên trở nên có độ rộng vô hạn,
chúng tương đương với các kernel methods (phi tham số)
đối với một lựa chọn cụ thể của kernel function
(về cơ bản là một distance function),
mà họ gọi là neural tangent kernel.
Mặc dù các mô hình neural tangent kernel hiện nay có thể chưa giải thích đầy đủ
hành vi của các deep network hiện đại,
thành công của chúng với vai trò là một công cụ phân tích
nhấn mạnh tính hữu ích của mô hình hóa phi tham số
trong việc hiểu hành vi của các deep network over-parametrized.


## Dừng sớm

Mặc dù các deep neural network có khả năng khớp các nhãn tùy ý,
ngay cả khi nhãn được gán sai hoặc ngẫu nhiên
:cite:`zhang2021understanding`,
khả năng này chỉ xuất hiện sau nhiều vòng lặp huấn luyện.
Một hướng nghiên cứu mới :cite:`Rolnick.Veit.Belongie.Shavit.2017`
đã cho thấy rằng trong bối cảnh có nhiễu nhãn,
neural network có xu hướng khớp dữ liệu được gán nhãn đúng trước
và chỉ sau đó mới nội suy dữ liệu bị gán nhãn sai.
Hơn nữa, đã xác lập được rằng hiện tượng này
chuyển trực tiếp thành một đảm bảo về khả năng tổng quát hóa:
bất cứ khi nào một mô hình đã khớp dữ liệu được gán nhãn đúng
nhưng chưa khớp các ví dụ được gán nhãn ngẫu nhiên có trong tập huấn luyện,
thì trên thực tế nó đã tổng quát hóa :cite:`Garg.Balakrishnan.Kolter.Lipton.2021`.

Những phát hiện này cùng nhau giúp tạo động lực cho *dừng sớm*,
một kỹ thuật kinh điển để regularize các deep neural network.
Ở đây, thay vì trực tiếp ràng buộc giá trị của weight,
ta ràng buộc số epoch huấn luyện.
Cách phổ biến nhất để xác định tiêu chí dừng
là theo dõi validation error trong suốt quá trình huấn luyện
(thường kiểm tra một lần sau mỗi epoch)
và dừng huấn luyện khi validation error
không giảm quá một lượng nhỏ nào đó $\epsilon$
trong một số epoch nhất định.
Điều này đôi khi được gọi là *tiêu chí patience*.
Bên cạnh tiềm năng dẫn đến khả năng tổng quát hóa tốt hơn
trong bối cảnh nhãn nhiễu,
một lợi ích khác của dừng sớm là tiết kiệm thời gian.
Khi tiêu chí patience được đáp ứng, ta có thể kết thúc huấn luyện.
Đối với các mô hình lớn có thể cần nhiều ngày huấn luyện
đồng thời trên tám hoặc nhiều GPU hơn,
dừng sớm được tinh chỉnh tốt có thể giúp các nhà nghiên cứu tiết kiệm nhiều ngày
và giúp đơn vị sử dụng lao động của họ tiết kiệm nhiều nghìn đô la.

Đáng chú ý, khi không có nhiễu nhãn và các dataset là *realizable*
(các lớp thực sự có thể phân tách được, ví dụ: phân biệt mèo với chó),
dừng sớm thường không dẫn đến những cải thiện đáng kể về khả năng tổng quát hóa.
Mặt khác, khi có nhiễu nhãn,
hoặc sự biến thiên nội tại trong nhãn
(ví dụ: dự đoán tỷ lệ tử vong ở bệnh nhân),
dừng sớm là rất quan trọng.
Huấn luyện mô hình cho đến khi chúng nội suy dữ liệu nhiễu thường là một ý tưởng tồi.


## Các Phương pháp Regularization Cổ điển cho Deep Network

Trong :numref:`chap_regression`, chúng ta đã mô tả
một số kỹ thuật regularization cổ điển
để ràng buộc độ phức tạp của mô hình.
Cụ thể, :numref:`sec_weight_decay`
đã giới thiệu một phương pháp gọi là weight decay,
bao gồm việc thêm một regularization term vào loss function
để phạt các giá trị lớn của weight.
Tùy thuộc vào weight norm nào bị phạt,
kỹ thuật này được gọi là ridge regularization (với penalty $\ell_2$)
hoặc lasso regularization (với penalty $\ell_1$).
Trong phân tích cổ điển về các regularizer này,
chúng được xem là đủ hạn chế đối với các giá trị
mà weight có thể nhận để ngăn mô hình khớp các nhãn tùy ý.

Trong các implementation deep learning,
weight decay vẫn là một công cụ phổ biến.
Tuy nhiên, các nhà nghiên cứu đã lưu ý
rằng cường độ thông thường của regularization $\ell_2$
không đủ để ngăn các network
nội suy dữ liệu :cite:`zhang2021understanding` và do đó các lợi ích, nếu được diễn giải
là regularization, có thể chỉ có ý nghĩa
khi kết hợp với tiêu chí dừng sớm.
Nếu không có dừng sớm, có thể
giống như số lượng layer
hoặc số lượng node (trong deep learning)
hoặc distance metric (trong 1-nearest neighbor),
các phương pháp này có thể dẫn đến khả năng tổng quát hóa tốt hơn
không phải vì chúng ràng buộc một cách có ý nghĩa
sức mạnh của neural network
mà bởi vì bằng cách nào đó chúng mã hóa các inductive bias
tương thích tốt hơn với các pattern
được tìm thấy trong các dataset quan tâm.
Vì vậy, các regularizer cổ điển vẫn phổ biến
trong các implementation deep learning,
ngay cả khi cơ sở lý thuyết
cho hiệu quả của chúng có thể hoàn toàn khác biệt.

Đáng chú ý, các nhà nghiên cứu deep learning cũng đã phát triển
dựa trên các kỹ thuật lần đầu trở nên phổ biến
trong các bối cảnh regularization cổ điển,
chẳng hạn như thêm nhiễu vào input của mô hình.
Trong phần tiếp theo, chúng ta sẽ giới thiệu
kỹ thuật dropout nổi tiếng
(do :citet:`Srivastava.Hinton.Krizhevsky.ea.2014` phát minh),
đã trở thành một trụ cột của deep learning,
mặc dù cơ sở lý thuyết cho hiệu quả của nó
vẫn bí ẩn tương tự.


## Tóm tắt

Không giống các mô hình tuyến tính cổ điển,
vốn có xu hướng có ít parameter hơn số ví dụ,
deep network có xu hướng được over-parametrized,
và đối với hầu hết các tác vụ đều có khả năng
khớp hoàn hảo tập huấn luyện.
*Chế độ nội suy (interpolation regime)* này thách thức
nhiều trực giác vốn được tin chắc.
Về mặt chức năng, neural network trông giống các mô hình tham số.
Nhưng xem chúng như các mô hình phi tham số
đôi khi có thể là một nguồn trực giác đáng tin cậy hơn.
Bởi vì thường mọi deep network đang được xem xét
đều có khả năng khớp tất cả các nhãn huấn luyện,
gần như mọi cải thiện đều phải đến từ việc giảm overfitting
(thu hẹp *generalization gap*).
Nghịch lý thay, các can thiệp
làm giảm generalization gap
đôi khi dường như làm tăng độ phức tạp mô hình
và những lúc khác lại dường như làm giảm độ phức tạp.
Tuy nhiên, các phương pháp này hiếm khi làm giảm độ phức tạp
đủ để lý thuyết cổ điển
giải thích khả năng tổng quát hóa của deep network,
và *vì sao một số lựa chọn nhất định dẫn đến khả năng tổng quát hóa được cải thiện*
phần lớn vẫn là một câu hỏi mở lớn
bất chấp những nỗ lực phối hợp của nhiều nhà nghiên cứu xuất sắc.


## Bài tập

1. Theo nghĩa nào các thước đo độ phức tạp truyền thống không giải thích được khả năng tổng quát hóa của deep neural network?
1. Vì sao *dừng sớm* có thể được xem là một kỹ thuật regularization?
1. Các nhà nghiên cứu thường xác định tiêu chí dừng như thế nào?
1. Yếu tố quan trọng nào dường như phân biệt các trường hợp mà dừng sớm mang lại cải thiện lớn về khả năng tổng quát hóa?
1. Ngoài khả năng tổng quát hóa, hãy mô tả một lợi ích khác của dừng sớm.

[Thảo luận](https://discuss.d2l.ai/t/7473)