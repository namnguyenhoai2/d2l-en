# Lan truyền xuôi, lan truyền ngược và đồ thị tính toán
:label:`sec_backprop`

Cho đến nay, chúng ta đã huấn luyện các mô hình
bằng stochastic gradient descent theo minibatch.
Tuy nhiên, khi triển khai thuật toán,
chúng ta chỉ quan tâm đến các phép tính liên quan
đến *lan truyền xuôi* qua mô hình.
Khi đến lúc tính gradient,
chúng ta chỉ cần gọi hàm backpropagation do framework deep learning cung cấp.

Việc tự động tính gradient
đơn giản hóa sâu sắc
quá trình triển khai các thuật toán deep learning.
Trước khi có automatic differentiation,
ngay cả những thay đổi nhỏ đối với các mô hình phức tạp cũng đòi hỏi
phải tính lại các đạo hàm phức tạp bằng tay.
Đáng ngạc nhiên là các bài báo học thuật thường phải dành
nhiều trang để suy ra các quy tắc cập nhật.
Mặc dù chúng ta vẫn phải dựa vào automatic differentiation
để có thể tập trung vào những phần thú vị,
bạn nên biết các gradient này
được tính như thế nào ở bên trong
nếu muốn vượt qua sự hiểu biết
hời hợt về deep learning.

Trong phần này, chúng ta sẽ đi sâu
vào chi tiết của *lan truyền ngược*
(thường được gọi là *backpropagation*).
Để cung cấp một số hiểu biết về cả
kỹ thuật lẫn cách triển khai chúng,
chúng ta dựa vào một số kiến thức toán học cơ bản và đồ thị tính toán.
Để bắt đầu, chúng ta tập trung trình bày
một MLP có một hidden layer
với weight decay ($\ell_2$ regularization, sẽ được mô tả trong các chương sau).

## Lan truyền xuôi

*Lan truyền xuôi* (hoặc *forward pass*) là việc tính toán và lưu trữ
các biến trung gian (bao gồm cả đầu ra)
cho một neural network theo thứ tự
từ input layer đến output layer.
Bây giờ, chúng ta sẽ lần lượt tìm hiểu cơ chế
của một neural network có một hidden layer.
Điều này có vẻ tẻ nhạt, nhưng theo lời bất hủ
của bậc thầy funk James Brown,
bạn phải "trả giá để làm chủ".


Để đơn giản, giả sử
rằng ví dụ đầu vào là $\mathbf{x}\in \mathbb{R}^d$
và hidden layer của chúng ta không bao gồm bias term.
Ở đây, biến trung gian là:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

trong đó $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
là tham số trọng số của hidden layer.
Sau khi đưa biến trung gian
$\mathbf{z}\in \mathbb{R}^h$ qua
hàm kích hoạt $\phi$,
ta thu được vector kích hoạt hidden có độ dài $h$:

$$\mathbf{h}= \phi (\mathbf{z}).$$

Đầu ra của hidden layer $\mathbf{h}$
cũng là một biến trung gian.
Giả sử các tham số của output layer
chỉ có trọng số
$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$,
chúng ta có thể thu được một biến output layer
với vector có độ dài $q$:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Giả sử hàm mất mát là $l$
và nhãn của ví dụ là $y$,
khi đó chúng ta có thể tính hạng tử mất mát
cho một ví dụ dữ liệu đơn lẻ,

$$L = l(\mathbf{o}, y).$$

Như chúng ta sẽ thấy trong định nghĩa về $\ell_2$ regularization
được giới thiệu sau,
với hyperparameter $\lambda$,
hạng tử regularization là

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_\textrm{F}^2 + \|\mathbf{W}^{(2)}\|_\textrm{F}^2\right),$$
:eqlabel:`eq_forward-s`

trong đó Frobenius norm của ma trận
đơn giản là $\ell_2$ norm được áp dụng
sau khi làm phẳng ma trận thành một vector.
Cuối cùng, mất mát đã regularize của mô hình
trên một ví dụ dữ liệu cho trước là:

$$J = L + s.$$

Chúng ta gọi $J$ là *hàm mục tiêu*
trong phần thảo luận sau đây.


## Đồ thị tính toán của lan truyền xuôi

Việc vẽ *đồ thị tính toán* giúp chúng ta trực quan hóa
các phụ thuộc của toán tử
và biến trong phép tính.
:numref:`fig_forward` chứa đồ thị liên quan
đến mạng đơn giản được mô tả ở trên,
trong đó hình vuông biểu thị biến và hình tròn biểu thị toán tử.
Góc dưới bên trái biểu thị đầu vào
và góc trên bên phải là đầu ra.
Lưu ý rằng hướng của các mũi tên
(minh họa luồng dữ liệu)
chủ yếu hướng sang phải và lên trên.

![Đồ thị tính toán của lan truyền xuôi.](../img/forward.svg)
:label:`fig_forward`

## Lan truyền ngược

*Backpropagation* là phương pháp tính
gradient của các tham số neural network.
Tóm lại, phương pháp này duyệt mạng theo thứ tự ngược,
từ output layer đến input layer,
theo *quy tắc chuỗi* của phép tính vi phân.
Thuật toán lưu trữ mọi biến trung gian
(đạo hàm riêng)
cần thiết khi tính gradient
theo một số tham số.
Giả sử chúng ta có các hàm
$\mathsf{Y}=f(\mathsf{X})$
và $\mathsf{Z}=g(\mathsf{Y})$,
trong đó đầu vào và đầu ra
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$
là các tensor có hình dạng tùy ý.
Bằng cách sử dụng quy tắc chuỗi,
chúng ta có thể tính đạo hàm
của $\mathsf{Z}$ theo $\mathsf{X}$ thông qua

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \textrm{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Ở đây, chúng ta sử dụng toán tử $\textrm{prod}$
để nhân các đối số của nó
sau khi thực hiện các thao tác cần thiết,
chẳng hạn như chuyển vị và hoán đổi vị trí đầu vào.
Đối với vector, điều này rất đơn giản:
nó chỉ là phép nhân ma trận--ma trận.
Đối với tensor có số chiều cao hơn,
chúng ta sử dụng phép tương ứng phù hợp.
Toán tử $\textrm{prod}$ che giấu toàn bộ phần ký hiệu phức tạp.

Hãy nhớ rằng
các tham số của mạng đơn giản có một hidden layer,
có đồ thị tính toán trong :numref:`fig_forward`,
là $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$.
Mục tiêu của backpropagation là
tính các gradient $\partial J/\partial \mathbf{W}^{(1)}$
và $\partial J/\partial \mathbf{W}^{(2)}$.
Để thực hiện điều này, chúng ta áp dụng quy tắc chuỗi
và lần lượt tính gradient của
mỗi biến trung gian và tham số.
Thứ tự tính toán được đảo ngược
so với thứ tự trong lan truyền xuôi,
vì chúng ta cần bắt đầu từ kết quả của đồ thị tính toán
và lần ngược về phía các tham số.
Bước đầu tiên là tính gradient
của hàm mục tiêu $J=L+s$
theo hạng tử mất mát $L$
và hạng tử regularization $s$:

$$\frac{\partial J}{\partial L} = 1 \; \textrm{and} \; \frac{\partial J}{\partial s} = 1.$$

Tiếp theo, chúng ta tính gradient của hàm mục tiêu
theo biến của output layer $\mathbf{o}$
dựa trên quy tắc chuỗi:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \textrm{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Tiếp theo, chúng ta tính gradient
của hạng tử regularization
theo cả hai tham số:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \textrm{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Bây giờ, chúng ta có thể tính gradient
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$
của các tham số mô hình gần output layer nhất.
Áp dụng quy tắc chuỗi cho ta:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

Để thu được gradient theo $\mathbf{W}^{(1)}$
chúng ta cần tiếp tục backpropagation
dọc theo output layer đến hidden layer.
Gradient theo đầu ra của hidden layer
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ được cho bởi


$$
\frac{\partial J}{\partial \mathbf{h}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Vì hàm kích hoạt $\phi$ áp dụng theo từng phần tử,
việc tính gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
của biến trung gian $\mathbf{z}$
yêu cầu chúng ta sử dụng toán tử nhân theo từng phần tử,
ký hiệu là $\odot$:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Cuối cùng, ta có thể thu được gradient
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
của các tham số mô hình gần lớp đầu vào nhất.
Theo quy tắc chuỗi, ta có

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## Huấn luyện Mạng nơ-ron

Khi huấn luyện mạng nơ-ron,
lan truyền xuôi và lan truyền ngược phụ thuộc lẫn nhau.
Cụ thể, đối với lan truyền xuôi,
ta duyệt đồ thị tính toán theo hướng của các phụ thuộc
và tính tất cả các biến trên đường đi của nó.
Sau đó, chúng được dùng cho backpropagation,
trong đó thứ tự tính toán trên đồ thị được đảo ngược.

Hãy lấy mạng đơn giản đã đề cập ở trên làm ví dụ minh họa.
Một mặt,
việc tính hạng tử regularization :eqref:`eq_forward-s`
trong quá trình lan truyền xuôi
phụ thuộc vào các giá trị hiện tại của tham số mô hình $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$.
Chúng được thuật toán tối ưu hóa cung cấp dựa trên backpropagation ở lần lặp gần nhất.
Mặt khác,
việc tính gradient cho tham số
:eqref:`eq_backprop-J-h` trong quá trình backpropagation
phụ thuộc vào giá trị hiện tại của đầu ra lớp ẩn $\mathbf{h}$,
được cung cấp bởi lan truyền xuôi.


Vì vậy, khi huấn luyện mạng nơ-ron, sau khi các tham số mô hình được khởi tạo,
ta luân phiên lan truyền xuôi với backpropagation,
cập nhật các tham số mô hình bằng gradient do backpropagation cung cấp.
Lưu ý rằng backpropagation tái sử dụng các giá trị trung gian đã lưu từ lan truyền xuôi để tránh các phép tính trùng lặp.
Một hệ quả là ta cần giữ lại
các giá trị trung gian cho đến khi backpropagation hoàn tất.
Đây cũng là một trong các lý do khiến việc huấn luyện
cần nhiều bộ nhớ hơn đáng kể so với dự đoán thông thường.
Ngoài ra, kích thước của các giá trị trung gian này xấp xỉ
tỷ lệ thuận với số lượng lớp mạng và kích thước batch.
Do đó,
việc huấn luyện các mạng sâu hơn với batch size lớn hơn
dễ dẫn đến lỗi *out-of-memory* hơn.


## Tóm tắt

Lan truyền xuôi tính toán tuần tự và lưu trữ các biến trung gian trong đồ thị tính toán được xác định bởi mạng nơ-ron. Quá trình này đi từ lớp đầu vào đến lớp đầu ra.
Backpropagation tính toán tuần tự và lưu trữ gradient của các biến trung gian và tham số trong mạng nơ-ron theo thứ tự đảo ngược.
Khi huấn luyện các mô hình deep learning, lan truyền xuôi và backpropagation phụ thuộc lẫn nhau,
và việc huấn luyện cần nhiều bộ nhớ hơn đáng kể so với dự đoán.


## Bài tập

1. Giả sử các đầu vào $\mathbf{X}$ của một hàm vô hướng $f$ nào đó là các ma trận $n \times m$. Gradient của $f$ theo $\mathbf{X}$ có số chiều là bao nhiêu?
1. Thêm một bias vào lớp ẩn của mô hình được mô tả trong phần này (bạn không cần đưa bias vào hạng tử regularization).
    1. Vẽ đồ thị tính toán tương ứng.
    1. Suy ra các phương trình lan truyền xuôi và lan truyền ngược.
1. Tính mức sử dụng bộ nhớ khi huấn luyện và dự đoán trong mô hình được mô tả ở phần này.
1. Giả sử bạn muốn tính đạo hàm bậc hai. Điều gì xảy ra với đồ thị tính toán? Bạn dự đoán phép tính sẽ mất bao lâu?
1. Giả sử đồ thị tính toán quá lớn đối với GPU của bạn.
    1. Bạn có thể phân vùng nó trên nhiều GPU không?
    1. Những ưu điểm và nhược điểm so với việc huấn luyện trên một minibatch nhỏ hơn là gì?

[Thảo luận](https://discuss.d2l.ai/t/102)