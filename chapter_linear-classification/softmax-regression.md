# Hồi quy Softmax
:label:`sec_softmax`

Trong :numref:`sec_linear_regression`, chúng ta đã giới thiệu hồi quy tuyến tính,
triển khai từ đầu trong :numref:`sec_linear_scratch`
và một lần nữa sử dụng các API cấp cao của một framework deep learning
trong :numref:`sec_linear_concise` để thực hiện phần việc nặng nhọc.

Hồi quy là công cụ chúng ta dùng khi
muốn trả lời các câu hỏi *bao nhiêu?* hoặc *bao nhiêu cái?*.
Nếu bạn muốn dự đoán số đô la (giá)
mà một ngôi nhà sẽ được bán,
hoặc số trận thắng mà một đội bóng chày có thể có,
hoặc số ngày một bệnh nhân
sẽ nằm viện trước khi xuất viện,
thì có lẽ bạn đang cần một mô hình hồi quy.
Tuy nhiên, ngay cả trong các mô hình hồi quy,
vẫn có những khác biệt quan trọng.
Chẳng hạn, giá của một ngôi nhà
sẽ không bao giờ âm và các thay đổi thường có thể *tương đối* so với giá cơ sở.
Do đó, có thể hiệu quả hơn khi hồi quy
trên logarit của giá.
Tương tự, số ngày bệnh nhân nằm viện
là một biến ngẫu nhiên *rời rạc không âm*.
Vì vậy, least mean squares cũng có thể không phải là một phương pháp lý tưởng.
Loại mô hình hóa thời gian đến sự kiện này
đi kèm nhiều phức tạp khác, được xử lý
trong một lĩnh vực chuyên biệt gọi là *survival modeling*.

Mục đích ở đây không phải là làm bạn choáng ngợp, mà chỉ
để cho bạn biết rằng việc ước lượng còn nhiều điều hơn
việc đơn thuần tối thiểu hóa sai số bình phương.
Và rộng hơn, supervised learning còn nhiều điều hơn hồi quy.
Trong phần này, chúng ta tập trung vào các bài toán *phân loại*
khi gác lại các câu hỏi *bao nhiêu?*
và thay vào đó tập trung vào các câu hỏi *thuộc loại nào?*.



* Email này thuộc thư mục spam hay hộp thư đến?
* Khách hàng này có nhiều khả năng đăng ký
  hay không đăng ký một dịch vụ thuê bao?
* Hình ảnh này mô tả một con lừa, chó, mèo hay gà trống?
* Aston có nhiều khả năng xem bộ phim nào tiếp theo?
* Bạn sẽ đọc phần nào của cuốn sách tiếp theo?

Trong cách nói thông thường, các practitioner machine learning
dùng từ *classification* để chỉ hai bài toán khác nhau một cách tinh tế:
(i) những bài toán mà chúng ta chỉ quan tâm đến
việc gán cứng các ví dụ vào các hạng mục (lớp);
và (ii) những bài toán mà chúng ta muốn gán mềm,
tức là đánh giá xác suất áp dụng của từng hạng mục.
Sự khác biệt này thường bị làm mờ đi, một phần
vì ngay cả khi chúng ta chỉ quan tâm đến các gán cứng,
chúng ta vẫn thường dùng các mô hình tạo ra gán mềm.

Hơn nữa, có những trường hợp nhiều hơn một nhãn có thể đúng.
Chẳng hạn, một bài báo có thể đồng thời đề cập đến
các chủ đề giải trí, kinh doanh và du hành không gian,
nhưng không đề cập đến y học hoặc thể thao.
Do đó, việc phân loại nó vào chỉ một trong các hạng mục trên
sẽ không thực sự hữu ích.
Bài toán này thường được gọi là [phân loại đa nhãn](https://en.wikipedia.org/wiki/Multi-label_classification).
Xem :citet:`Tsoumakas.Katakis.2007` để biết tổng quan
và :citet:`Huang.Xu.Yu.2015`
để biết một thuật toán hiệu quả khi gắn nhãn hình ảnh.

## Phân loại
:label:`subsec_classification-problem`

Để bắt đầu, hãy xét
một bài toán phân loại hình ảnh đơn giản.
Ở đây, mỗi đầu vào bao gồm một hình ảnh thang độ xám $2\times2$.
Chúng ta có thể biểu diễn giá trị mỗi pixel bằng một vô hướng,
thu được bốn đặc trưng $x_1, x_2, x_3, x_4$.
Ngoài ra, giả sử mỗi hình ảnh thuộc một
trong các hạng mục "cat", "chicken" và "dog".

Tiếp theo, chúng ta phải chọn cách biểu diễn các nhãn.
Chúng ta có hai lựa chọn rõ ràng.
Có lẽ trực giác tự nhiên nhất là
chọn $y \in \{1, 2, 3\}$,
trong đó các số nguyên lần lượt biểu diễn
$\{\textrm{dog}, \textrm{cat}, \textrm{chicken}\}$.
Đây là một cách tuyệt vời để *lưu trữ* thông tin như vậy trên máy tính.
Nếu các hạng mục có một thứ tự tự nhiên nào đó,
chẳng hạn nếu chúng ta đang cố dự đoán
$\{\textrm{baby}, \textrm{toddler}, \textrm{adolescent}, \textrm{young adult}, \textrm{adult}, \textrm{geriatric}\}$,
thì thậm chí có thể hợp lý khi xem đây là
một bài toán [hồi quy thứ tự](https://en.wikipedia.org/wiki/Ordinal_regression)
và giữ các nhãn ở định dạng này.
Xem :citet:`Moon.Smola.Chang.ea.2010` để biết tổng quan
về các loại hàm mất mát xếp hạng khác nhau
và :citet:`Beutel.Murray.Faloutsos.ea.2014` để biết một phương pháp Bayesian
xử lý các phản hồi có nhiều hơn một mode.

Nhìn chung, các bài toán phân loại không có
thứ tự tự nhiên giữa các lớp.
May mắn thay, từ lâu các nhà thống kê đã phát minh ra một cách đơn giản
để biểu diễn dữ liệu phân loại: *one-hot encoding*.
Một one-hot encoding là một vector
có số thành phần bằng số hạng mục.
Thành phần tương ứng với hạng mục của một instance cụ thể được đặt thành 1
và tất cả các thành phần khác được đặt thành 0.
Trong trường hợp của chúng ta, một nhãn $y$ sẽ là vector ba chiều,
với $(1, 0, 0)$ tương ứng với "cat", $(0, 1, 0)$ với "chicken",
và $(0, 0, 1)$ với "dog":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

### Mô hình tuyến tính

Để ước lượng các xác suất có điều kiện
liên quan đến mọi lớp có thể có,
chúng ta cần một mô hình có nhiều đầu ra, mỗi đầu ra cho một lớp.
Để xử lý phân loại bằng các mô hình tuyến tính,
chúng ta sẽ cần số hàm affine bằng số đầu ra.
Nói chính xác, chúng ta chỉ cần ít hơn một hàm,
vì hạng mục cuối cùng phải là phần chênh lệch
giữa $1$ và tổng của các hạng mục còn lại,
nhưng vì lý do đối xứng,
chúng ta sử dụng một cách tham số hóa hơi dư thừa.
Mỗi đầu ra tương ứng với hàm affine riêng của nó.
Trong trường hợp của chúng ta, vì có 4 đặc trưng và 3 hạng mục đầu ra có thể có,
chúng ta cần 12 vô hướng để biểu diễn các trọng số ($w$ có chỉ số dưới),
và 3 vô hướng để biểu diễn các bias ($b$ có chỉ số dưới). Điều này cho:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Sơ đồ neural network tương ứng
được hiển thị trong :numref:`fig_softmaxreg`.
Cũng như trong hồi quy tuyến tính,
chúng ta sử dụng một neural network một tầng.
Và vì việc tính mỗi đầu ra, $o_1, o_2$ và $o_3$,
phụ thuộc vào mọi đầu vào, $x_1$, $x_2$, $x_3$ và $x_4$,
tầng đầu ra cũng có thể được mô tả là một *fully connected layer*.

![Hồi quy softmax là một neural network một tầng.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Để có ký hiệu ngắn gọn hơn, chúng ta sử dụng vector và ma trận:
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$
phù hợp hơn nhiều cho toán học và code.
Lưu ý rằng chúng ta đã gom tất cả trọng số vào một ma trận $3 \times 4$ và tất cả bias
$\mathbf{b} \in \mathbb{R}^3$ vào một vector.

### Softmax
:label:`subsec_softmax_operation`

Giả sử có một hàm mất mát phù hợp,
chúng ta có thể thử trực tiếp tối thiểu hóa chênh lệch
giữa $\mathbf{o}$ và các nhãn $\mathbf{y}$.
Mặc dù việc xem phân loại
như một bài toán hồi quy có giá trị vector hoạt động tốt đáng ngạc nhiên,
nó vẫn không thỏa đáng theo các khía cạnh sau:

* Không có gì đảm bảo rằng các đầu ra $o_i$ có tổng bằng $1$ như cách chúng ta mong đợi ở xác suất.
* Không có gì đảm bảo rằng các đầu ra $o_i$ thậm chí không âm, ngay cả khi tổng các đầu ra của chúng bằng $1$, hoặc rằng chúng không vượt quá $1$.

Cả hai khía cạnh đều khiến bài toán ước lượng khó giải
và nghiệm rất nhạy cảm với outlier.
Chẳng hạn, nếu giả sử có
một quan hệ tuyến tính dương
giữa số lượng phòng ngủ và khả năng
một người sẽ mua nhà,
xác suất có thể vượt quá $1$
khi nói đến việc mua một biệt thự!
Do đó, chúng ta cần một cơ chế để "ép" các đầu ra lại.

Có nhiều cách để đạt được mục tiêu này.
Chẳng hạn, chúng ta có thể giả sử các đầu ra
$\mathbf{o}$ là các phiên bản bị nhiễu của $\mathbf{y}$,
trong đó nhiễu được tạo ra bằng cách cộng nhiễu $\boldsymbol{\epsilon}$
lấy mẫu từ phân phối chuẩn.
Nói cách khác, $\mathbf{y} = \mathbf{o} + \boldsymbol{\epsilon}$,
trong đó $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
Đây được gọi là [mô hình probit](https://en.wikipedia.org/wiki/Probit_model),
lần đầu được giới thiệu bởi :citet:`Fechner.1860`.
Dù hấp dẫn, nó không hoạt động tốt bằng
và cũng không dẫn đến một bài toán tối ưu hóa đặc biệt thuận tiện,
khi so sánh với softmax.

Một cách khác để đạt được mục tiêu này
(và bảo đảm tính không âm) là sử dụng
hàm mũ $P(y = i) \propto \exp o_i$.
Điều này thực sự thỏa mãn yêu cầu rằng
xác suất lớp có điều kiện
tăng theo $o_i$ tăng, tức là nó đơn điệu,
và mọi xác suất đều không âm.
Sau đó, chúng ta có thể biến đổi các giá trị này để tổng của chúng bằng $1$
bằng cách chia mỗi giá trị cho tổng của chúng.
Quá trình này được gọi là *normalization*.
Kết hợp hai phần này
cho chúng ta hàm *softmax*:

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{where}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$
:eqlabel:`eq_softmax_y_and_o`

Lưu ý rằng tọa độ lớn nhất của $\mathbf{o}$
tương ứng với lớp có khả năng xảy ra cao nhất theo $\hat{\mathbf{y}}$.
Hơn nữa, vì phép toán softmax
bảo toàn thứ tự giữa các đối số của nó,
chúng ta không cần tính softmax
để xác định lớp nào đã được gán xác suất cao nhất. Do đó,

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$


Ý tưởng về softmax có từ :citet:`Gibbs.1902`,
người đã điều chỉnh các ý tưởng từ vật lý.
Xa hơn nữa trong quá khứ, Boltzmann,
cha đẻ của vật lý thống kê hiện đại,
đã sử dụng thủ thuật này để mô hình hóa một phân phối
trên các trạng thái năng lượng trong các phân tử khí.
Cụ thể, ông đã phát hiện rằng mức độ phổ biến
của một trạng thái năng lượng trong một ensemble nhiệt động lực học,
chẳng hạn như các phân tử trong chất khí,
tỷ lệ với $\exp(-E/kT)$.
Ở đây, $E$ là năng lượng của một trạng thái,
$T$ là nhiệt độ, và $k$ là hằng số Boltzmann.
Khi các nhà thống kê nói về việc tăng hoặc giảm
"nhiệt độ" của một hệ thống thống kê,
họ đề cập đến việc thay đổi $T$
để ưu tiên các trạng thái có năng lượng thấp hơn hoặc cao hơn.
Theo ý tưởng của Gibbs, năng lượng tương đương với sai số.
Các mô hình dựa trên năng lượng (energy-based models) :cite:`Ranzato.Boureau.Chopra.ea.2007`
sử dụng quan điểm này khi mô tả
các bài toán trong deep learning.

### Vector hóa
:label:`subsec_softmax_vectorization`

Để cải thiện hiệu quả tính toán,
chúng ta vector hóa các phép tính trong các minibatch dữ liệu.
Giả sử rằng ta được cung cấp một minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$
gồm $n$ mẫu có số chiều (số lượng đầu vào) là $d$.
Hơn nữa, giả sử rằng đầu ra có $q$ hạng mục.
Khi đó, trọng số thỏa mãn $\mathbf{W} \in \mathbb{R}^{d \times q}$
và bias thỏa mãn $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Điều này tăng tốc phép toán chi phối thành
một phép nhân ma trận--ma trận $\mathbf{X} \mathbf{W}$.
Hơn nữa, vì mỗi hàng trong $\mathbf{X}$ biểu diễn một mẫu dữ liệu,
bản thân phép toán softmax có thể được tính *theo hàng*:
với mỗi hàng của $\mathbf{O}$, lấy hàm mũ của tất cả các phần tử
rồi chuẩn hóa chúng bằng tổng.
Tuy nhiên, lưu ý rằng cần cẩn thận
để tránh lấy hàm mũ và logarit của các số lớn,
vì điều này có thể gây tràn số hoặc tràn dưới số.
Các framework deep learning tự động xử lý việc này.

## Hàm mất mát
:label:`subsec_softmax-regression-loss-func`

Bây giờ khi đã có một ánh xạ từ các đặc trưng $\mathbf{x}$
sang các xác suất $\mathbf{\hat{y}}$,
chúng ta cần một cách để tối ưu hóa độ chính xác của ánh xạ này.
Chúng ta sẽ dựa vào ước lượng hợp lý cực đại (maximum likelihood estimation),
chính là phương pháp mà ta đã gặp
khi đưa ra biện minh xác suất
cho hàm mất mát sai số bình phương trung bình trong
:numref:`subsec_normal_distribution_and_squared_loss`.

### Log-Likelihood

Hàm softmax cho ta một vector $\hat{\mathbf{y}}$,
mà ta có thể diễn giải là các xác suất có điều kiện (được ước lượng)
của từng lớp, với bất kỳ đầu vào $\mathbf{x}$ nào,
chẳng hạn $\hat{y}_1$ = $P(y=\textrm{cat} \mid \mathbf{x})$.
Trong phần sau, chúng ta giả sử rằng đối với một tập dữ liệu
có đặc trưng $\mathbf{X}$, các nhãn $\mathbf{Y}$
được biểu diễn bằng một vector nhãn one-hot encoding.
Ta có thể so sánh các ước lượng với thực tế
bằng cách kiểm tra xác suất của các lớp thực tế
theo mô hình của chúng ta, với các đặc trưng đã cho:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

Chúng ta được phép sử dụng phép phân tích này
vì giả sử rằng mỗi nhãn được lấy mẫu độc lập
từ phân phối tương ứng $P(\mathbf{y}\mid\mathbf{x}^{(i)})$.
Vì việc tối đa hóa tích các hạng tử khá bất tiện,
ta lấy logarit âm để thu được bài toán tương đương
là tối thiểu hóa negative log-likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

trong đó, với bất kỳ cặp nhãn $\mathbf{y}$
và dự đoán của mô hình $\hat{\mathbf{y}}$
trên $q$ lớp nào, hàm mất mát $l$ là

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

Vì các lý do sẽ được giải thích sau,
hàm mất mát trong :eqref:`eq_l_cross_entropy`
thường được gọi là *hàm mất mát cross-entropy*.
Vì $\mathbf{y}$ là một vector one-hot có độ dài $q$,
tổng trên tất cả tọa độ $j$ của nó triệt tiêu mọi hạng tử trừ một.
Lưu ý rằng mất mát $l(\mathbf{y}, \hat{\mathbf{y}})$
bị chặn dưới bởi $0$
bất cứ khi nào $\hat{\mathbf{y}}$ là một vector xác suất:
không phần tử riêng lẻ nào lớn hơn $1$,
do đó logarit âm của chúng không thể nhỏ hơn $0$;
$l(\mathbf{y}, \hat{\mathbf{y}}) = 0$ chỉ khi ta dự đoán
nhãn thực tế với *sự chắc chắn*.
Điều này không thể xảy ra với bất kỳ thiết lập hữu hạn nào của trọng số
vì để đưa đầu ra softmax về gần $1$
cần đưa đầu vào tương ứng $o_i$ tới vô cùng
(hoặc đưa tất cả các đầu ra khác $o_j$ với $j \neq i$ tới âm vô cùng).
Ngay cả khi mô hình của ta có thể gán xác suất đầu ra là $0$,
bất kỳ sai lầm nào khi đưa ra mức độ tin cậy cao như vậy
đều sẽ chịu mất mát vô hạn ($-\log 0 = \infty$).


### Softmax và hàm mất mát Cross-Entropy
:label:`subsec_softmax_and_derivatives`

Vì hàm softmax
và hàm mất mát cross-entropy tương ứng rất phổ biến,
việc hiểu rõ hơn một chút cách chúng được tính là đáng giá.
Thay :eqref:`eq_softmax_y_and_o` vào định nghĩa của hàm mất mát
trong :eqref:`eq_l_cross_entropy`
và sử dụng định nghĩa của softmax, ta thu được

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

Để hiểu rõ hơn một chút điều đang diễn ra,
hãy xét đạo hàm theo bất kỳ logit $o_j$ nào. Ta có

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

Nói cách khác, đạo hàm là chênh lệch
giữa xác suất mà mô hình của ta gán,
được biểu diễn bởi phép toán softmax,
và điều thực sự đã xảy ra, được biểu diễn
bởi các phần tử trong vector nhãn one-hot.
Theo nghĩa này, nó rất giống
với điều ta đã thấy trong hồi quy,
nơi gradient là chênh lệch
giữa quan sát $y$ và ước lượng $\hat{y}$.
Đây không phải là sự trùng hợp.
Trong bất kỳ mô hình họ hàm mũ nào,
gradient của log-likelihood đều được cho chính xác bởi hạng tử này.
Thực tế này giúp việc tính gradient trở nên dễ dàng trong thực tế.

Bây giờ hãy xét trường hợp ta không chỉ quan sát một kết quả đơn lẻ
mà là toàn bộ một phân phối trên các kết quả.
Ta có thể sử dụng cùng biểu diễn như trước cho nhãn $\mathbf{y}$.
Khác biệt duy nhất là thay vì
một vector chỉ chứa các phần tử nhị phân,
chẳng hạn $(0, 0, 1)$, giờ đây ta có một vector xác suất tổng quát,
chẳng hạn $(0.1, 0.2, 0.7)$.
Phép toán mà ta đã sử dụng trước đó để xác định mất mát $l$
trong :eqref:`eq_l_cross_entropy`
vẫn hoạt động tốt,
chỉ là cách diễn giải có tính tổng quát hơn một chút.
Đó là giá trị kỳ vọng của mất mát đối với một phân phối trên các nhãn.
Mất mát này được gọi là *hàm mất mát cross-entropy* và là
một trong những hàm mất mát được sử dụng phổ biến nhất cho các bài toán phân loại.
Ta có thể làm rõ tên gọi này bằng cách giới thiệu những kiến thức cơ bản về lý thuyết thông tin.
Tóm lại, nó đo số bit cần thiết để mã hóa những gì ta quan sát, $\mathbf{y}$,
so với điều mà ta dự đoán sẽ xảy ra, $\hat{\mathbf{y}}$.
Sau đây, chúng ta sẽ đưa ra một giải thích rất cơ bản. Để biết thêm
chi tiết về lý thuyết thông tin, xem
:citet:`Cover.Thomas.1999` hoặc :citet:`mackay2003information`.



## Các khái niệm cơ bản về lý thuyết thông tin
:label:`subsec_info_theory_basics`

Nhiều bài báo deep learning sử dụng trực giác và thuật ngữ từ lý thuyết thông tin.
Để hiểu được chúng, ta cần một ngôn ngữ chung.
Đây là một hướng dẫn sinh tồn.
*Lý thuyết thông tin* giải quyết bài toán
mã hóa, giải mã, truyền tải,
và thao tác thông tin (còn được gọi là dữ liệu).

### Entropy

Ý tưởng trung tâm trong lý thuyết thông tin là định lượng
lượng thông tin chứa trong dữ liệu.
Điều này đặt ra một giới hạn cho khả năng nén dữ liệu của chúng ta.
Với một phân phối $P$, *entropy* của nó, $H[P]$, được định nghĩa là:

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

Một trong những định lý nền tảng của lý thuyết thông tin phát biểu
rằng để mã hóa dữ liệu được rút ngẫu nhiên từ phân phối $P$,
ta cần ít nhất $H[P]$ "nat" để mã hóa nó :cite:`Shannon.1948`.
Nếu bạn thắc mắc "nat" là gì, nó tương đương với bit
nhưng khi sử dụng mã có cơ số $e$ thay vì cơ số 2.
Do đó, một nat là $\frac{1}{\log(2)} \approx 1.44$ bit.


### Độ ngạc nhiên

Bạn có thể đang tự hỏi nén có liên quan gì đến dự đoán.
Hãy tưởng tượng rằng chúng ta có một luồng dữ liệu mà ta muốn nén.
Nếu ta luôn dễ dàng dự đoán token tiếp theo,
thì dữ liệu này dễ nén.
Hãy xét ví dụ cực đoan, trong đó mọi token trong luồng
luôn nhận cùng một giá trị.
Đó là một luồng dữ liệu rất nhàm chán!
Nó không chỉ nhàm chán mà còn dễ dự đoán.
Vì các token luôn giống nhau,
ta không phải truyền bất kỳ thông tin nào
để truyền đạt nội dung của luồng.
Dễ dự đoán, dễ nén.

Tuy nhiên, nếu ta không thể dự đoán hoàn hảo mọi sự kiện,
thì đôi khi ta có thể bất ngờ.
Sự bất ngờ của ta lớn hơn khi một sự kiện được gán xác suất thấp hơn.
Claude Shannon đã chọn $\log \frac{1}{P(j)} = -\log P(j)$
để định lượng *mức độ bất ngờ* của một người khi quan sát sự kiện $j$
sau khi đã gán cho nó xác suất (chủ quan) $P(j)$.
Khi đó, entropy được định nghĩa trong :eqref:`eq_softmax_reg_entropy`
là *mức độ bất ngờ kỳ vọng*
khi một người gán các xác suất đúng
thực sự khớp với quá trình sinh dữ liệu.


### Xem Lại Cross-Entropy

Vì vậy, nếu entropy là mức độ bất ngờ mà
một người biết xác suất thực trải nghiệm,
thì bạn có thể tự hỏi, cross-entropy là gì?
Cross-entropy *từ* $P$ *đến* $Q$, ký hiệu là $H(P, Q)$,
là mức độ bất ngờ kỳ vọng của một người quan sát có các xác suất chủ quan $Q$
khi thấy dữ liệu thực tế được sinh theo các xác suất $P$.
Giá trị này được cho bởi $H(P, Q) \stackrel{\textrm{def}}{=} \sum_j - P(j) \log Q(j)$.
Cross-entropy thấp nhất đạt được khi $P=Q$.
Trong trường hợp này, cross-entropy từ $P$ đến $Q$ là $H(P, P)= H(P)$.

Tóm lại, ta có thể xem mục tiêu phân loại cross-entropy
theo hai cách: (i) tối đa hóa likelihood của dữ liệu quan sát được;
và (ii) tối thiểu hóa mức độ bất ngờ của ta (và do đó số bit)
cần để truyền đạt các nhãn.

## Tóm Tắt và Thảo Luận

Trong phần này, chúng ta đã gặp hàm loss không tầm thường đầu tiên,
cho phép tối ưu hóa trên các không gian đầu ra *rời rạc*.
Điểm then chốt trong thiết kế của nó là ta đã dùng một cách tiếp cận xác suất,
coi các hạng mục rời rạc là các lần lấy mẫu từ một phân phối xác suất.
Như một hệ quả, ta đã gặp softmax,
một activation function tiện lợi giúp biến đổi
đầu ra của một layer neural network thông thường
thành các phân phối xác suất rời rạc hợp lệ.
Ta đã thấy rằng đạo hàm của cross-entropy loss
khi kết hợp với softmax
hoạt động rất giống
với đạo hàm của squared error;
cụ thể là lấy chênh lệch giữa
hành vi kỳ vọng và dự đoán của nó.
Và, mặc dù ta mới chỉ có thể
chạm đến bề mặt của chủ đề này,
ta đã gặp các mối liên hệ thú vị
với vật lý thống kê và lý thuyết thông tin.

Mặc dù như vậy là đủ để bạn bắt đầu,
và hy vọng đủ để khơi gợi hứng thú,
chúng ta hầu như chưa đi sâu vào đây.
Trong số những điều khác, ta đã bỏ qua các cân nhắc về tính toán.
Cụ thể, với bất kỳ fully connected layer nào có $d$ đầu vào và $q$ đầu ra,
chi phí tham số hóa và tính toán là $\mathcal{O}(dq)$,
điều này có thể quá cao trong thực tế.
May mắn thay, chi phí biến đổi $d$ đầu vào thành $q$ đầu ra này
có thể được giảm thông qua xấp xỉ và nén.
Ví dụ, Deep Fried Convnets :cite:`Yang.Moczulski.Denil.ea.2015`
sử dụng kết hợp các phép hoán vị,
biến đổi Fourier và scaling
để giảm chi phí từ bậc hai xuống log-linear.
Các kỹ thuật tương tự cũng áp dụng cho những
xấp xỉ ma trận có cấu trúc nâng cao hơn :cite:`sindhwani2015structured`.
Cuối cùng, ta có thể dùng các phép phân rã dạng quaternion
để giảm chi phí xuống còn $\mathcal{O}(\frac{dq}{n})$,
một lần nữa nếu ta sẵn sàng đánh đổi một lượng nhỏ độ chính xác
để lấy chi phí tính toán và lưu trữ :cite:`Zhang.Tay.Zhang.ea.2021`
dựa trên hệ số nén $n$.
Đây là một lĩnh vực nghiên cứu đang hoạt động.
Điều khiến nó trở nên thách thức là
ta không nhất thiết hướng tới
biểu diễn gọn nhất
hay số lượng phép toán dấu phẩy động nhỏ nhất,
mà là giải pháp
có thể được thực thi hiệu quả nhất trên các GPU hiện đại.

## Bài Tập

1. Ta có thể khám phá sâu hơn mối liên hệ giữa exponential families và softmax.
    1. Tính đạo hàm bậc hai của cross-entropy loss $l(\mathbf{y},\hat{\mathbf{y}})$ cho softmax.
    1. Tính phương sai của phân phối được cho bởi $\mathrm{softmax}(\mathbf{o})$ và chứng minh rằng nó khớp với đạo hàm bậc hai đã tính ở trên.
1. Giả sử ta có ba lớp xuất hiện với xác suất bằng nhau, tức là vector xác suất là $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    1. Vấn đề là gì nếu ta cố thiết kế một mã nhị phân cho nó?
    1. Bạn có thể thiết kế một mã tốt hơn không? Gợi ý: điều gì xảy ra nếu ta cố mã hóa hai quan sát độc lập? Nếu ta mã hóa chung $n$ quan sát thì sao?
1. Khi mã hóa các tín hiệu được truyền qua dây vật lý, các kỹ sư không phải lúc nào cũng dùng mã nhị phân. Ví dụ, [PAM-3](https://en.wikipedia.org/wiki/Ternary_signal) sử dụng ba mức tín hiệu $\{-1, 0, 1\}$ thay vì hai mức $\{0, 1\}$. Bạn cần bao nhiêu đơn vị tam phân để truyền một số nguyên trong phạm vi $\{0, \ldots, 7\}$? Vì sao đây có thể là một ý tưởng tốt hơn xét về mặt điện tử?
1. [Mô hình Bradley--Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) sử dụng
một mô hình logistic để biểu diễn sở thích. Để một người dùng chọn giữa táo và cam, ta
giả định các điểm số $o_{\textrm{apple}}$ và $o_{\textrm{orange}}$. Các yêu cầu của ta là điểm số lớn hơn phải dẫn đến xác suất cao hơn khi chọn mục tương ứng và
mục có điểm số lớn nhất là mục có khả năng được chọn cao nhất :cite:`Bradley.Terry.1952`.
    1. Chứng minh rằng softmax thỏa mãn yêu cầu này.
    1. Điều gì xảy ra nếu bạn muốn cho phép một lựa chọn mặc định là không chọn táo lẫn cam? Gợi ý: bây giờ người dùng có ba lựa chọn.
1. Softmax có tên gọi từ ánh xạ sau: $\textrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    1. Chứng minh rằng $\textrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    1. Bạn có thể làm cho chênh lệch giữa hai hàm nhỏ đến mức nào? Gợi ý: không mất tính tổng quát, bạn có thể đặt $b = 0$ và $a \geq b$.
    1. Chứng minh rằng điều này đúng với $\lambda^{-1} \textrm{RealSoftMax}(\lambda a, \lambda b)$, với điều kiện $\lambda > 0$.
    1. Chỉ ra rằng khi $\lambda \to \infty$, ta có $\lambda^{-1} \textrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    1. Xây dựng một hàm softmin tương tự.
    1. Mở rộng điều này cho nhiều hơn hai số.
1. Hàm $g(\mathbf{x}) \stackrel{\textrm{def}}{=} \log \sum_i \exp x_i$ đôi khi cũng được gọi là [hàm log-partition](https://en.wikipedia.org/wiki/Partition_function_(mathematics)).
    1. Chứng minh rằng hàm này là lồi. Gợi ý: để làm điều này, hãy sử dụng thực tế rằng đạo hàm bậc nhất tương ứng với các xác suất từ hàm softmax và chỉ ra rằng đạo hàm bậc hai là phương sai.
    1. Chỉ ra rằng $g$ bất biến theo phép tịnh tiến, tức là $g(\mathbf{x} + b) = g(\mathbf{x})$.
    1. Điều gì xảy ra nếu một số tọa độ $x_i$ rất lớn? Điều gì xảy ra nếu tất cả chúng đều rất nhỏ?
    1. Chỉ ra rằng nếu ta chọn $b = \mathrm{max}_i x_i$ thì ta thu được một cách triển khai ổn định về mặt số học.
1. Giả sử ta có một phân phối xác suất $P$. Giả sử ta chọn một phân phối khác $Q$ với $Q(i) \propto P(i)^\alpha$ cho $\alpha > 0$.
    1. Lựa chọn $\alpha$ nào tương ứng với việc tăng gấp đôi temperature? Lựa chọn nào tương ứng với việc giảm một nửa?
    1. Điều gì xảy ra nếu ta cho temperature tiến đến $0$?
    1. Điều gì xảy ra nếu ta cho temperature tiến đến $\infty$?

[Thảo luận](https://discuss.d2l.ai/t/46)