```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Hồi quy tuyến tính
:label:`sec_linear_regression`

Các bài toán *hồi quy* xuất hiện bất cứ khi nào chúng ta muốn dự đoán một giá trị số.
Các ví dụ phổ biến bao gồm dự đoán giá (nhà, cổ phiếu, v.v.),
dự đoán thời gian nằm viện (của bệnh nhân),
dự báo nhu cầu (đối với doanh số bán lẻ), cùng nhiều ví dụ khác.
Không phải mọi bài toán dự đoán đều là bài toán hồi quy cổ điển.
Ở phần sau, chúng ta sẽ giới thiệu các bài toán phân loại,
trong đó mục tiêu là dự đoán tư cách thành viên trong một tập hợp các danh mục.

Làm ví dụ xuyên suốt, giả sử rằng chúng ta muốn
ước tính giá nhà (tính bằng đô la)
dựa trên diện tích (tính bằng feet vuông) và tuổi nhà (tính bằng năm).
Để phát triển một mô hình dự đoán giá nhà,
chúng ta cần có dữ liệu,
bao gồm giá bán, diện tích và tuổi của mỗi căn nhà.
Theo thuật ngữ machine learning,
tập dữ liệu này được gọi là *training dataset* hoặc *training set*,
và mỗi hàng (chứa dữ liệu tương ứng với một lần bán)
được gọi là một *example* (hoặc *data point*, *instance*, *sample*).
Đại lượng mà chúng ta cố gắng dự đoán (giá)
được gọi là *label* (hoặc *target*).
Các biến (tuổi và diện tích)
làm cơ sở cho các dự đoán
được gọi là *features* (hoặc *covariates*).

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from jax import numpy as jnp
import math
import time
```

## Kiến thức cơ bản

*Hồi quy tuyến tính* vừa là công cụ đơn giản nhất
vừa là công cụ phổ biến nhất trong các công cụ tiêu chuẩn
để giải quyết các bài toán hồi quy.
Có nguồn gốc từ đầu thế kỷ 19 :cite:`Legendre.1805,Gauss.1809`,
hồi quy tuyến tính bắt nguồn từ một vài giả định đơn giản.
Trước hết, chúng ta giả định rằng mối quan hệ
giữa features $\mathbf{x}$ và target $y$
xấp xỉ tuyến tính,
nghĩa là kỳ vọng có điều kiện $E[Y \mid X=\mathbf{x}]$
có thể được biểu diễn dưới dạng tổng có trọng số
của các features $\mathbf{x}$.
Thiết lập này cho phép giá trị target
vẫn có thể lệch khỏi giá trị kỳ vọng của nó
do nhiễu quan sát.
Tiếp theo, chúng ta có thể áp đặt giả định rằng mọi nhiễu như vậy
đều có tính chất tốt, tuân theo phân phối Gaussian.
Thông thường, chúng ta sẽ dùng $n$ để biểu thị
số lượng example trong tập dữ liệu.
Chúng ta dùng chỉ số trên để đánh số các sample và target,
còn chỉ số dưới để lập chỉ mục các tọa độ.
Cụ thể hơn,
$\mathbf{x}^{(i)}$ biểu thị sample thứ $i^{\textrm{th}}$
và $x_j^{(i)}$ biểu thị tọa độ thứ $j^{\textrm{th}}$ của nó.

### Mô hình
:label:`subsec_linear_model`

Cốt lõi của mọi lời giải là một mô hình
mô tả cách features có thể được biến đổi
thành một ước lượng của target.
Giả định về tính tuyến tính có nghĩa là
giá trị kỳ vọng của target (giá) có thể được biểu diễn
dưới dạng tổng có trọng số của các features (diện tích và tuổi):

$$\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b.$$
:eqlabel:`eq_price-area`

Ở đây, $w_{\textrm{area}}$ và $w_{\textrm{age}}$
được gọi là *weights*, còn $b$ được gọi là *bias*
(hoặc *offset* hay *intercept*).
Các weights xác định mức độ ảnh hưởng của từng feature đến dự đoán của chúng ta.
Bias xác định giá trị của ước lượng khi mọi feature đều bằng không.
Mặc dù chúng ta sẽ không bao giờ thấy một ngôi nhà mới xây có diện tích chính xác bằng không,
chúng ta vẫn cần bias vì nó cho phép
biểu diễn mọi hàm tuyến tính của features
(thay vì giới hạn chúng ta ở các đường thẳng đi qua gốc tọa độ).
Nói một cách chặt chẽ, :eqref:`eq_price-area` là một *biến đổi affine* của input features, được đặc trưng bởi một *biến đổi tuyến tính* của features thông qua tổng có trọng số, kết hợp với một *phép tịnh tiến* thông qua bias được thêm vào.
Cho trước một tập dữ liệu, mục tiêu của chúng ta là chọn
các weights $\mathbf{w}$ và bias $b$
sao cho, trung bình,
các dự đoán của mô hình khớp với giá thực tế quan sát được trong dữ liệu
gần nhất có thể.


Trong các lĩnh vực thường tập trung
vào những tập dữ liệu chỉ có vài features,
việc biểu diễn tường minh mô hình ở dạng đầy đủ,
như trong :eqref:`eq_price-area`, là phổ biến.
Trong machine learning, chúng ta thường làm việc
với các tập dữ liệu nhiều chiều,
nơi việc sử dụng
ký hiệu đại số tuyến tính cô đọng sẽ thuận tiện hơn.
Khi input của chúng ta gồm $d$ features,
chúng ta có thể gán cho mỗi feature một chỉ mục (từ $1$ đến $d$)
và biểu diễn dự đoán $\hat{y}$
(nói chung, ký hiệu "mũ" biểu thị một ước lượng) là

$$\hat{y} = w_1  x_1 + \cdots + w_d  x_d + b.$$

Gom tất cả features vào một vector $\mathbf{x} \in \mathbb{R}^d$
và tất cả weights vào một vector $\mathbf{w} \in \mathbb{R}^d$,
chúng ta có thể biểu diễn mô hình một cách cô đọng thông qua tích vô hướng
giữa $\mathbf{w}$ và $\mathbf{x}$:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

Trong :eqref:`eq_linreg-y`, vector $\mathbf{x}$
tương ứng với features của một example đơn lẻ.
Chúng ta thường thấy thuận tiện
khi tham chiếu đến features của toàn bộ tập dữ liệu gồm $n$ examples
thông qua *design matrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$.
Ở đây, $\mathbf{X}$ chứa một hàng cho mỗi example
và một cột cho mỗi feature.
Với một tập hợp features $\mathbf{X}$,
các dự đoán $\hat{\mathbf{y}} \in \mathbb{R}^n$
có thể được biểu diễn thông qua phép nhân ma trận--vector:

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$
:eqlabel:`eq_linreg-y-vec`

trong đó broadcasting (:numref:`subsec_broadcasting`) được áp dụng trong quá trình cộng.
Cho features của một training dataset $\mathbf{X}$
và các labels tương ứng (đã biết) $\mathbf{y}$,
mục tiêu của hồi quy tuyến tính là tìm
vector weight $\mathbf{w}$ và hạng tử bias $b$
sao cho, với features của một example dữ liệu mới
được lấy mẫu từ cùng phân phối với $\mathbf{X}$,
label của example mới sẽ được dự đoán (theo kỳ vọng)
với sai số nhỏ nhất.

Ngay cả khi chúng ta tin rằng mô hình tốt nhất để
dự đoán $y$ từ $\mathbf{x}$ là tuyến tính,
chúng ta cũng không kỳ vọng tìm được một tập dữ liệu thực tế gồm $n$ examples mà
$y^{(i)}$ bằng chính xác $\mathbf{w}^\top \mathbf{x}^{(i)}+b$
với mọi $1 \leq i \leq n$.
Ví dụ, bất kể chúng ta dùng công cụ nào để quan sát
features $\mathbf{X}$ và labels $\mathbf{y}$, vẫn có thể tồn tại một lượng nhỏ sai số đo lường.
Do đó, ngay cả khi chúng ta tin chắc
rằng mối quan hệ nền tảng là tuyến tính,
chúng ta vẫn sẽ đưa vào một hạng tử nhiễu để giải thích các sai số đó.

Trước khi có thể bắt đầu tìm kiếm *parameters*
(hoặc *model parameters*) $\mathbf{w}$ và $b$ tốt nhất,
chúng ta sẽ cần thêm hai điều:
(i) một thước đo chất lượng của một mô hình cho trước;
và (ii) một quy trình cập nhật mô hình để cải thiện chất lượng của nó.

### Hàm mất mát
:label:`subsec_linear-regression-loss-function`

Một cách tự nhiên, việc khớp mô hình với dữ liệu đòi hỏi
chúng ta phải thống nhất một thước đo về *độ phù hợp*
(hoặc tương đương là *độ không phù hợp*).
*Hàm mất mát* định lượng khoảng cách
giữa các giá trị target *thực tế* và *dự đoán*.
Mất mát thường là một số không âm,
trong đó giá trị càng nhỏ càng tốt
và dự đoán hoàn hảo có mất mát bằng 0.
Đối với các bài toán hồi quy, hàm mất mát phổ biến nhất là sai số bình phương.
Khi dự đoán của chúng ta cho example $i$ là $\hat{y}^{(i)}$
và label thực tương ứng là $y^{(i)}$,
*sai số bình phương* được cho bởi:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

Hằng số $\frac{1}{2}$ không tạo ra khác biệt thực sự
nhưng thuận tiện về mặt ký hiệu,
vì nó triệt tiêu khi chúng ta lấy đạo hàm của mất mát.
Vì training dataset được cung cấp cho chúng ta,
do đó nằm ngoài khả năng kiểm soát của chúng ta,
sai số thực nghiệm chỉ là một hàm của các model parameters.
Trong :numref:`fig_fit_linreg`, chúng ta trực quan hóa độ khớp của một mô hình hồi quy tuyến tính
trong một bài toán có input một chiều.

![Khớp một mô hình hồi quy tuyến tính với dữ liệu một chiều.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

Lưu ý rằng các khác biệt lớn giữa
ước lượng $\hat{y}^{(i)}$ và target $y^{(i)}$
dẫn đến đóng góp vào mất mát còn lớn hơn,
do dạng bậc hai của nó
(tính bậc hai này có thể là con dao hai lưỡi; tuy khuyến khích mô hình tránh các sai số lớn,
nó cũng có thể dẫn đến độ nhạy quá mức với dữ liệu bất thường).
Để đo chất lượng của một mô hình trên toàn bộ tập dữ liệu gồm $n$ examples,
chúng ta chỉ cần lấy trung bình (hoặc tương đương, tính tổng)
các mất mát trên training set:

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Khi huấn luyện mô hình, chúng ta tìm các parameters ($\mathbf{w}^*, b^*$)
làm nhỏ nhất tổng mất mát trên tất cả các training examples:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Nghiệm giải tích

Không giống hầu hết các mô hình mà chúng ta sẽ đề cập,
hồi quy tuyến tính đặt ra cho chúng ta
một bài toán tối ưu hóa dễ đến đáng ngạc nhiên.
Cụ thể, chúng ta có thể tìm các tham số tối ưu
(được đánh giá trên dữ liệu huấn luyện)
bằng phương pháp giải tích, áp dụng công thức đơn giản sau.
Trước hết, chúng ta có thể gộp bias $b$ vào tham số $\mathbf{w}$
bằng cách thêm một cột gồm toàn số 1 vào ma trận thiết kế.
Khi đó, bài toán dự đoán của chúng ta là tối thiểu hóa $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$.
Miễn là ma trận thiết kế $\mathbf{X}$ có đầy đủ hạng
(không feature nào phụ thuộc tuyến tính vào các feature khác),
sẽ chỉ có một điểm tới hạn trên bề mặt loss
và nó tương ứng với giá trị nhỏ nhất của loss trên toàn bộ miền.
Lấy đạo hàm của loss theo $\mathbf{w}$
và đặt nó bằng không, ta thu được:

$$\begin{aligned}
    \partial_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 =
    2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0
    \textrm{ and hence }
    \mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}.
\end{aligned}$$

Giải theo $\mathbf{w}$ cho chúng ta nghiệm tối ưu
của bài toán tối ưu hóa.
Lưu ý rằng nghiệm này

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$$

sẽ chỉ là duy nhất
khi ma trận $\mathbf X^\top \mathbf X$ khả nghịch,
tức là khi các cột của ma trận thiết kế
độc lập tuyến tính :cite:`Golub.Van-Loan.1996`.



Mặc dù các bài toán đơn giản như hồi quy tuyến tính
có thể có nghiệm giải tích,
bạn không nên quen với may mắn như vậy.
Dù các nghiệm giải tích cho phép thực hiện phân tích toán học rõ ràng,
yêu cầu phải có nghiệm giải tích quá hạn chế
đến mức sẽ loại trừ gần như mọi khía cạnh thú vị của deep learning.

### Gradient Descent Ngẫu Nhiên Theo Minibatch

May mắn thay, ngay cả trong các trường hợp chúng ta không thể giải mô hình bằng phương pháp giải tích,
chúng ta vẫn thường có thể huấn luyện mô hình hiệu quả trong thực tế.
Hơn nữa, đối với nhiều tác vụ, những mô hình khó tối ưu hóa đó
lại tốt hơn rất nhiều, nên việc tìm cách huấn luyện chúng
hoàn toàn xứng đáng với công sức bỏ ra.

Kỹ thuật then chốt để tối ưu hóa gần như mọi mô hình deep learning,
và là kỹ thuật chúng ta sẽ sử dụng xuyên suốt cuốn sách này,
là giảm lỗi lặp đi lặp lại
bằng cách cập nhật các tham số theo hướng
làm giảm dần hàm loss.
Thuật toán này được gọi là *gradient descent*.

Cách áp dụng gradient descent đơn giản nhất
là lấy đạo hàm của hàm loss,
vốn là trung bình các loss được tính
trên từng ví dụ trong toàn bộ dataset.
Trên thực tế, điều này có thể cực kỳ chậm:
chúng ta phải duyệt qua toàn bộ dataset trước khi thực hiện chỉ một lần cập nhật,
ngay cả khi các bước cập nhật có thể rất mạnh :cite:`Liu.Nocedal.1989`.
Tệ hơn nữa, nếu dữ liệu huấn luyện có nhiều phần dư thừa,
lợi ích của một lần cập nhật trên toàn bộ dữ liệu là hạn chế.

Ở thái cực còn lại, ta chỉ xét một ví dụ tại một thời điểm và thực hiện
các bước cập nhật dựa trên từng quan sát.
Thuật toán thu được, *stochastic gradient descent* (SGD),
có thể là một chiến lược hiệu quả :cite:`Bottou.2010`, ngay cả với các dataset lớn.
Đáng tiếc, SGD có các nhược điểm, cả về tính toán lẫn thống kê.
Một vấn đề xuất phát từ thực tế là bộ xử lý nhanh hơn rất nhiều
trong việc nhân và cộng các số so với
việc chuyển dữ liệu từ bộ nhớ chính sang cache của bộ xử lý.
Thực hiện một phép nhân ma trận--vector hiệu quả hơn
tới một bậc độ lớn so với thực hiện số lượng tương ứng các phép toán vector--vector.
Điều này có nghĩa là việc xử lý
từng sample một có thể mất nhiều thời gian hơn so với một batch đầy đủ.
Vấn đề thứ hai là một số layer,
chẳng hạn như batch normalization (sẽ được mô tả trong :numref:`sec_batch_norm`),
chỉ hoạt động tốt khi chúng ta có thể truy cập
nhiều hơn một quan sát tại một thời điểm.

Giải pháp cho cả hai vấn đề là chọn một chiến lược trung gian:
thay vì lấy một batch đầy đủ hoặc chỉ một sample tại một thời điểm,
chúng ta lấy một *minibatch* các quan sát :cite:`Li.Zhang.Chen.ea.2014`.
Việc chọn kích thước cụ thể của minibatch này phụ thuộc vào nhiều yếu tố,
chẳng hạn như lượng bộ nhớ, số lượng accelerator,
lựa chọn layer, và tổng kích thước dataset.
Dù vậy, một số từ 32 đến 256,
tốt nhất là bội số của một lũy thừa lớn của $2$, là một khởi đầu tốt.
Điều này dẫn chúng ta đến *minibatch stochastic gradient descent*.

Ở dạng cơ bản nhất, trong mỗi lần lặp $t$,
trước tiên chúng ta lấy mẫu ngẫu nhiên một minibatch $\mathcal{B}_t$
gồm một số lượng cố định $|\mathcal{B}|$ ví dụ huấn luyện.
Sau đó, chúng ta tính đạo hàm (gradient) của loss trung bình
trên minibatch theo các tham số mô hình.
Cuối cùng, chúng ta nhân gradient
với một giá trị dương nhỏ $\eta$ được xác định trước,
gọi là *learning rate*,
và trừ hạng tử thu được khỏi các giá trị tham số hiện tại.
Ta có thể biểu diễn phép cập nhật như sau:

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

Tóm lại, minibatch SGD tiến hành như sau:
(i) khởi tạo các giá trị của tham số mô hình, thường là ngẫu nhiên;
(ii) lặp lại việc lấy mẫu các minibatch ngẫu nhiên từ dữ liệu,
cập nhật các tham số theo hướng gradient âm.
Với loss bậc hai và phép biến đổi affine,
điều này có khai triển dạng đóng:

$$\begin{aligned} \mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_b l^{(i)}(\mathbf{w}, b) &&  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Vì chúng ta chọn một minibatch $\mathcal{B}$
nên cần chuẩn hóa theo kích thước của nó $|\mathcal{B}|$.
Kích thước minibatch và learning rate thường do người dùng xác định.
Những tham số có thể điều chỉnh mà không được cập nhật
trong vòng lặp huấn luyện được gọi là *hyperparameters*.
Chúng có thể được tinh chỉnh tự động bằng nhiều kỹ thuật, chẳng hạn như Bayesian optimization
:cite:`Frazier.2018`. Cuối cùng, chất lượng của nghiệm
thường được đánh giá trên một *validation dataset* (hoặc *validation set*) riêng biệt.

Sau khi huấn luyện trong một số lần lặp được xác định trước
(hoặc đến khi thỏa một tiêu chí dừng khác),
chúng ta lưu các tham số mô hình ước lượng,
được ký hiệu là $\hat{\mathbf{w}}, \hat{b}$.
Lưu ý rằng ngay cả khi hàm của chúng ta thực sự tuyến tính và không có nhiễu,
các tham số này sẽ không phải là các giá trị tối thiểu chính xác của loss, thậm chí cũng không có tính xác định.
Mặc dù thuật toán hội tụ chậm về phía các giá trị tối thiểu,
nó thường sẽ không tìm được chúng chính xác trong một số bước hữu hạn.
Hơn nữa, các minibatch $\mathcal{B}$
được dùng để cập nhật tham số được chọn ngẫu nhiên.
Điều này phá vỡ tính xác định.

Hồi quy tuyến tính là một bài toán học
có cực tiểu toàn cục
(bất cứ khi nào $\mathbf{X}$ có đầy đủ hạng, hay tương đương,
bất cứ khi nào $\mathbf{X}^\top \mathbf{X}$ khả nghịch).
Tuy nhiên, các bề mặt loss của mạng sâu chứa nhiều điểm yên ngựa và cực tiểu.
May mắn thay, chúng ta thường không quan tâm đến việc tìm
một tập tham số chính xác mà chỉ cần bất kỳ tập tham số nào
dẫn đến các dự đoán chính xác (và do đó có loss thấp).
Trên thực tế, các practitioner deep learning
hiếm khi gặp khó khăn trong việc tìm các tham số
tối thiểu hóa loss *trên các tập huấn luyện*
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018,Frankle.Carbin.2018`.
Nhiệm vụ khó khăn hơn là tìm các tham số
dẫn đến dự đoán chính xác trên dữ liệu chưa từng thấy trước đó,
một thách thức được gọi là *generalization*.
Chúng ta sẽ quay lại các chủ đề này xuyên suốt cuốn sách.

### Dự đoán

Với mô hình $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$,
giờ đây chúng ta có thể đưa ra *dự đoán* cho một ví dụ mới,
chẳng hạn dự đoán giá bán của một ngôi nhà chưa từng thấy trước đó
dựa trên diện tích $x_1$ và tuổi nhà $x_2$.
Những người thực hành deep learning thường gọi giai đoạn dự đoán là *suy luận*
(inference), nhưng đây có phần là một cách gọi không chính xác---*suy luận* nói chung
đề cập đến bất kỳ kết luận nào được rút ra dựa trên bằng chứng,
bao gồm cả giá trị của các tham số
và nhãn có khả năng xảy ra cho một mẫu chưa thấy.
Nếu có gì khác biệt, trong tài liệu thống kê,
*inference* thường chỉ suy luận tham số hơn,
và việc dùng quá tải thuật ngữ này gây ra sự nhầm lẫn không cần thiết
khi những người thực hành deep learning trao đổi với các nhà thống kê.
Trong phần sau, chúng ta sẽ dùng *dự đoán* bất cứ khi nào có thể.



## Vector hóa để Tăng tốc

Khi huấn luyện mô hình, chúng ta thường muốn xử lý
toàn bộ minibatch các ví dụ cùng lúc.
Để thực hiện hiệu quả, (**chúng ta**) (~~nên~~)
(**vector hóa các phép tính và tận dụng
các thư viện đại số tuyến tính tốc độ cao
thay vì viết các vòng lặp for tốn kém trong Python.**)

Để thấy vì sao điều này lại quan trọng đến vậy,
hãy (**xét hai phương pháp cộng vector.**)
Trước tiên, chúng ta khởi tạo hai vector 10.000 chiều
có tất cả phần tử bằng 1.
Trong phương pháp đầu tiên, chúng ta lặp qua các vector bằng vòng lặp for của Python.
Trong phương pháp thứ hai, chúng ta dựa vào một lệnh gọi duy nhất tới `+`.

```{.python .input}
%%tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Bây giờ chúng ta có thể benchmark các tác vụ.
Trước hết, [**chúng ta cộng chúng, từng tọa độ một,
bằng vòng lặp for.**]

```{.python .input}
%%tab mxnet, pytorch
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab tensorflow
c = tf.Variable(d2l.zeros(n))
t = time.time()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab jax
# JAX arrays are immutable, meaning that once created their contents
# cannot be changed. For updating individual elements, JAX provides
# an indexed update syntax that returns an updated copy
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c = c.at[i].set(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

(**Ngoài ra, chúng ta dựa vào toán tử `+` đã được nạp chồng để tính tổng theo từng phần tử.**)

```{.python .input}
%%tab all
t = time.time()
d = a + b
f'{time.time() - t:.5f} sec'
```

Phương pháp thứ hai nhanh hơn đáng kể so với phương pháp thứ nhất.
Vector hóa mã thường mang lại mức tăng tốc theo bậc độ lớn.
Hơn nữa, chúng ta chuyển nhiều phần toán học hơn cho thư viện
nên không phải tự viết quá nhiều phép tính,
giảm khả năng xảy ra lỗi và tăng tính di động của mã.


## Phân phối Chuẩn và Squared Loss
:label:`subsec_normal_distribution_and_squared_loss`

Cho đến nay, chúng ta đã đưa ra một động cơ khá thực dụng
cho mục tiêu squared loss:
các tham số tối ưu trả về kỳ vọng có điều kiện $E[Y\mid X]$
bất cứ khi nào quy luật cơ bản thực sự là tuyến tính,
và hàm mất mát áp dụng hình phạt lớn cho các ngoại lệ.
Chúng ta cũng có thể đưa ra một động cơ chính quy hơn
cho mục tiêu squared loss
bằng cách đưa ra các giả định xác suất
về phân phối của nhiễu.

Hồi quy tuyến tính được phát minh vào đầu thế kỷ 19.
Dù từ lâu vẫn còn tranh luận liệu Gauss hay Legendre
là người đầu tiên nảy ra ý tưởng này,
Gauss cũng là người đã phát hiện ra phân phối chuẩn
(còn được gọi là *Gaussian*).
Hóa ra phân phối chuẩn
và hồi quy tuyến tính với squared loss
có một mối liên hệ sâu sắc hơn là cùng nguồn gốc.

Để bắt đầu, hãy nhớ rằng một phân phối chuẩn
có trung bình $\mu$ và phương sai $\sigma^2$ (độ lệch chuẩn $\sigma$)
được cho bởi

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Bên dưới, [**chúng ta định nghĩa một hàm để tính phân phối chuẩn**].

```{.python .input}
%%tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    if tab.selected('jax'):
        return p * jnp.exp(-0.5 * (x - mu)**2 / sigma**2)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
```

Giờ đây chúng ta có thể (**trực quan hóa các phân phối chuẩn**).

```{.python .input}
%%tab mxnet
# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.python .input}

%%tab pytorch, tensorflow, jax
if tab.selected('jax'):
    # Use JAX NumPy for visualization
    x = jnp.arange(-7, 7, 0.01)
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    # Use NumPy again for visualization
    x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

Lưu ý rằng việc thay đổi giá trị trung bình tương ứng
với sự dịch chuyển theo trục $x$,
còn việc tăng phương sai
làm phân phối trải rộng hơn,
hạ thấp đỉnh của nó.

Một cách để lý giải hồi quy tuyến tính với squared loss
là giả định rằng các quan sát phát sinh từ các phép đo có nhiễu,
trong đó nhiễu $\epsilon$ tuân theo phân phối chuẩn
$\mathcal{N}(0, \sigma^2)$:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \textrm{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Do đó, giờ đây chúng ta có thể viết *likelihood*
quan sát một $y$ cụ thể với $\mathbf{x}$ đã cho như sau

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Như vậy, likelihood có thể được phân tích thành nhân tử.
Theo *nguyên lý hợp lý cực đại*,
các giá trị tốt nhất của tham số $\mathbf{w}$ và $b$ là những giá trị
tối đa hóa *likelihood* của toàn bộ tập dữ liệu:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)} \mid \mathbf{x}^{(i)}).$$

Đẳng thức này đúng vì mọi cặp $(\mathbf{x}^{(i)}, y^{(i)})$
được lấy mẫu độc lập với nhau.
Các estimator được chọn theo nguyên lý hợp lý cực đại
được gọi là *maximum likelihood estimators*.
Mặc dù việc tối đa hóa tích của nhiều hàm mũ
có thể trông khó khăn,
chúng ta có thể đơn giản hóa đáng kể mà không thay đổi mục tiêu
bằng cách tối đa hóa logarit của likelihood thay thế.
Vì các lý do lịch sử, bài toán tối ưu thường được biểu diễn
dưới dạng tối thiểu hóa thay vì tối đa hóa.
Vì vậy, không thay đổi điều gì,
chúng ta có thể *tối thiểu hóa* *negative log-likelihood*,
được biểu diễn như sau:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Nếu giả định $\sigma$ là cố định,
chúng ta có thể bỏ qua hạng tử đầu tiên,
vì nó không phụ thuộc vào $\mathbf{w}$ hoặc $b$.
Hạng tử thứ hai giống hệt
hàm squared error loss đã được giới thiệu trước đó,
ngoại trừ hằng số nhân $\frac{1}{\sigma^2}$.
May mắn thay, nghiệm cũng không phụ thuộc vào $\sigma$.
Suy ra rằng việc tối thiểu hóa mean squared error
tương đương với ước lượng hợp lý cực đại
của một mô hình tuyến tính dưới giả định nhiễu Gaussian cộng tính.


## Hồi quy Tuyến tính như một Neural Network

Mặc dù các mô hình tuyến tính không đủ phong phú
để biểu diễn nhiều mạng phức tạp
mà chúng ta sẽ giới thiệu trong cuốn sách này,
các neural network (nhân tạo) đủ phong phú
để bao hàm các mô hình tuyến tính như những mạng
trong đó mọi đặc trưng được biểu diễn bởi một neuron đầu vào,
và tất cả chúng đều được kết nối trực tiếp với đầu ra.

:numref:`fig_single_neuron` minh họa
hồi quy tuyến tính như một neural network.
Sơ đồ làm nổi bật mẫu kết nối,
chẳng hạn cách mỗi đầu vào được kết nối với đầu ra,
nhưng không thể hiện các giá trị cụ thể của trọng số hoặc bias.

![Hồi quy tuyến tính là một neural network một tầng.](../img/singleneuron.svg)
:label:`fig_single_neuron`

Các đầu vào là $x_1, \ldots, x_d$.
Chúng ta gọi $d$ là *số lượng đầu vào*
hoặc *số chiều đặc trưng* trong tầng đầu vào.
Đầu ra của mạng là $o_1$.
Vì chúng ta chỉ đang cố gắng dự đoán
một giá trị số duy nhất,
nên chỉ có một neuron đầu ra.
Lưu ý rằng các giá trị đầu vào đều được *cho trước*.
Chỉ có một neuron được *tính toán*.
Tóm lại, chúng ta có thể xem hồi quy tuyến tính
là một neural network fully connected một tầng.
Chúng ta sẽ gặp các mạng
có nhiều tầng hơn rất nhiều
trong các chương sau.

### Sinh học

Vì hồi quy tuyến tính có trước khoa học thần kinh tính toán,
việc mô tả hồi quy tuyến tính
bằng các thuật ngữ neural network có thể có vẻ không đúng thời đại.
Tuy nhiên, chúng là điểm khởi đầu tự nhiên
khi các nhà điều khiển học và sinh lý thần kinh
Warren McCulloch và Walter Pitts bắt đầu phát triển
các mô hình neuron nhân tạo.
Hãy xét hình minh họa đơn giản
về một neuron sinh học trong :numref:`fig_Neuron`,
bao gồm *dendrite* (đầu cuối đầu vào),
*nucleus* (CPU), *axon* (dây đầu ra),
và *axon terminals* (đầu cuối đầu ra),
cho phép kết nối với các neuron khác qua *synapse*.

![Neuron thực (nguồn: "Anatomy and Physiology" của Chương trình Surveillance, Epidemiology and End Results (SEER) thuộc Viện Ung thư Quốc gia Hoa Kỳ).](../img/neuron.svg)
:label:`fig_Neuron`

Thông tin $x_i$ đến từ các neuron khác
(hoặc cảm biến môi trường) được tiếp nhận tại dendrite.
Cụ thể, thông tin đó được gán trọng số
bởi *synaptic weights* $w_i$,
quyết định tác động của các đầu vào,
ví dụ kích hoạt hoặc ức chế thông qua tích $x_i w_i$.
Các đầu vào có trọng số đến từ nhiều nguồn
được tổng hợp trong nucleus
thành tổng có trọng số $y = \sum_i x_i w_i + b$,
có thể trải qua một bước hậu xử lý phi tuyến bằng hàm $\sigma(y)$.
Thông tin này sau đó được gửi qua axon đến axon terminals,
nơi nó đến đích
(ví dụ, một cơ cấu chấp hành như cơ)
hoặc được đưa vào neuron khác thông qua dendrite của neuron đó.

Chắc chắn, ý tưởng ở cấp độ cao rằng nhiều đơn vị như vậy
có thể được kết hợp, miễn là chúng có kết nối và thuật toán học phù hợp,
để tạo ra hành vi thú vị và phức tạp hơn nhiều
so với những gì một neuron đơn lẻ có thể biểu đạt
bắt nguồn từ nghiên cứu của chúng ta về các hệ thần kinh sinh học thực.
Đồng thời, phần lớn nghiên cứu deep learning ngày nay
lấy cảm hứng từ nhiều nguồn rộng hơn.
Chúng ta dẫn lại :citet:`Russell.Norvig.2016`
người đã chỉ ra rằng mặc dù máy bay có thể được *lấy cảm hứng* từ chim,
ngành điểu học đã không còn là động lực chính
cho đổi mới trong hàng không học trong nhiều thế kỷ.
Tương tự, cảm hứng trong deep learning ngày nay
đến ở mức độ tương đương hoặc lớn hơn
từ toán học, ngôn ngữ học, tâm lý học,
thống kê, khoa học máy tính và nhiều lĩnh vực khác.

## Tóm tắt

Trong phần này, chúng ta đã giới thiệu
hồi quy tuyến tính truyền thống,
trong đó các tham số của một hàm tuyến tính
được chọn để tối thiểu hóa squared loss trên tập huấn luyện.
Chúng ta cũng đã lý giải lựa chọn hàm mục tiêu này
dựa trên một số cân nhắc thực tế
và thông qua cách diễn giải
hồi quy tuyến tính là ước lượng hợp lý cực đại
dưới giả định về tính tuyến tính và nhiễu Gaussian.
Sau khi thảo luận cả các cân nhắc về tính toán
lẫn các mối liên hệ với thống kê,
chúng ta đã chỉ ra cách các mô hình tuyến tính như vậy có thể được biểu diễn
dưới dạng các neural network đơn giản, trong đó đầu vào
được kết nối trực tiếp với đầu ra.
Mặc dù chúng ta sẽ sớm chuyển hoàn toàn sang các mô hình phi tuyến,
chúng đủ để giới thiệu hầu hết các thành phần
mà mọi mô hình của chúng ta đều cần:
các dạng tham số, mục tiêu khả vi,
tối ưu hóa bằng minibatch stochastic gradient descent,
và cuối cùng là đánh giá trên dữ liệu chưa từng thấy trước đó.



## Bài tập

1. Giả sử rằng chúng ta có dữ liệu $x_1, \ldots, x_n \in \mathbb{R}$. Mục tiêu của chúng ta là tìm một hằng số $b$ sao cho $\sum_i (x_i - b)^2$ được tối thiểu hóa.
    1. Tìm nghiệm giải tích cho giá trị tối ưu của $b$.
    1. Bài toán này và nghiệm của nó liên hệ như thế nào với phân phối chuẩn?
    1. Điều gì xảy ra nếu chúng ta thay đổi loss từ $\sum_i (x_i - b)^2$ thành $\sum_i |x_i-b|$? Bạn có thể tìm nghiệm tối ưu cho $b$ không?
1. Chứng minh rằng các hàm affine có thể biểu diễn bởi $\mathbf{x}^\top \mathbf{w} + b$ tương đương với các hàm tuyến tính trên $(\mathbf{x}, 1)$.
1. Giả sử rằng bạn muốn tìm các hàm bậc hai của $\mathbf{x}$, tức là $f(\mathbf{x}) = b + \sum_i w_i x_i + \sum_{j \leq i} w_{ij} x_{i} x_{j}$. Bạn sẽ biểu diễn điều này trong một deep network như thế nào?
1. Hãy nhớ rằng một trong các điều kiện để bài toán hồi quy tuyến tính có thể giải được là ma trận thiết kế $\mathbf{X}^\top \mathbf{X}$ có full rank.
    1. Điều gì xảy ra nếu không phải như vậy?
    1. Bạn có thể khắc phục điều đó như thế nào? Điều gì xảy ra nếu bạn thêm một lượng nhỏ nhiễu Gaussian độc lập theo từng tọa độ vào tất cả các phần tử của $\mathbf{X}$?
    1. Giá trị kỳ vọng của ma trận thiết kế $\mathbf{X}^\top \mathbf{X}$ trong trường hợp này là gì?
    1. Điều gì xảy ra với stochastic gradient descent khi $\mathbf{X}^\top \mathbf{X}$ không có full rank?
1. Giả sử rằng mô hình nhiễu chi phối nhiễu cộng $\epsilon$ là phân phối mũ. Tức là, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Viết negative log-likelihood của dữ liệu dưới mô hình $-\log P(\mathbf y \mid \mathbf X)$.
    1. Bạn có thể tìm một nghiệm dạng đóng không?
    1. Đề xuất một thuật toán minibatch stochastic gradient descent để giải bài toán này. Điều gì có thể xảy ra sai (gợi ý: điều gì xảy ra gần stationary point khi chúng ta tiếp tục cập nhật các tham số)? Bạn có thể khắc phục điều này không?
1. Giả sử rằng chúng ta muốn thiết kế một neural network có hai lớp bằng cách ghép hai lớp tuyến tính. Tức là, đầu ra của lớp thứ nhất trở thành đầu vào của lớp thứ hai. Tại sao phép ghép ngây thơ như vậy lại không hiệu quả?
1. Điều gì xảy ra nếu bạn muốn dùng hồi quy để ước lượng giá thực tế của nhà ở hoặc giá cổ phiếu?
    1. Chỉ ra rằng giả định nhiễu Gaussian cộng là không phù hợp. Gợi ý: liệu giá có thể âm không? Còn các biến động thì sao?
    1. Tại sao hồi quy theo logarithm của giá sẽ tốt hơn nhiều, tức là $y = \log \textrm{price}$?
    1. Bạn cần lo lắng về điều gì khi xử lý pennystock, tức là cổ phiếu có giá rất thấp? Gợi ý: bạn có thể giao dịch ở mọi mức giá có thể không? Tại sao đây là vấn đề lớn hơn đối với cổ phiếu rẻ? Để biết thêm thông tin, hãy xem lại mô hình Black--Scholes nổi tiếng về định giá quyền chọn :cite:`Black.Scholes.1973`.
1. Giả sử chúng ta muốn dùng hồi quy để ước lượng *số lượng* táo bán được trong một cửa hàng tạp hóa.
    1. Các vấn đề với mô hình nhiễu Gaussian cộng là gì? Gợi ý: bạn đang bán táo, không phải dầu mỏ.
    1. [Phân phối Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) mô tả các phân phối trên số đếm. Nó được cho bởi $p(k \mid \lambda) = \lambda^k e^{-\lambda}/k!$. Ở đây $\lambda$ là hàm tốc độ và $k$ là số sự kiện quan sát được. Chứng minh rằng $\lambda$ là giá trị kỳ vọng của số đếm $k$.
    1. Thiết kế một hàm loss liên quan đến phân phối Poisson.
    1. Thay vào đó, thiết kế một hàm loss để ước lượng $\log \lambda$.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/259)
:end_tab: