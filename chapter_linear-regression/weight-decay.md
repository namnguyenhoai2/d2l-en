```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Suy giảm trọng số
:label:`sec_weight_decay`

Giờ đây, sau khi đã mô tả vấn đề overfitting,
chúng ta có thể giới thiệu kỹ thuật *regularization* đầu tiên.
Hãy nhớ rằng chúng ta luôn có thể giảm thiểu overfitting
bằng cách thu thập thêm dữ liệu huấn luyện.
Tuy nhiên, việc đó có thể tốn kém, mất nhiều thời gian
hoặc hoàn toàn nằm ngoài tầm kiểm soát của chúng ta,
khiến điều này không thể thực hiện trong ngắn hạn.
Hiện tại, chúng ta có thể giả định rằng mình đã có
lượng dữ liệu chất lượng cao nhiều nhất trong phạm vi nguồn lực cho phép
và tập trung vào các công cụ sẵn có
khi xem dataset là một dữ kiện cố định.

Hãy nhớ rằng trong ví dụ hồi quy đa thức
(:numref:`subsec_polynomial-curve-fitting`)
chúng ta có thể giới hạn năng lực của model
bằng cách điều chỉnh bậc
của đa thức được khớp.
Thực tế, giới hạn số lượng feature
là một kỹ thuật phổ biến để giảm thiểu overfitting.
Tuy nhiên, việc đơn giản loại bỏ feature
có thể là một công cụ quá thô.
Tiếp tục với ví dụ hồi quy đa thức,
hãy xem điều gì có thể xảy ra
với input có số chiều cao.
Phần mở rộng tự nhiên của đa thức
cho dữ liệu đa biến được gọi là *monomial*,
vốn đơn giản là tích của các lũy thừa của các biến.
Bậc của một monomial là tổng các số mũ.
Ví dụ, $x_1^2 x_2$ và $x_3 x_5^2$
đều là các monomial bậc 3.

Lưu ý rằng số lượng số hạng có bậc $d$
tăng vọt nhanh chóng khi $d$ lớn hơn.
Với $k$ biến, số lượng monomial
bậc $d$ là ${k - 1 + d} \choose {k - 1}$.
Ngay cả những thay đổi nhỏ về bậc, chẳng hạn từ $2$ lên $3$,
cũng làm độ phức tạp của model tăng đáng kể.
Do đó, chúng ta thường cần một công cụ tinh chỉnh hơn
để điều chỉnh độ phức tạp của hàm.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
```

## Chuẩn và suy giảm trọng số

(**Thay vì trực tiếp thao tác với số lượng tham số,
*suy giảm trọng số* hoạt động bằng cách giới hạn các giá trị
mà tham số có thể nhận.**)
Bên ngoài lĩnh vực deep learning, kỹ thuật này thường được gọi là regularization $\ell_2$
khi được tối ưu bằng stochastic gradient descent theo minibatch,
và có thể là kỹ thuật được sử dụng rộng rãi nhất
để regularize các model machine learning tham số.
Kỹ thuật này dựa trên trực giác cơ bản rằng trong tất cả các hàm $f$,
hàm $f = 0$
(gán giá trị $0$ cho mọi input)
theo một nghĩa nào đó là *đơn giản nhất*,
và chúng ta có thể đo độ phức tạp
của một hàm bằng khoảng cách từ các tham số của nó đến zero.
Nhưng chính xác thì chúng ta nên đo
khoảng cách giữa một hàm và zero như thế nào?
Không có một câu trả lời duy nhất đúng.
Trên thực tế, toàn bộ các nhánh toán học,
bao gồm một phần của giải tích hàm
và lý thuyết không gian Banach,
được dành để giải quyết những vấn đề như vậy.

Một cách diễn giải đơn giản có thể là
đo độ phức tạp của một hàm tuyến tính
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
bằng một chuẩn nào đó của vector trọng số, chẳng hạn $\| \mathbf{w} \|^2$.
Hãy nhớ rằng chúng ta đã giới thiệu chuẩn $\ell_2$ và chuẩn $\ell_1$,
là các trường hợp đặc biệt của chuẩn $\ell_p$ tổng quát hơn,
trong :numref:`subsec_lin-algebra-norms`.
Phương pháp phổ biến nhất để đảm bảo vector trọng số nhỏ
là thêm chuẩn của nó dưới dạng một hạng tử phạt
vào bài toán cực tiểu hóa loss.
Vì vậy, chúng ta thay mục tiêu ban đầu,
*cực tiểu hóa loss dự đoán trên các label huấn luyện*,
bằng mục tiêu mới,
*cực tiểu hóa tổng của loss dự đoán và hạng tử phạt*.
Giờ đây, nếu vector trọng số tăng quá lớn,
thuật toán học của chúng ta có thể tập trung
vào việc cực tiểu hóa chuẩn trọng số $\| \mathbf{w} \|^2$
thay vì cực tiểu hóa lỗi huấn luyện.
Đó chính xác là điều chúng ta mong muốn.
Để minh họa bằng code,
chúng ta sử dụng lại ví dụ trước đây
từ :numref:`sec_linear_regression` về hồi quy tuyến tính.
Trong đó, loss của chúng ta được cho bởi

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Hãy nhớ rằng $\mathbf{x}^{(i)}$ là các feature,
$y^{(i)}$ là label của một ví dụ dữ liệu bất kỳ $i$, còn $(\mathbf{w}, b)$
lần lượt là các tham số trọng số và bias.
Để phạt kích thước của vector trọng số,
bằng cách nào đó chúng ta phải thêm $\| \mathbf{w} \|^2$ vào hàm loss,
nhưng model nên cân bằng loss tiêu chuẩn
với hạng tử phạt cộng thêm mới này như thế nào?
Trong thực tế, chúng ta mô tả sự cân bằng này
thông qua *hằng số regularization* $\lambda$,
một hyperparameter không âm
được điều chỉnh bằng dữ liệu validation:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


Với $\lambda = 0$, chúng ta thu lại hàm loss ban đầu.
Với $\lambda > 0$, chúng ta giới hạn kích thước của $\| \mathbf{w} \|$.
Theo quy ước, chúng ta chia cho $2$:
khi lấy đạo hàm của một hàm bậc hai,
$2$ và $1/2$ triệt tiêu lẫn nhau, đảm bảo biểu thức
cho phép cập nhật trông gọn gàng và đơn giản.
Một độc giả tinh ý có thể thắc mắc tại sao chúng ta dùng
chuẩn bình phương thay vì chuẩn thông thường
(tức là khoảng cách Euclidean).
Chúng ta làm vậy để thuận tiện cho việc tính toán.
Bằng cách bình phương chuẩn $\ell_2$, chúng ta loại bỏ căn bậc hai,
chỉ còn lại tổng bình phương
của từng thành phần trong vector trọng số.
Điều này giúp dễ dàng tính đạo hàm của hạng tử phạt:
tổng các đạo hàm bằng đạo hàm của tổng.

Hơn nữa, bạn có thể hỏi tại sao ngay từ đầu chúng ta dùng chuẩn $\ell_2$
chứ không phải, chẳng hạn, chuẩn $\ell_1$.
Thực tế, các lựa chọn khác cũng hợp lệ và
phổ biến trong lĩnh vực thống kê.
Trong khi các model tuyến tính regularize bằng $\ell_2$ tạo thành
thuật toán *ridge regression* kinh điển,
hồi quy tuyến tính regularize bằng $\ell_1$
cũng là một phương pháp nền tảng tương tự trong thống kê,
thường được gọi là *lasso regression*.
Một lý do để dùng chuẩn $\ell_2$
là nó áp dụng mức phạt lớn hơn
lên các thành phần lớn của vector trọng số.
Điều này định hướng thuật toán học của chúng ta
về các model phân bổ trọng số đồng đều
trên số lượng feature lớn hơn.
Trong thực tế, điều này có thể khiến chúng bền vững hơn
trước sai số đo lường trong một biến đơn lẻ.
Ngược lại, các hạng tử phạt $\ell_1$ dẫn đến những model
tập trung trọng số vào một tập feature nhỏ
bằng cách đưa các trọng số còn lại về zero.
Điều này cung cấp cho chúng ta một phương pháp hiệu quả để *feature selection*,
có thể hữu ích vì những lý do khác.
Ví dụ, nếu model của chúng ta chỉ dựa vào một vài feature,
thì chúng ta có thể không cần thu thập, lưu trữ hoặc truyền dữ liệu
cho các feature khác (đã bị loại bỏ).

Sử dụng cùng ký hiệu trong :eqref:`eq_linreg_batch_update`,
các bước cập nhật của stochastic gradient descent theo minibatch
cho hồi quy regularize bằng $\ell_2$ như sau:

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Như trước đây, chúng ta cập nhật $\mathbf{w}$ dựa trên mức độ
mà tại đó ước lượng của chúng ta khác với quan sát.
Tuy nhiên, chúng ta cũng thu nhỏ kích thước của $\mathbf{w}$ về zero.
Đó là lý do phương pháp này đôi khi được gọi là "suy giảm trọng số":
chỉ xét riêng hạng tử phạt,
thuật toán tối ưu hóa của chúng ta *làm suy giảm*
trọng số ở mỗi bước huấn luyện.
Không giống feature selection,
suy giảm trọng số cung cấp cho chúng ta một cơ chế điều chỉnh liên tục độ phức tạp của hàm.
Các giá trị $\lambda$ nhỏ hơn tương ứng
với $\mathbf{w}$ ít bị ràng buộc hơn,
trong khi các giá trị $\lambda$ lớn hơn
ràng buộc $\mathbf{w}$ đáng kể hơn.
Việc có đưa hạng tử phạt bias tương ứng $b^2$ vào hay không
có thể khác nhau giữa các implementation,
và có thể khác nhau giữa các layer của neural network.
Thông thường, chúng ta không regularize hạng tử bias.
Ngoài ra,
mặc dù regularization $\ell_2$ có thể không tương đương với suy giảm trọng số đối với các thuật toán tối ưu hóa khác,
ý tưởng regularization thông qua
việc thu nhỏ kích thước của trọng số
vẫn đúng.

## Hồi quy tuyến tính số chiều cao

Chúng ta có thể minh họa lợi ích của suy giảm trọng số
thông qua một ví dụ tổng hợp đơn giản.

Trước tiên, chúng ta [**tạo một số dữ liệu như trước**]:

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Trong dataset tổng hợp này, label của chúng ta được xác định
bởi một hàm tuyến tính nền tảng của các input,
bị nhiễu bởi nhiễu Gaussian
có trung bình bằng zero và độ lệch chuẩn 0.01.
Để minh họa,
chúng ta có thể làm cho ảnh hưởng của overfitting rõ rệt hơn
bằng cách tăng số chiều của bài toán lên $d = 200$
và làm việc với một tập huấn luyện nhỏ chỉ gồm 20 ví dụ.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Implementation từ đầu

Bây giờ, hãy thử triển khai suy giảm trọng số từ đầu.
Vì stochastic gradient descent theo minibatch
là optimizer của chúng ta,
chúng ta chỉ cần thêm hạng tử phạt $\ell_2$ bình phương
vào hàm loss ban đầu.

### (**Định nghĩa hạng tử phạt chuẩn $\ell_2$**)

Có lẽ cách thuận tiện nhất để triển khai hạng tử phạt này
là bình phương tất cả các hạng tử tại chỗ rồi tính tổng.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Định nghĩa Model

Trong model cuối cùng,
hồi quy tuyến tính và squared loss không thay đổi kể từ :numref:`sec_linear_scratch`,
vì vậy chúng ta chỉ cần định nghĩa một subclass của `d2l.LinearRegressionScratch`. Thay đổi duy nhất ở đây là loss của chúng ta hiện bao gồm hạng tử phạt.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

Đoạn code sau khớp model của chúng ta trên tập huấn luyện gồm 20 ví dụ và đánh giá nó trên tập validation gồm 100 ví dụ.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**Huấn luyện không có Regularization**]

Giờ đây, chúng ta chạy đoạn mã này với `lambd = 0`,  
tắt weight decay.  
Lưu ý rằng chúng ta bị overfit nghiêm trọng,  
làm giảm lỗi huấn luyện nhưng không giảm  
lỗi validation---một trường hợp overfitting điển hình.

```{.python .input}
%%tab all
train_scratch(0)
```

### [**Sử dụng Weight Decay**]

Dưới đây, chúng ta chạy với weight decay đáng kể.  
Lưu ý rằng lỗi huấn luyện tăng lên  
nhưng lỗi validation giảm xuống.  
Đây chính xác là tác động  
mà chúng ta mong đợi từ regularization.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**Triển khai ngắn gọn**]

Vì weight decay được sử dụng rộng rãi  
trong quá trình tối ưu hóa mạng neural,  
framework deep learning giúp việc này trở nên đặc biệt thuận tiện,  
tích hợp weight decay trực tiếp vào thuật toán tối ưu hóa  
để dễ dàng sử dụng cùng với mọi hàm loss.  
Hơn nữa, việc tích hợp này mang lại lợi ích về mặt tính toán,  
cho phép sử dụng các thủ thuật triển khai để thêm weight decay vào thuật toán  
mà không làm phát sinh thêm chi phí tính toán.  
Vì phần weight decay của phép cập nhật  
chỉ phụ thuộc vào giá trị hiện tại của từng parameter,  
optimizer trong mọi trường hợp cũng phải truy cập mỗi parameter một lần.

:begin_tab:`mxnet`
Dưới đây, chúng ta chỉ định hyperparameter weight decay trực tiếp  
thông qua `wd` khi khởi tạo `Trainer`.  
Theo mặc định, Gluon làm suy giảm đồng thời cả  
weights và biases.  
Lưu ý rằng hyperparameter `wd`  
sẽ được nhân với `wd_mult`  
khi cập nhật các parameter của model.  
Do đó, nếu đặt `wd_mult` bằng 0,  
parameter bias $b$ sẽ không bị suy giảm.
:end_tab:

:begin_tab:`pytorch`
Dưới đây, chúng ta chỉ định hyperparameter weight decay trực tiếp  
thông qua `weight_decay` khi khởi tạo optimizer.  
Theo mặc định, PyTorch làm suy giảm đồng thời cả  
weights và biases, nhưng  
chúng ta có thể cấu hình optimizer để xử lý các parameter khác nhau  
theo những chính sách khác nhau.  
Ở đây, chúng ta chỉ đặt `weight_decay` cho  
weights (các parameter `net.weight`), vì vậy  
bias (parameter `net.bias`) sẽ không bị suy giảm.
:end_tab:

:begin_tab:`tensorflow`
Dưới đây, chúng ta tạo một regularizer $\ell_2$ với  
hyperparameter weight decay `wd` và áp dụng nó cho weights của layer  
thông qua argument `kernel_regularizer`.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**Đồ thị trông tương tự như đồ thị khi  
chúng ta tự triển khai weight decay**].  
Tuy nhiên, phiên bản này chạy nhanh hơn  
và dễ triển khai hơn,  
những lợi ích này sẽ càng  
rõ rệt hơn khi bạn xử lý các bài toán lớn hơn  
và công việc này trở nên thường xuyên hơn.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

Cho đến nay, chúng ta đã đề cập đến một cách hiểu  
về việc một hàm tuyến tính đơn giản là gì.  
Tuy nhiên, ngay cả với các hàm phi tuyến đơn giản, tình huống cũng có thể phức tạp hơn nhiều. Để thấy điều này, khái niệm [không gian Hilbert kernel tái tạo (reproducing kernel Hilbert space, RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)  
cho phép áp dụng các công cụ được giới thiệu  
cho các hàm tuyến tính vào bối cảnh phi tuyến.  
Đáng tiếc là các thuật toán dựa trên RKHS  
thường khó mở rộng khi làm việc với dữ liệu lớn, nhiều chiều.  
Trong cuốn sách này, chúng ta thường áp dụng heuristic phổ biến  
rằng weight decay được áp dụng  
cho tất cả các layer của một deep network.

## Tóm tắt

Regularization là một phương pháp phổ biến để xử lý overfitting. Các kỹ thuật regularization cổ điển thêm một hạng tử penalty vào hàm loss (khi huấn luyện) để giảm độ phức tạp của model được học.
Một lựa chọn cụ thể để giữ cho model đơn giản là sử dụng penalty $\ell_2$. Điều này dẫn đến weight decay trong các bước cập nhật của thuật toán stochastic gradient descent theo minibatch.
Trong thực tế, chức năng weight decay được cung cấp trong các optimizer của framework deep learning.
Các tập parameter khác nhau có thể có hành vi cập nhật khác nhau trong cùng một vòng lặp huấn luyện.



## Bài tập

1. Thử nghiệm với giá trị của $\lambda$ trong bài toán ước lượng ở phần này. Vẽ accuracy huấn luyện và validation như một hàm của $\lambda$. Bạn quan sát thấy điều gì?
1. Sử dụng một tập validation để tìm giá trị tối ưu của $\lambda$. Đó có thực sự là giá trị tối ưu không? Điều này có quan trọng không?
1. Các phương trình cập nhật sẽ có dạng như thế nào nếu thay vì $\|\mathbf{w}\|^2$, chúng ta sử dụng $\sum_i |w_i|$ làm penalty được lựa chọn ($\ell_1$ regularization)?
1. Chúng ta biết rằng $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Bạn có thể tìm một phương trình tương tự cho các ma trận không (xem chuẩn Frobenius trong :numref:`subsec_lin-algebra-norms`)?
1. Ôn lại mối quan hệ giữa lỗi huấn luyện và lỗi tổng quát hóa. Ngoài weight decay, tăng cường huấn luyện và sử dụng một model có độ phức tạp phù hợp, những cách nào khác có thể giúp chúng ta xử lý overfitting?
1. Trong thống kê Bayes, chúng ta sử dụng tích của prior và likelihood để đi đến posterior thông qua $P(w \mid x) \propto P(x \mid w) P(w)$. Làm thế nào bạn có thể xác định $P(w)$ bằng regularization?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17979)
:end_tab: