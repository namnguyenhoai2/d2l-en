```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Triển khai Softmax Regression từ đầu
:label:`sec_softmax_scratch`

Vì softmax regression là nền tảng đến vậy,
chúng tôi cho rằng bạn nên biết
cách tự triển khai nó.
Ở đây, chúng ta chỉ giới hạn ở việc định nghĩa
các khía cạnh riêng của softmax trong mô hình
và tái sử dụng các thành phần khác
từ phần linear regression,
bao gồm cả vòng lặp huấn luyện.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## Softmax

Hãy bắt đầu với phần quan trọng nhất:
ánh xạ từ các giá trị vô hướng sang xác suất.
Để ôn lại, hãy nhớ thao tác của toán tử tổng
dọc theo các chiều cụ thể trong một tensor,
như đã thảo luận trong :numref:`subsec_lin-alg-reduction`
và :numref:`subsec_lin-alg-non-reduction`.
[**Với một ma trận `X`, chúng ta có thể tính tổng trên mọi phần tử (mặc định) hoặc chỉ
trên các phần tử cùng trục.**]
Biến `axis` cho phép chúng ta tính tổng theo hàng và cột:

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Việc tính softmax yêu cầu ba bước:
(i) lấy hàm mũ của từng hạng tử;
(ii) tính tổng trên mỗi hàng để tính hằng số chuẩn hóa cho mỗi mẫu;
(iii) chia mỗi hàng cho hằng số chuẩn hóa của nó,
đảm bảo rằng tổng kết quả bằng 1:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

Mẫu số (hoặc logarithm của nó)
được gọi là *hàm phân hoạch* (partition function).
Nó được giới thiệu trong [vật lý thống kê](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
để tổng hợp trên mọi trạng thái có thể có trong một tập hợp nhiệt động lực học.
Việc triển khai rất đơn giản:

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Với mọi đầu vào `X`, [**chúng ta biến từng phần tử
thành một số không âm.
Tổng mỗi hàng bằng 1,**]
đúng như yêu cầu đối với một xác suất. Lưu ý: đoạn mã trên *không* đủ vững chắc trước các đối số rất lớn hoặc rất nhỏ. Mặc dù nó đủ để minh họa điều đang diễn ra, bạn *không nên* dùng nguyên văn đoạn mã này cho bất kỳ mục đích nghiêm túc nào. Các framework deep learning đã tích hợp sẵn những cơ chế bảo vệ như vậy và từ đây chúng ta sẽ sử dụng softmax tích hợp sẵn.

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## Mô hình

Giờ đây chúng ta đã có mọi thứ cần thiết
để triển khai [**mô hình softmax regression.**]
Như trong ví dụ linear regression,
mỗi mẫu sẽ được biểu diễn
bằng một vector có độ dài cố định.
Vì dữ liệu thô ở đây bao gồm
các ảnh pixel $28 \times 28$,
[**chúng ta làm phẳng mỗi ảnh,
coi chúng là các vector có độ dài 784.**]
Trong các chương sau, chúng ta sẽ giới thiệu
convolutional neural networks,
vốn khai thác cấu trúc không gian
theo cách thỏa đáng hơn.


Trong softmax regression,
số lượng đầu ra của mạng
phải bằng số lượng lớp.
(**Vì dataset của chúng ta có 10 lớp,
mạng có chiều đầu ra là 10.**)
Do đó, trọng số của chúng ta tạo thành một ma trận $784 \times 10$
cộng với một vector hàng $1 \times 10$ cho các bias.
Tương tự linear regression,
chúng ta khởi tạo trọng số `W`
bằng nhiễu Gaussian.
Các bias được khởi tạo bằng không.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

Đoạn mã dưới đây định nghĩa cách mạng
ánh xạ mỗi đầu vào thành một đầu ra.
Lưu ý rằng chúng ta làm phẳng mỗi ảnh pixel $28 \times 28$ trong batch
thành một vector bằng `reshape`
trước khi truyền dữ liệu qua mô hình.

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## Hàm mất mát Cross-Entropy

Tiếp theo, chúng ta cần triển khai hàm mất mát cross-entropy
(được giới thiệu trong :numref:`subsec_softmax-regression-loss-func`).
Đây có thể là hàm mất mát phổ biến nhất
trong toàn bộ deep learning.
Hiện nay, các ứng dụng của deep learning
có thể dễ dàng được biểu diễn dưới dạng bài toán phân loại
nhiều hơn rất nhiều so với các ứng dụng phù hợp hơn
với bài toán hồi quy.

Hãy nhớ rằng cross-entropy lấy negative log-likelihood
của xác suất dự đoán được gán cho nhãn đúng.
Để đạt hiệu quả, chúng ta tránh dùng vòng lặp Python for và dùng indexing thay thế.
Cụ thể, mã hóa one-hot trong $\mathbf{y}$
cho phép chúng ta chọn các hạng tử tương ứng trong $\hat{\mathbf{y}}$.

Để thấy cách này hoạt động, chúng ta [**tạo dữ liệu mẫu `y_hat`
với 2 ví dụ về xác suất dự đoán trên 3 lớp và các nhãn tương ứng `y`.**]
Các nhãn đúng lần lượt là $0$ và $2$ (tức là lớp thứ nhất và thứ ba).
[**Dùng `y` làm chỉ số của các xác suất trong `y_hat`,**]
chúng ta có thể chọn các hạng tử một cách hiệu quả.

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
Bây giờ chúng ta có thể (**triển khai hàm mất mát cross-entropy**) bằng cách lấy trung bình logarithm của các xác suất đã chọn.
:end_tab:

:begin_tab:`jax`
Bây giờ chúng ta có thể (**triển khai hàm mất mát cross-entropy**) bằng cách lấy trung bình logarithm của các xác suất đã chọn.

Lưu ý rằng để sử dụng `jax.jit` nhằm tăng tốc các phần triển khai JAX, và
để đảm bảo `loss` là một hàm thuần, hàm `cross_entropy` được định nghĩa lại
bên trong `loss` để tránh sử dụng bất kỳ biến hoặc hàm toàn cục nào
có thể khiến hàm `loss` không còn thuần.
Chúng tôi giới thiệu tới độc giả quan tâm [tài liệu JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) về `jax.jit` và các hàm thuần.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return cross_entropy(y_hat, y), {}
```

## Huấn luyện

Chúng ta tái sử dụng phương thức `fit` được định nghĩa trong :numref:`sec_linear_scratch` để [**huấn luyện mô hình trong 10 epoch.**]
Lưu ý rằng số epoch (`max_epochs`),
kích thước minibatch (`batch_size`),
và learning rate (`lr`)
là các hyperparameter có thể điều chỉnh.
Điều đó có nghĩa là mặc dù các giá trị này không được
học trong vòng lặp huấn luyện chính,
chúng vẫn ảnh hưởng đến hiệu năng
của mô hình, cả về hiệu năng huấn luyện
lẫn khả năng tổng quát hóa.
Trong thực tế, bạn nên chọn các giá trị này
dựa trên phần dữ liệu *validation*
và sau đó, cuối cùng, đánh giá mô hình cuối cùng
trên phần dữ liệu *test*.
Như đã thảo luận trong :numref:`subsec_generalization-model-selection`,
chúng ta sẽ xem dữ liệu test của Fashion-MNIST
là tập validation, do đó
báo cáo validation loss và validation accuracy
trên phần dữ liệu này.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Dự đoán

Bây giờ quá trình huấn luyện đã hoàn tất,
mô hình của chúng ta sẵn sàng [**phân loại một số ảnh.**]

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

Chúng ta quan tâm nhiều hơn đến những ảnh được gán nhãn *không chính xác*. Chúng ta trực quan hóa chúng bằng cách
so sánh nhãn thực tế của chúng
(dòng văn bản đầu ra thứ nhất)
với dự đoán từ mô hình
(dòng văn bản đầu ra thứ hai).

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Tóm tắt

Đến đây, chúng ta đang bắt đầu có một số kinh nghiệm
trong việc giải quyết các bài toán linear regression
và phân loại.
Nhờ đó, chúng ta đã đạt đến thứ có thể xem là
trạng thái tiên tiến nhất của mô hình hóa thống kê trong thập niên 1960--1970.
Trong phần tiếp theo, chúng tôi sẽ chỉ cho bạn cách tận dụng
các framework deep learning để triển khai mô hình này
hiệu quả hơn nhiều.

## Bài tập

1. Trong phần này, chúng ta đã trực tiếp triển khai hàm softmax dựa trên định nghĩa toán học của phép toán softmax. Như đã thảo luận trong :numref:`sec_softmax`, điều này có thể gây ra bất ổn số học.
    1. Kiểm tra xem `softmax` có vẫn hoạt động chính xác nếu một đầu vào có giá trị $100$ hay không.
    1. Kiểm tra xem `softmax` có vẫn hoạt động chính xác nếu giá trị lớn nhất trong tất cả đầu vào nhỏ hơn $-100$ hay không.
    1. Triển khai một cách khắc phục bằng cách xem xét giá trị tương đối so với phần tử lớn nhất trong đối số.
1. Triển khai một hàm `cross_entropy` tuân theo định nghĩa của hàm mất mát cross-entropy $\sum_i y_i \log \hat{y}_i$.
    1. Thử dùng nó trong ví dụ mã của phần này.
    1. Theo bạn, vì sao nó chạy chậm hơn?
    1. Bạn có nên sử dụng nó không? Khi nào thì việc sử dụng nó hợp lý?
    1. Bạn cần cẩn thận điều gì? Gợi ý: hãy xét miền xác định của logarithm.
1. Có phải lúc nào trả về nhãn có xác suất cao nhất cũng là ý hay không? Ví dụ, bạn có làm điều này cho chẩn đoán y tế không? Bạn sẽ cố gắng giải quyết vấn đề này như thế nào?
1. Giả sử chúng ta muốn dùng softmax regression để dự đoán từ tiếp theo dựa trên một số đặc trưng. Những vấn đề nào có thể phát sinh từ một vocabulary lớn?
1. Thử nghiệm với các hyperparameter của mã trong phần này. Cụ thể:
    1. Vẽ biểu đồ thể hiện validation loss thay đổi thế nào khi bạn thay đổi learning rate.
    1. Validation loss và training loss có thay đổi khi bạn thay đổi kích thước minibatch không? Bạn cần tăng hoặc giảm đến mức nào trước khi thấy tác động?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17982)
:end_tab: