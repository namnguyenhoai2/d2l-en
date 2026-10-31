```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Triển khai Perceptron Đa tầng
:label:`sec_mlp-implementation`

Perceptron đa tầng (MLP) không phức tạp hơn nhiều để triển khai so với các mô hình tuyến tính đơn giản. Điểm khác biệt chính về mặt khái niệm là giờ đây chúng ta nối nhiều tầng lại với nhau.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
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
```

## Triển khai từ đầu

Hãy bắt đầu lại bằng cách tự triển khai một mạng như vậy từ đầu.

### Khởi tạo Tham số Mô hình

Hãy nhớ rằng Fashion-MNIST chứa 10 lớp,
và mỗi hình ảnh bao gồm một lưới $28 \times 28 = 784$
giá trị pixel thang độ xám.
Như trước, hiện tại chúng ta sẽ bỏ qua cấu trúc không gian
giữa các pixel,
vì vậy có thể xem đây là một tập dữ liệu phân loại
với 784 đặc trưng đầu vào và 10 lớp.
Để bắt đầu, chúng ta sẽ [**triển khai một MLP
với một tầng ẩn và 256 đơn vị ẩn.**]
Cả số lượng tầng lẫn độ rộng của chúng đều có thể điều chỉnh
(chúng được xem là các siêu tham số).
Thông thường, chúng ta chọn độ rộng tầng chia hết cho các lũy thừa lớn hơn của 2.
Điều này hiệu quả về mặt tính toán do cách
bộ nhớ được cấp phát và đánh địa chỉ trong phần cứng.

Một lần nữa, chúng ta sẽ biểu diễn các tham số bằng nhiều tensor.
Lưu ý rằng *với mỗi tầng*, chúng ta phải theo dõi
một ma trận trọng số và một vector bias.
Như mọi khi, chúng ta cấp phát bộ nhớ
cho gradient của loss đối với các tham số này.

:begin_tab:`mxnet`
Trong đoạn mã dưới đây, trước hết chúng ta xác định và khởi tạo các tham số,
sau đó bật tính năng theo dõi gradient.
:end_tab:

:begin_tab:`pytorch`
Trong đoạn mã dưới đây, chúng ta sử dụng `nn.Parameter`
để tự động đăng ký
một thuộc tính lớp làm tham số được `autograd` theo dõi (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
Trong đoạn mã dưới đây, chúng ta sử dụng `tf.Variable`
để xác định tham số mô hình.
:end_tab:

:begin_tab:`jax`
Trong đoạn mã dưới đây, chúng ta sử dụng `flax.linen.Module.param`
để xác định tham số mô hình.
:end_tab:

```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### Mô hình

Để bảo đảm chúng ta hiểu cách mọi thứ hoạt động,
chúng ta sẽ [**tự triển khai hàm kích hoạt ReLU**]
thay vì gọi trực tiếp hàm `relu` tích hợp sẵn.

```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

Vì chúng ta bỏ qua cấu trúc không gian,
chúng ta `reshape` mỗi hình ảnh hai chiều thành
một vector phẳng có độ dài `num_inputs`.
Cuối cùng, chúng ta (**triển khai mô hình của mình**)
chỉ với vài dòng mã. Vì sử dụng autograd tích hợp sẵn của framework, đó là tất cả những gì cần thiết.

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Huấn luyện

May mắn thay, [**vòng lặp huấn luyện cho MLP
hoàn toàn giống với vòng lặp cho hồi quy softmax.**] Chúng ta xác định mô hình, dữ liệu và trainer, rồi cuối cùng gọi phương thức `fit` trên mô hình và dữ liệu.

```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Triển khai Ngắn gọn

Như bạn có thể mong đợi, bằng cách dựa vào các API cấp cao, chúng ta có thể triển khai MLP ngắn gọn hơn nữa.

### Mô hình

So với cách triển khai ngắn gọn
của hồi quy softmax
(:numref:`sec_softmax_concise`),
khác biệt duy nhất là chúng ta thêm
*hai* tầng kết nối đầy đủ, trong khi trước đây chỉ thêm *một* tầng.
Tầng đầu tiên là [**tầng ẩn**],
tầng thứ hai là tầng đầu ra.

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

Trước đây, chúng ta đã xác định các phương thức `forward` cho mô hình để biến đổi đầu vào bằng các tham số mô hình.
Các thao tác này về cơ bản là một pipeline:
bạn lấy một đầu vào và
áp dụng một phép biến đổi (ví dụ,
nhân ma trận với trọng số rồi cộng bias),
sau đó lặp lại việc dùng đầu ra của phép biến đổi hiện tại làm
đầu vào cho phép biến đổi tiếp theo.
Tuy nhiên, bạn có thể đã nhận thấy rằng
không có phương thức `forward` nào được xác định ở đây.
Thực tế, `MLP` kế thừa phương thức `forward` từ lớp `Module` (:numref:`subsec_oo-design-models`) để
chỉ đơn giản gọi `self.net(X)` (`X` là đầu vào),
vốn hiện được xác định như một chuỗi các phép biến đổi
thông qua lớp `Sequential`.
Lớp `Sequential` trừu tượng hóa quá trình forward,
cho phép chúng ta tập trung vào các phép biến đổi.
Chúng ta sẽ thảo luận thêm về cách lớp `Sequential` hoạt động trong :numref:`subsec_model-construction-sequential`.


### Huấn luyện

[**Vòng lặp huấn luyện**] hoàn toàn giống
như khi chúng ta triển khai hồi quy softmax.
Tính mô-đun này cho phép chúng ta tách biệt
các vấn đề liên quan đến kiến trúc mô hình
khỏi các yếu tố trực giao khác.

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Tóm tắt

Giờ đây, khi đã thực hành nhiều hơn về thiết kế mạng sâu, bước chuyển từ mạng sâu một tầng sang nhiều tầng không còn là một thách thức đáng kể nữa. Đặc biệt, chúng ta có thể tái sử dụng thuật toán huấn luyện và data loader. Tuy nhiên, lưu ý rằng việc triển khai MLP từ đầu vẫn khá rườm rà: việc đặt tên và theo dõi các tham số mô hình khiến việc mở rộng mô hình trở nên khó khăn. Chẳng hạn, hãy tưởng tượng bạn muốn chèn thêm một tầng giữa tầng 42 và 43. Tầng này giờ đây có thể là tầng 42b, trừ khi chúng ta sẵn sàng thực hiện việc đổi tên tuần tự. Hơn nữa, nếu tự triển khai mạng từ đầu, framework sẽ khó thực hiện các tối ưu hóa hiệu năng có ý nghĩa hơn nhiều.

Dẫu vậy, giờ đây bạn đã đạt đến trình độ tiên tiến nhất của cuối thập niên 1980, khi các mạng sâu kết nối đầy đủ là phương pháp được ưu tiên cho việc mô hình hóa mạng nơ-ron. Bước tiếp theo về mặt khái niệm của chúng ta sẽ là xem xét hình ảnh. Trước khi làm điều đó, chúng ta cần ôn lại một số kiến thức cơ bản về thống kê và các chi tiết về cách tính toán mô hình hiệu quả.

## Bài tập

1. Thay đổi số lượng đơn vị ẩn `num_hiddens` và vẽ biểu đồ thể hiện số lượng này ảnh hưởng đến độ chính xác của mô hình như thế nào. Giá trị tốt nhất của siêu tham số này là gì?
1. Hãy thử thêm một tầng ẩn để xem nó ảnh hưởng đến kết quả như thế nào.
1. Vì sao việc chèn một tầng ẩn chỉ có một nơ-ron là một ý tưởng không tốt? Điều gì có thể xảy ra?
1. Việc thay đổi learning rate làm thay đổi kết quả của bạn như thế nào? Khi cố định mọi tham số khác, learning rate nào cho kết quả tốt nhất? Điều này liên quan thế nào đến số epoch?
1. Hãy tối ưu hóa đồng thời tất cả siêu tham số, tức là learning rate, số epoch, số tầng ẩn và số đơn vị ẩn trên mỗi tầng.
    1. Kết quả tốt nhất bạn có thể đạt được khi tối ưu hóa tất cả chúng là gì?
    1. Vì sao việc xử lý nhiều siêu tham số lại khó hơn nhiều?
    1. Hãy mô tả một chiến lược hiệu quả để tối ưu hóa đồng thời nhiều tham số.
1. So sánh tốc độ của framework và cách triển khai từ đầu đối với một bài toán khó. Nó thay đổi như thế nào theo độ phức tạp của mạng?
1. Đo tốc độ nhân tensor--ma trận đối với các ma trận được căn chỉnh tốt và không được căn chỉnh. Ví dụ, hãy kiểm tra các ma trận có kích thước 1024, 1025, 1026, 1028 và 1032.
    1. Điều này thay đổi như thế nào giữa GPU và CPU?
    1. Xác định độ rộng bus bộ nhớ của CPU và GPU của bạn.
1. Hãy thử các hàm kích hoạt khác nhau. Hàm nào hoạt động tốt nhất?
1. Có khác biệt nào giữa các cách khởi tạo trọng số của mạng không? Điều đó có quan trọng không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17985)
:end_tab: