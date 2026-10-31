```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dropout
:label:`sec_dropout`


Hãy suy nghĩ ngắn gọn về những gì chúng ta
mong đợi từ một mô hình dự đoán tốt.
Chúng ta muốn nó hoạt động tốt trên dữ liệu chưa từng thấy.
Lý thuyết tổng quát hóa cổ điển
gợi ý rằng để thu hẹp khoảng cách giữa
hiệu năng train và test,
chúng ta nên hướng đến một mô hình đơn giản.
Tính đơn giản có thể thể hiện dưới dạng
một số lượng chiều nhỏ.
Chúng ta đã tìm hiểu điều này khi thảo luận về
các hàm cơ sở đơn thức của mô hình tuyến tính
trong :numref:`sec_generalization_basics`.
Ngoài ra, như đã thấy khi thảo luận về weight decay
(chính quy hóa $\ell_2$) trong :numref:`sec_weight_decay`,
chuẩn (nghịch đảo) của các tham số cũng
là một thước đo hữu ích về tính đơn giản.
Một khái niệm hữu ích khác về tính đơn giản là độ trơn,
nghĩa là hàm không nên nhạy cảm
với những thay đổi nhỏ ở đầu vào.
Ví dụ, khi phân loại ảnh,
chúng ta kỳ vọng rằng việc thêm một ít nhiễu ngẫu nhiên
vào các pixel hầu như sẽ không gây hại.

:citet:`Bishop.1995` đã chính thức hóa
ý tưởng này khi ông chứng minh rằng huấn luyện với nhiễu đầu vào
tương đương với chính quy hóa Tikhonov.
Công trình này thiết lập một mối liên hệ toán học rõ ràng
giữa yêu cầu một hàm phải trơn (và do đó đơn giản),
và yêu cầu hàm đó phải có khả năng chống chịu
trước các nhiễu động ở đầu vào.

Sau đó, :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`
đã phát triển một ý tưởng thông minh để áp dụng ý tưởng của Bishop
cho cả các tầng bên trong của mạng.
Ý tưởng của họ, được gọi là *dropout*, bao gồm
việc chèn nhiễu khi tính toán
từng tầng bên trong trong quá trình lan truyền xuôi,
và đã trở thành một kỹ thuật tiêu chuẩn
để huấn luyện mạng nơ-ron.
Phương pháp này được gọi là *dropout* vì theo đúng nghĩa đen
chúng ta *loại bỏ* một số nơ-ron trong quá trình huấn luyện.
Trong suốt quá trình huấn luyện, tại mỗi vòng lặp,
dropout tiêu chuẩn sẽ đưa về không
một tỷ lệ nào đó các nút trong mỗi tầng
trước khi tính tầng tiếp theo.

Cần làm rõ rằng chúng ta đang áp đặt
cách diễn giải riêng của mình về mối liên hệ với Bishop.
Bài báo gốc về dropout
đưa ra trực giác thông qua một phép so sánh đáng ngạc nhiên
với sinh sản hữu tính.
Các tác giả cho rằng overfitting của mạng nơ-ron
được đặc trưng bởi trạng thái trong đó
mỗi tầng phụ thuộc vào một
mẫu kích hoạt cụ thể ở tầng trước đó,
và gọi điều kiện này là *co-adaptation*.
Theo họ, dropout phá vỡ co-adaptation
tương tự như sinh sản hữu tính được cho là
phá vỡ các gen đồng thích nghi.
Mặc dù cách biện minh này cho lý thuyết chắc chắn vẫn còn gây tranh cãi,
bản thân kỹ thuật dropout đã chứng tỏ tính bền vững,
và nhiều dạng dropout khác nhau được triển khai
trong hầu hết các thư viện deep learning. 


Thách thức then chốt là cách chèn nhiễu này.
Một ý tưởng là chèn nhiễu theo cách *không chệch*
để giá trị kỳ vọng của mỗi tầng---khi cố định
các tầng khác---bằng giá trị mà nó sẽ nhận được nếu không có nhiễu.
Trong công trình của Bishop, ông thêm nhiễu Gaussian
vào đầu vào của một mô hình tuyến tính.
Tại mỗi vòng lặp huấn luyện, ông thêm nhiễu
được lấy mẫu từ một phân phối có trung bình bằng không
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ vào đầu vào $\mathbf{x}$,
tạo ra một điểm bị nhiễu $\mathbf{x}' = \mathbf{x} + \epsilon$.
Theo kỳ vọng, $E[\mathbf{x}'] = \mathbf{x}$.

Trong chính quy hóa dropout tiêu chuẩn,
chúng ta đưa về không một tỷ lệ nào đó các nút trong mỗi tầng
rồi *khử chệch* từng tầng bằng cách chuẩn hóa
theo tỷ lệ nút được giữ lại (không bị loại bỏ).
Nói cách khác,
với *xác suất dropout* $p$,
mỗi kích hoạt trung gian $h$ được thay thế bởi
một biến ngẫu nhiên $h'$ như sau:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ with probability } p \\
    \frac{h}{1-p} & \textrm{ otherwise}
\end{cases}
\end{aligned}
$$

Theo thiết kế, giá trị kỳ vọng không thay đổi, tức là $E[h'] = h$.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Dropout trong thực tế

Hãy nhớ lại MLP với một tầng ẩn và năm đơn vị ẩn
trong :numref:`fig_mlp`.
Khi áp dụng dropout cho một tầng ẩn,
đưa từng đơn vị ẩn về không với xác suất $p$,
kết quả có thể được xem là một mạng
chỉ chứa một tập con các nơ-ron ban đầu.
Trong :numref:`fig_dropout2`, $h_2$ và $h_5$ bị loại bỏ.
Do đó, việc tính toán các đầu ra
không còn phụ thuộc vào $h_2$ hay $h_5$
và gradient tương ứng của chúng cũng biến mất
khi thực hiện backpropagation.
Theo cách này, việc tính toán tầng đầu ra
không thể phụ thuộc quá mức vào bất kỳ
một phần tử nào trong $h_1, \ldots, h_5$.

![MLP trước và sau dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

Thông thường, chúng ta tắt dropout tại thời điểm test.
Với một mô hình đã huấn luyện và một ví dụ mới,
chúng ta không loại bỏ bất kỳ nút nào
và do đó không cần chuẩn hóa.
Tuy nhiên, có một số ngoại lệ:
một số nhà nghiên cứu sử dụng dropout tại thời điểm test như một heuristic
để ước lượng *độ không chắc chắn* của các dự đoán từ mạng nơ-ron:
nếu các dự đoán nhất quán trên nhiều đầu ra dropout khác nhau,
thì chúng ta có thể nói rằng mạng tự tin hơn.

## Triển khai từ đầu

Để triển khai hàm dropout cho một tầng đơn lẻ,
chúng ta phải lấy số mẫu
từ một biến ngẫu nhiên Bernoulli (nhị phân) bằng với số chiều của tầng,
trong đó biến ngẫu nhiên nhận giá trị $1$ (giữ lại)
với xác suất $1-p$ và $0$ (loại bỏ) với xác suất $p$.
Một cách dễ để triển khai là trước hết lấy mẫu
từ phân phối đều $U[0, 1]$.
Sau đó, chúng ta có thể giữ lại những nút mà
mẫu tương ứng lớn hơn $p$, và loại bỏ các nút còn lại.

Trong đoạn mã sau, chúng ta (**triển khai một hàm `dropout_layer`
loại bỏ các phần tử trong tensor đầu vào `X`
với xác suất `dropout`**),
đồng thời co giãn lại các phần tử còn lại như đã mô tả ở trên:
chia các phần tử được giữ lại cho `1.0-dropout`.

```{.python .input}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

Chúng ta có thể [**kiểm tra hàm `dropout_layer` trên một vài ví dụ**].
Trong các dòng mã sau,
chúng ta truyền đầu vào `X` qua phép toán dropout,
với các xác suất lần lượt là 0, 0.5 và 1.

```{.python .input}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### Định nghĩa mô hình

Mô hình dưới đây áp dụng dropout cho đầu ra
của từng tầng ẩn (sau hàm kích hoạt).
Chúng ta có thể đặt xác suất dropout riêng cho từng tầng.
Một lựa chọn phổ biến là đặt
xác suất dropout thấp hơn ở gần tầng đầu vào.
Chúng ta đảm bảo rằng dropout chỉ hoạt động trong quá trình huấn luyện.

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)

    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    def setup(self):
        self.lin1 = nn.Dense(self.num_hiddens_1)
        self.lin2 = nn.Dense(self.num_hiddens_2)
        self.lin3 = nn.Dense(self.num_outputs)
        self.relu = nn.relu

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**Huấn luyện**]

Phần sau tương tự như việc huấn luyện các MLP đã mô tả trước đó.

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**Triển khai ngắn gọn**]

Với các API cấp cao, tất cả những gì chúng ta cần làm là thêm một tầng `Dropout`
sau mỗi tầng fully connected,
truyền xác suất dropout
làm đối số duy nhất cho constructor của nó.
Trong quá trình huấn luyện, tầng `Dropout` sẽ ngẫu nhiên
loại bỏ các đầu ra của tầng trước đó
(hoặc tương đương, các đầu vào của tầng tiếp theo)
theo xác suất dropout đã chỉ định.
Khi không ở chế độ huấn luyện,
tầng `Dropout` chỉ đơn giản truyền dữ liệu đi trong quá trình kiểm thử.

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class DropoutMLP(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    @nn.compact
    def __call__(self, X):
        x = nn.relu(nn.Dense(self.num_hiddens_1)(X.reshape((X.shape[0], -1))))
        x = nn.Dropout(self.dropout_1, deterministic=not self.training)(x)
        x = nn.relu(nn.Dense(self.num_hiddens_2)(x))
        x = nn.Dropout(self.dropout_2, deterministic=not self.training)(x)
        return nn.Dense(self.num_outputs)(x)
```

:begin_tab:`jax`
Lưu ý rằng chúng ta cần định nghĩa lại hàm mất mát vì một mạng
có tầng dropout cần một PRNGKey khi sử dụng `Module.apply()`,
và RNG seed này cần được đặt tên rõ ràng là `dropout`. Key này
được tầng `dropout` trong Flax sử dụng để tạo mask dropout
ngẫu nhiên ở bên trong. Điều quan trọng là phải sử dụng một key `dropout_rng` duy nhất
với mỗi epoch trong vòng lặp huấn luyện, nếu không mask dropout được tạo ra
sẽ không mang tính ngẫu nhiên và không khác biệt giữa các lần chạy epoch.
`dropout_rng` này có thể được lưu trong
đối tượng `TrainState` (trong lớp `d2l.Trainer` được định nghĩa tại
:numref:`oo-design-training`) dưới dạng một thuộc tính, và ở mỗi epoch
nó được thay thế bằng một `dropout_rng` mới. Chúng ta đã xử lý việc này bằng
phương thức `fit_epoch` được định nghĩa trong :numref:`sec_linear_scratch`.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False,  # To be used later (e.g., batch norm)
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

Tiếp theo, chúng ta [**huấn luyện mô hình**].

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## Tóm tắt

Ngoài việc kiểm soát số lượng chiều và kích thước của vector trọng số, dropout là một công cụ khác để tránh overfitting. Các công cụ thường được sử dụng kết hợp.
Lưu ý rằng dropout
chỉ được sử dụng trong quá trình huấn luyện:
nó thay thế một kích hoạt $h$ bằng một biến ngẫu nhiên có giá trị kỳ vọng là $h$.


## Bài tập

1. Điều gì xảy ra nếu bạn thay đổi xác suất dropout cho tầng thứ nhất và tầng thứ hai? Đặc biệt, điều gì xảy ra nếu bạn hoán đổi xác suất của hai tầng? Hãy thiết kế một thí nghiệm để trả lời các câu hỏi này, mô tả kết quả của bạn một cách định lượng, và tóm tắt những kết luận định tính.
1. Tăng số epoch và so sánh kết quả thu được khi sử dụng dropout với khi không sử dụng nó.
1. Phương sai của các kích hoạt trong mỗi tầng ẩn là bao nhiêu khi có và không có dropout? Hãy vẽ biểu đồ thể hiện sự thay đổi của đại lượng này theo thời gian cho cả hai mô hình.
1. Vì sao dropout thường không được sử dụng tại thời điểm test?
1. Dùng mô hình trong phần này làm ví dụ, hãy so sánh tác động của việc sử dụng dropout và weight decay. Điều gì xảy ra khi dropout và weight decay được dùng cùng lúc? Kết quả có mang tính cộng dồn không? Có lợi ích giảm dần (hoặc tệ hơn) không? Chúng có triệt tiêu lẫn nhau không?
1. Điều gì xảy ra nếu chúng ta áp dụng dropout cho từng trọng số riêng lẻ của ma trận trọng số thay vì các kích hoạt?
1. Hãy phát minh một kỹ thuật khác để chèn nhiễu ngẫu nhiên tại mỗi tầng, khác với kỹ thuật dropout tiêu chuẩn. Bạn có thể phát triển một phương pháp vượt trội hơn dropout trên tập dữ liệu Fashion-MNIST (với một kiến trúc cố định) không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/261)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17987)
:end_tab: