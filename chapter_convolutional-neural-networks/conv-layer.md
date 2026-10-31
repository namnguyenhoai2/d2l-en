```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Phép tích chập cho ảnh
:label:`sec_conv_layer`

Giờ đây, khi đã hiểu cách các lớp tích chập hoạt động về mặt lý thuyết,
chúng ta sẵn sàng xem chúng hoạt động trong thực tế như thế nào.
Dựa trên động lực xây dựng mạng nơ-ron tích chập
như các kiến trúc hiệu quả để khai thác cấu trúc trong dữ liệu ảnh,
chúng ta sẽ tiếp tục dùng ảnh làm ví dụ xuyên suốt.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Phép toán tương quan chéo

Hãy nhớ rằng, nói một cách chặt chẽ thì các lớp tích chập
là một cách gọi không hoàn toàn chính xác, vì các phép toán mà chúng biểu diễn
được mô tả chính xác hơn là các phép tương quan chéo.
Dựa trên mô tả về các lớp tích chập trong :numref:`sec_why-conv`,
trong một lớp như vậy, một tensor đầu vào
và một tensor kernel được kết hợp
để tạo ra một tensor đầu ra thông qua một (**phép toán tương quan chéo.**)

Tạm thời hãy bỏ qua các kênh và xem cách điều này hoạt động
với dữ liệu hai chiều và các biểu diễn ẩn.
Trong :numref:`fig_correlation`,
đầu vào là một tensor hai chiều
có chiều cao 3 và chiều rộng 3.
Chúng ta ký hiệu hình dạng của tensor là $3 \times 3$ hoặc ($3$, $3$).
Chiều cao và chiều rộng của kernel đều là 2.
Hình dạng của *cửa sổ kernel* (hay *cửa sổ tích chập*)
được xác định bởi chiều cao và chiều rộng của kernel
(trong trường hợp này là $2 \times 2$).

![Phép toán tương quan chéo hai chiều. Các phần được tô bóng là phần tử đầu ra đầu tiên, cũng như các phần tử của tensor đầu vào và kernel được dùng để tính đầu ra: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

Trong phép toán tương quan chéo hai chiều,
chúng ta bắt đầu với cửa sổ tích chập được đặt
ở góc trên bên trái của tensor đầu vào
và trượt nó trên tensor đầu vào,
theo cả hướng từ trái sang phải lẫn từ trên xuống dưới.
Khi cửa sổ tích chập trượt đến một vị trí nhất định,
tensor con đầu vào nằm trong cửa sổ đó
và tensor kernel được nhân theo từng phần tử,
sau đó tensor kết quả được cộng lại
để tạo ra một giá trị vô hướng duy nhất.
Kết quả này cho giá trị của tensor đầu ra
tại vị trí tương ứng.
Ở đây, tensor đầu ra có chiều cao 2 và chiều rộng 2
và bốn phần tử được suy ra từ
phép toán tương quan chéo hai chiều:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Lưu ý rằng trên mỗi trục, kích thước đầu ra
nhỏ hơn một chút so với kích thước đầu vào.
Do kernel có chiều rộng và chiều cao lớn hơn $1$,
chúng ta chỉ có thể tính tương quan chéo đúng cách
tại các vị trí mà kernel nằm trọn trong ảnh,
kích thước đầu ra được xác định bằng kích thước đầu vào $n_\textrm{h} \times n_\textrm{w}$
trừ đi kích thước của kernel tích chập $k_\textrm{h} \times k_\textrm{w}$
theo công thức

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$

Điều này xảy ra vì chúng ta cần đủ không gian
để "trượt" kernel tích chập trên ảnh.
Sau này, chúng ta sẽ thấy cách giữ nguyên kích thước
bằng cách đệm ảnh bằng các số 0 quanh biên
để có đủ không gian trượt kernel.
Tiếp theo, chúng ta triển khai quy trình này trong hàm `corr2d`,
hàm này nhận một tensor đầu vào `X` và một tensor kernel `K`
và trả về một tensor đầu ra `Y`.

```{.python .input}
%%tab mxnet
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab jax
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y = Y.at[i, j].set((X[i:i + h, j:j + w] * K).sum())
    return Y
```

```{.python .input}
%%tab tensorflow
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

Chúng ta có thể tạo tensor đầu vào `X` và tensor kernel `K`
từ :numref:`fig_correlation`
để [**xác thực đầu ra của phần triển khai ở trên**]
cho phép toán tương quan chéo hai chiều.

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Các lớp tích chập

Một lớp tích chập thực hiện tương quan chéo giữa đầu vào và kernel
rồi thêm một bias vô hướng để tạo ra đầu ra.
Hai tham số của một lớp tích chập
là kernel và bias vô hướng.
Khi huấn luyện các mô hình dựa trên lớp tích chập,
chúng ta thường khởi tạo kernel ngẫu nhiên,
tương tự như đối với lớp kết nối đầy đủ.

Giờ đây, chúng ta sẵn sàng [**triển khai một lớp tích chập hai chiều**]
dựa trên hàm `corr2d` đã được định nghĩa ở trên.
Trong phương thức khởi tạo `__init__`,
chúng ta khai báo `weight` và `bias` là hai tham số mô hình.
Phương thức lan truyền thuận
gọi hàm `corr2d` và cộng thêm bias.

```{.python .input}
%%tab mxnet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
%%tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

```{.python .input}
%%tab jax
class Conv2D(nn.Module):
    kernel_size: int

    def setup(self):
        self.weight = nn.param('w', nn.initializers.uniform, self.kernel_size)
        self.bias = nn.param('b', nn.initializers.zeros, 1)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

Trong phép tích chập $h \times w$
hoặc một kernel tích chập $h \times w$,
chiều cao và chiều rộng của kernel tích chập lần lượt là $h$ và $w$.
Chúng ta cũng gọi
một lớp tích chập có kernel tích chập $h \times w$
đơn giản là một lớp tích chập $h \times w$.


## Phát hiện biên của đối tượng trong ảnh

Hãy dành chút thời gian để tìm hiểu [**một ứng dụng đơn giản của lớp tích chập:
phát hiện biên của một đối tượng trong ảnh**]
bằng cách tìm vị trí thay đổi pixel.
Trước tiên, chúng ta tạo một "ảnh" gồm $6\times 8$ pixel.
Bốn cột ở giữa có màu đen ($0$), còn các phần còn lại có màu trắng ($1$).

```{.python .input}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
%%tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

```{.python .input}
%%tab jax
X = jnp.ones((6, 8))
X = X.at[:, 2:6].set(0)
X
```

Tiếp theo, chúng ta tạo một kernel `K` có chiều cao 1 và chiều rộng 2.
Khi thực hiện phép tương quan chéo với đầu vào,
nếu các phần tử kề nhau theo chiều ngang giống nhau,
đầu ra là 0. Nếu không, đầu ra khác 0.
Lưu ý rằng kernel này là một trường hợp đặc biệt của toán tử sai phân hữu hạn. Tại vị trí $(i,j)$, nó tính $x_{i,j} - x_{(i+1),j}$, tức là tính chênh lệch giữa giá trị của các pixel kề nhau theo chiều ngang. Đây là một xấp xỉ rời rạc của đạo hàm bậc nhất theo hướng ngang. Xét cho cùng, với một hàm $f(i,j)$, đạo hàm của nó là $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$. Hãy xem điều này hoạt động trong thực tế như thế nào.

```{.python .input}
%%tab all
K = d2l.tensor([[1.0, -1.0]])
```

Chúng ta sẵn sàng thực hiện phép tương quan chéo
với các đối số `X` (đầu vào của chúng ta) và `K` (kernel của chúng ta).
Như bạn có thể thấy, [**chúng ta phát hiện giá trị $1$ cho biên từ trắng sang đen
và $-1$ cho biên từ đen sang trắng.**]
Mọi đầu ra khác đều có giá trị $0$.

```{.python .input}
%%tab all
Y = corr2d(X, K)
Y
```

Giờ đây, chúng ta có thể áp dụng kernel cho ảnh đã chuyển vị.
Đúng như dự kiến, nó triệt tiêu. [**Kernel `K` chỉ phát hiện các biên dọc.**]

```{.python .input}
%%tab all
corr2d(d2l.transpose(X), K)
```

## Học một kernel

Việc thiết kế bộ phát hiện biên bằng sai phân hữu hạn `[1, -1]` là rất tiện
nếu chúng ta biết chính xác đây là điều mình cần tìm.
Tuy nhiên, khi xét các kernel lớn hơn
và các lớp tích chập liên tiếp,
có thể không thể chỉ định chính xác
mỗi filter nên thực hiện điều gì theo cách thủ công.

Bây giờ, hãy xem liệu chúng ta có thể [**học kernel đã tạo ra `Y` từ `X`**]
chỉ bằng cách quan sát các cặp đầu vào--đầu ra hay không.
Trước tiên, chúng ta tạo một lớp tích chập
và khởi tạo kernel của nó bằng một tensor ngẫu nhiên.
Tiếp theo, trong mỗi vòng lặp, chúng ta sẽ dùng sai số bình phương
để so sánh `Y` với đầu ra của lớp tích chập.
Sau đó, chúng ta có thể tính gradient để cập nhật kernel.
Để đơn giản,
trong phần sau,
chúng ta sử dụng lớp dựng sẵn
cho các lớp tích chập hai chiều
và bỏ qua bias.

```{.python .input}
%%tab mxnet
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Learning rate

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
%%tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
%%tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Learning rate

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

```{.python .input}
%%tab jax
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv(1, kernel_size=(1, 2), use_bias=False, padding='VALID')

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 6, 8, 1))
Y = Y.reshape((1, 6, 7, 1))
lr = 3e-2  # Learning rate

params = conv2d.init(jax.random.PRNGKey(d2l.get_seed()), X)

def loss(params, X, Y):
    Y_hat = conv2d.apply(params, X)
    return ((Y_hat - Y) ** 2).sum()

for i in range(10):
    l, grads = jax.value_and_grad(loss)(params, X, Y)
    # Update the kernel
    params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l:.3f}')
```

Lưu ý rằng sai số đã giảm xuống một giá trị nhỏ sau 10 vòng lặp. Bây giờ chúng ta sẽ [**xem tensor kernel đã học được.**]

```{.python .input}
%%tab mxnet
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
%%tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

```{.python .input}
%%tab jax
params['params']['kernel'].reshape((1, 2))
```

Thật vậy, tensor kernel đã học được rất gần
với tensor kernel `K` mà chúng ta đã định nghĩa trước đó.

## Tương quan chéo và tích chập

Hãy nhớ lại nhận xét của chúng ta trong :numref:`sec_why-conv` về sự tương ứng
giữa phép tương quan chéo và phép tích chập.
Ở đây, chúng ta tiếp tục xét các lớp tích chập hai chiều.
Điều gì xảy ra nếu các lớp như vậy
thực hiện các phép tích chập chặt chẽ
như được định nghĩa trong :eqref:`eq_2d-conv-discrete`
thay vì tương quan chéo?
Để thu được đầu ra của phép *tích chập* chặt chẽ, chúng ta chỉ cần lật tensor kernel hai chiều theo cả chiều ngang lẫn chiều dọc, sau đó thực hiện phép *tương quan chéo* với tensor đầu vào.

Đáng chú ý là vì kernel được học từ dữ liệu trong deep learning,
đầu ra của các lớp tích chập không bị ảnh hưởng
dù các lớp đó
thực hiện
phép tích chập chặt chẽ
hay phép tương quan chéo.

Để minh họa điều này, giả sử một lớp tích chập thực hiện *tương quan chéo* và học kernel trong :numref:`fig_correlation`, ở đây được ký hiệu là ma trận $\mathbf{K}$.
Giả sử các điều kiện khác không đổi,
khi lớp này thay vào đó thực hiện *tích chập* chặt chẽ,
kernel đã học $\mathbf{K}'$ sẽ giống $\mathbf{K}$
sau khi $\mathbf{K}'$ được
lật theo cả chiều ngang lẫn chiều dọc.
Nói cách khác,
khi lớp tích chập
thực hiện *tích chập* chặt chẽ
với đầu vào trong :numref:`fig_correlation`
và $\mathbf{K}'$,
nó sẽ thu được cùng đầu ra trong :numref:`fig_correlation`
(tương quan chéo của đầu vào và $\mathbf{K}$).

Theo thuật ngữ chuẩn trong tài liệu deep learning,
chúng ta sẽ tiếp tục gọi phép toán tương quan chéo
là phép tích chập (convolution), dù xét một cách chặt chẽ thì chúng có đôi chút khác biệt.
Hơn nữa,
chúng ta dùng thuật ngữ *phần tử* để chỉ
một mục (hoặc thành phần) của bất kỳ tensor nào biểu diễn một biểu diễn tầng hoặc một kernel tích chập.


## Feature Map và Receptive Field

Như được mô tả trong :numref:`subsec_why-conv-channels`,
đầu ra của tầng tích chập trong
:numref:`fig_correlation`
đôi khi được gọi là *feature map*,
vì nó có thể được xem là
các biểu diễn (đặc trưng) đã học
trong các chiều không gian (ví dụ: chiều rộng và chiều cao)
cho tầng tiếp theo.
Trong CNN,
với bất kỳ phần tử $x$ nào của một tầng,
*receptive field* của nó đề cập đến
tất cả các phần tử (từ mọi tầng trước đó)
có thể ảnh hưởng đến việc tính toán $x$
trong quá trình lan truyền xuôi (forward propagation).
Lưu ý rằng receptive field
có thể lớn hơn kích thước thực tế của đầu vào.

Hãy tiếp tục sử dụng :numref:`fig_correlation` để giải thích receptive field.
Với kernel tích chập $2 \times 2$,
receptive field của phần tử đầu ra được tô bóng (có giá trị $19$)
là
bốn phần tử trong vùng được tô bóng của đầu vào.
Bây giờ, hãy ký hiệu đầu ra $2 \times 2$
là $\mathbf{Y}$
và xét một CNN sâu hơn
với một tầng tích chập $2 \times 2$ bổ sung nhận $\mathbf{Y}$
làm đầu vào và cho ra
một phần tử duy nhất $z$.
Trong trường hợp này,
receptive field của $z$
trên $\mathbf{Y}$ bao gồm cả bốn phần tử của $\mathbf{Y}$,
trong khi
receptive field
trên đầu vào bao gồm toàn bộ chín phần tử đầu vào.
Do đó,
khi bất kỳ phần tử nào trong một feature map
cần receptive field lớn hơn
để phát hiện các đặc trưng đầu vào trên một vùng rộng hơn,
chúng ta có thể xây dựng một mạng sâu hơn.


Receptive field có tên gọi bắt nguồn từ sinh lý thần kinh.
Một loạt thí nghiệm trên nhiều loài động vật với các kích thích khác nhau
:cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968` đã khảo sát phản ứng của cái gọi là vỏ não
thị giác đối với các kích thích nói trên. Nhìn chung, họ nhận thấy rằng các tầng thấp hơn phản ứng với các cạnh và
các hình dạng liên quan. Sau đó, :citet:`Field.1987` đã minh họa hiệu ứng này trên các ảnh tự nhiên
bằng những gì chỉ có thể gọi là các kernel tích chập.
Chúng tôi in lại một hình quan trọng trong :numref:`field_visual` để minh họa những điểm tương đồng nổi bật.

![Hình và chú thích lấy từ :citet:`Field.1987`: Một ví dụ về mã hóa với sáu kênh khác nhau. (Trái) Ví dụ về sáu loại cảm biến gắn với mỗi kênh. (Phải) Tích chập của ảnh ở (Giữa) với sáu cảm biến được hiển thị ở (Trái). Phản hồi của từng cảm biến được xác định bằng cách lấy mẫu các ảnh đã lọc này tại khoảng cách tỷ lệ với kích thước của cảm biến (được biểu thị bằng các chấm). Sơ đồ này chỉ hiển thị phản hồi của các cảm biến đối xứng chẵn.](../img/field-visual.png)
:label:`field_visual`

Hóa ra, mối quan hệ này thậm chí còn đúng với các đặc trưng được tính bởi các tầng sâu hơn của những mạng được huấn luyện cho các tác vụ phân loại ảnh, như được minh họa, chẳng hạn, trong :citet:`Kuzovkin.Vicente.Petton.ea.2018`. Chỉ cần nói rằng, phép tích chập đã chứng tỏ là một công cụ cực kỳ mạnh mẽ cho thị giác máy tính, cả trong sinh học lẫn trong mã nguồn. Vì vậy, không có gì ngạc nhiên (khi nhìn lại) rằng chúng đã mở đường cho thành công gần đây của deep learning.

## Tóm tắt

Phép tính cốt lõi cần thiết cho một tầng tích chập là phép toán tương quan chéo. Chúng ta đã thấy rằng chỉ cần một vòng lặp for lồng nhau đơn giản để tính giá trị của nó. Nếu có nhiều kênh đầu vào và nhiều kênh đầu ra, chúng ta đang thực hiện một phép toán ma trận--ma trận giữa các kênh. Như có thể thấy, phép tính này đơn giản và, quan trọng nhất, có tính *cục bộ* cao. Điều này cho phép tối ưu hóa phần cứng đáng kể, và nhiều kết quả gần đây trong thị giác máy tính chỉ khả thi nhờ điều đó. Xét cho cùng, điều này có nghĩa là các nhà thiết kế chip có thể đầu tư vào tính toán nhanh thay vì bộ nhớ khi tối ưu hóa cho phép tích chập. Mặc dù điều này có thể không dẫn đến các thiết kế tối ưu cho những ứng dụng khác, nó mở ra cánh cửa cho thị giác máy tính phổ biến và có chi phí phải chăng.

Về bản thân phép tích chập, chúng có thể được sử dụng cho nhiều mục đích, chẳng hạn như phát hiện cạnh và đường thẳng, làm mờ ảnh hoặc làm sắc nét ảnh. Quan trọng nhất, không cần nhà thống kê (hoặc kỹ sư) phải phát minh các bộ lọc phù hợp. Thay vào đó, chúng ta có thể đơn giản *học* chúng từ dữ liệu. Điều này thay thế các heuristic feature engineering bằng thống kê dựa trên bằng chứng. Cuối cùng, và khá thú vị, các bộ lọc này không chỉ có lợi cho việc xây dựng mạng sâu mà còn tương ứng với receptive field và feature map trong não bộ. Điều này cho chúng ta niềm tin rằng mình đang đi đúng hướng.

## Bài tập

1. Xây dựng một ảnh `X` có các cạnh chéo.
    1. Điều gì xảy ra nếu bạn áp dụng kernel `K` trong phần này lên ảnh đó?
    1. Điều gì xảy ra nếu bạn chuyển vị `X`?
    1. Điều gì xảy ra nếu bạn chuyển vị `K`?
1. Tự thiết kế một số kernel.
    1. Với vector có hướng $\mathbf{v} = (v_1, v_2)$, hãy suy ra một kernel phát hiện cạnh có thể phát hiện
       các cạnh trực giao với $\mathbf{v}$, tức là các cạnh theo hướng $(v_2, -v_1)$.
    1. Suy ra một toán tử sai phân hữu hạn cho đạo hàm bậc hai. Kích thước tối thiểu
       của kernel tích chập liên kết với nó là bao nhiêu? Những cấu trúc nào trong ảnh phản hồi mạnh nhất với nó?
    1. Bạn sẽ thiết kế kernel làm mờ như thế nào? Vì sao bạn có thể muốn sử dụng kernel như vậy?
    1. Kích thước tối thiểu của một kernel để thu được đạo hàm bậc $d$ là bao nhiêu?
1. Khi bạn cố gắng tự động tìm gradient cho lớp `Conv2D` mà chúng ta đã tạo, bạn thấy loại thông báo lỗi nào?
1. Bạn biểu diễn phép toán tương quan chéo dưới dạng phép nhân ma trận bằng cách thay đổi các tensor đầu vào và kernel như thế nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/271)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17996)
:end_tab: