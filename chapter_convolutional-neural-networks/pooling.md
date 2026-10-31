```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Pooling
:label:`sec_pooling`

Trong nhiều trường hợp, tác vụ cuối cùng của chúng ta đặt ra một câu hỏi toàn cục về ảnh,
ví dụ: *ảnh có chứa một con mèo không?* Do đó, các đơn vị của tầng cuối cùng
cần nhạy cảm với toàn bộ đầu vào.
Bằng cách dần dần tổng hợp thông tin, tạo ra các ánh xạ ngày càng thô hơn,
chúng ta đạt được mục tiêu cuối cùng là học một biểu diễn toàn cục,
đồng thời vẫn giữ được mọi ưu điểm của các tầng convolution ở các tầng xử lý trung gian.
Càng đi sâu vào mạng,
trường tiếp nhận (receptive field, so với đầu vào)
mà mỗi nút ẩn nhạy cảm càng lớn. Việc giảm độ phân giải không gian
đẩy nhanh quá trình này,
vì các kernel convolution bao phủ một vùng hiệu dụng lớn hơn.

Hơn nữa, khi phát hiện các đặc trưng cấp thấp, chẳng hạn như các cạnh
(như đã thảo luận trong :numref:`sec_conv_layer`),
chúng ta thường muốn các biểu diễn của mình có tính bất biến nhất định đối với phép tịnh tiến.
Ví dụ, nếu lấy ảnh `X`
có ranh giới rõ nét giữa màu đen và trắng
và dịch toàn bộ ảnh sang phải một pixel,
tức là `Z[i, j] = X[i, j + 1]`,
thì đầu ra cho ảnh mới `Z` có thể khác biệt rất nhiều.
Cạnh sẽ bị dịch đi một pixel.
Trên thực tế, các đối tượng hầu như không bao giờ xuất hiện chính xác ở cùng một vị trí.
Thực tế, ngay cả khi dùng chân máy và đối tượng đứng yên,
rung động của máy ảnh do chuyển động của màn trập
cũng có thể làm mọi thứ dịch chuyển khoảng một pixel
(các máy ảnh cao cấp được trang bị nhiều tính năng đặc biệt để xử lý vấn đề này).

Phần này giới thiệu *các tầng pooling*,
có hai mục đích:
giảm thiểu độ nhạy của các tầng convolution đối với vị trí
và giảm lấy mẫu không gian của các biểu diễn.

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
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Maximum Pooling và Average Pooling

Giống như các tầng convolution, các toán tử *pooling*
bao gồm một cửa sổ có hình dạng cố định được trượt qua
mọi vùng trong đầu vào theo stride của nó,
tính một đầu ra duy nhất cho mỗi vị trí mà
cửa sổ hình dạng cố định đi qua (đôi khi được gọi là *pooling window*).
Tuy nhiên, không giống phép tính cross-correlation
giữa đầu vào và kernel trong tầng convolution,
tầng pooling không chứa tham số nào (không có *kernel*).
Thay vào đó, các toán tử pooling mang tính xác định,
thường tính giá trị lớn nhất hoặc giá trị trung bình
của các phần tử trong pooling window.
Các phép toán này lần lượt được gọi là *maximum pooling* (*max-pooling* viết tắt)
và *average pooling*.

*Average pooling* gần như xuất hiện từ khi có CNN. Ý tưởng này tương tự như
giảm lấy mẫu một ảnh. Thay vì chỉ lấy giá trị của mỗi pixel thứ hai (hoặc thứ ba)
cho ảnh có độ phân giải thấp hơn, ta có thể lấy trung bình các pixel lân cận để thu được
một ảnh có tỷ lệ tín hiệu trên nhiễu tốt hơn, vì đang kết hợp thông tin
từ nhiều pixel liền kề. *Max-pooling* được giới thiệu trong
:citet:`Riesenhuber.Poggio.1999` trong ngữ cảnh khoa học thần kinh nhận thức để mô tả
cách việc tổng hợp thông tin có thể được thực hiện theo cấp bậc nhằm
nhận dạng đối tượng; một phiên bản trước đó đã tồn tại trong nhận dạng giọng nói :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. Trong hầu hết mọi trường hợp, max-pooling, như cách gọi khác của nó,
được ưu tiên hơn average pooling.

Trong cả hai trường hợp, cũng như với toán tử cross-correlation,
ta có thể xem pooling window
bắt đầu từ góc trên bên trái của tensor đầu vào
và trượt qua nó từ trái sang phải và từ trên xuống dưới.
Tại mỗi vị trí mà pooling window đi qua,
nó tính giá trị lớn nhất hoặc trung bình
của subtensor đầu vào trong cửa sổ,
tùy thuộc vào việc sử dụng max pooling hay average pooling.


![Max-pooling với pooling window có hình dạng $2\times 2$. Các phần được tô bóng là phần tử đầu ra đầu tiên cũng như các phần tử tensor đầu vào được dùng để tính đầu ra: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

Tensor đầu ra trong :numref:`fig_pooling` có chiều cao là 2 và chiều rộng là 2.
Bốn phần tử được suy ra từ giá trị lớn nhất trong từng pooling window:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Tổng quát hơn, ta có thể định nghĩa một tầng pooling $p \times q$ bằng cách tổng hợp trên
một vùng có kích thước như vậy. Quay lại bài toán phát hiện cạnh,
ta sử dụng đầu ra của tầng convolution
làm đầu vào cho max-pooling $2\times 2$.
Gọi `X` là đầu vào của tầng convolution và `Y` là đầu ra của tầng pooling.
Bất kể các giá trị của `X[i, j]`, `X[i, j + 1]`,
`X[i+1, j]` và `X[i+1, j + 1]` có khác nhau hay không,
tầng pooling luôn xuất ra `Y[i, j] = 1`.
Nói cách khác, khi dùng tầng max-pooling $2\times 2$,
ta vẫn có thể phát hiện liệu mẫu được tầng convolution nhận diện
có di chuyển không quá một phần tử theo chiều cao hoặc chiều rộng hay không.

Trong đoạn mã dưới đây, chúng ta (**triển khai lan truyền xuôi
của tầng pooling**) trong hàm `pool2d`.
Hàm này tương tự với hàm `corr2d`
trong :numref:`sec_conv_layer`.
Tuy nhiên, không cần kernel, đầu ra được tính
là giá trị lớn nhất hoặc trung bình của từng vùng trong đầu vào.

```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab jax
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = jnp.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].max())
            elif mode == 'avg':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].mean())
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

Ta có thể xây dựng tensor đầu vào `X` trong :numref:`fig_pooling` để [**xác thực đầu ra của tầng max-pooling hai chiều**].

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Ta cũng có thể thử nghiệm với (**tầng average pooling**).

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**Padding và Stride**]

Cũng như các tầng convolution, các tầng pooling
làm thay đổi hình dạng đầu ra.
Và như trước, ta có thể điều chỉnh phép toán để đạt được hình dạng đầu ra mong muốn
bằng cách padding đầu vào và điều chỉnh stride.
Ta có thể minh họa việc sử dụng padding và stride
trong các tầng pooling thông qua tầng max-pooling hai chiều có sẵn của framework deep learning.
Trước tiên, ta xây dựng tensor đầu vào `X` có hình dạng gồm bốn chiều,
trong đó số lượng ví dụ (batch size) và số lượng kênh đều bằng 1.

:begin_tab:`tensorflow`
Lưu ý rằng không giống các framework khác, TensorFlow
ưu tiên và được tối ưu hóa cho đầu vào *channels-last*.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow, jax
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Vì pooling tổng hợp thông tin từ một vùng, (**các framework deep learning mặc định đặt kích thước pooling window và stride giống nhau.**) Ví dụ, nếu sử dụng pooling window có hình dạng `(3, 3)`
thì mặc định ta nhận được hình dạng stride là `(3, 3)`.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab jax
# Pooling has no model parameters, hence it needs no initialization
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

Không cần phải nói, [**stride và padding có thể được chỉ định thủ công**] để ghi đè các giá trị mặc định của framework nếu cần.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

Dĩ nhiên, ta có thể chỉ định một pooling window hình chữ nhật bất kỳ với chiều cao và chiều rộng tùy ý, như ví dụ dưới đây cho thấy.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
%%tab jax

X_padded = jnp.pad(X, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(2, 3), strides=(2, 3), padding='VALID')
```

## Nhiều kênh

Khi xử lý dữ liệu đầu vào nhiều kênh,
[**tầng pooling xử lý pooling riêng cho từng kênh đầu vào**],
thay vì cộng các đầu vào qua các kênh
như trong tầng convolution.
Điều này có nghĩa là số kênh đầu ra của tầng pooling
bằng số kênh đầu vào.
Dưới đây, chúng ta sẽ nối các tensor `X` và `X + 1`
theo chiều kênh để xây dựng đầu vào có hai kênh.

:begin_tab:`tensorflow`
Lưu ý rằng điều này sẽ yêu cầu
nối theo chiều cuối cùng trong TensorFlow do cú pháp channels-last.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# Concatenate along `dim=3` due to channels-last syntax
X = d2l.concat([X, X + 1], 3)
X
```

Như có thể thấy, số kênh đầu ra vẫn là hai sau khi pooling.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

:begin_tab:`tensorflow`
Lưu ý rằng thoạt nhìn, đầu ra của TensorFlow pooling có vẻ khác biệt, tuy nhiên
về mặt số học, kết quả được trình bày giống như MXNet và PyTorch.
Khác biệt nằm ở số chiều, và khi đọc
đầu ra theo chiều dọc, ta sẽ có đầu ra giống như các cách triển khai khác.
:end_tab:

## Tóm tắt

Pooling là một phép toán cực kỳ đơn giản. Nó thực hiện đúng như tên gọi: tổng hợp kết quả trên một cửa sổ các giá trị. Mọi ngữ nghĩa của convolution, chẳng hạn như stride và padding, đều được áp dụng giống như trước. Lưu ý rằng pooling không phụ thuộc vào kênh, tức là nó giữ nguyên số lượng kênh và áp dụng riêng cho từng kênh. Cuối cùng, trong hai lựa chọn pooling phổ biến, max-pooling được ưu tiên hơn average pooling, vì nó mang lại một mức độ bất biến nhất định cho đầu ra. Một lựa chọn phổ biến là chọn kích thước pooling window $2 \times 2$ để giảm độ phân giải không gian của đầu ra xuống còn một phần tư.

Lưu ý rằng có nhiều cách khác để giảm độ phân giải ngoài pooling. Ví dụ, trong stochastic pooling :cite:`Zeiler.Fergus.2013` và fractional max-pooling :cite:`Graham.2014`, việc tổng hợp được kết hợp với ngẫu nhiên hóa. Điều này có thể cải thiện đôi chút độ chính xác trong một số trường hợp. Cuối cùng, như chúng ta sẽ thấy sau này với cơ chế attention, có các cách tinh vi hơn để tổng hợp đầu ra, ví dụ bằng cách sử dụng sự liên kết giữa một query và các vector biểu diễn.


## Bài tập

1. Triển khai average pooling thông qua một phép convolution. 
1. Chứng minh rằng không thể triển khai max-pooling chỉ bằng một phép convolution. 
1. Max-pooling có thể được thực hiện bằng các phép toán ReLU, tức là $\textrm{ReLU}(x) = \max(0, x)$.
    1. Biểu diễn $\max (a, b)$ chỉ bằng các phép toán ReLU.
    1. Dùng cách này để triển khai max-pooling bằng các phép convolution và các lớp ReLU. 
    1. Bạn cần bao nhiêu channel và layer cho một phép convolution $2 \times 2$? Bao nhiêu cho một phép convolution $3 \times 3$?
1. Chi phí tính toán của lớp pooling là bao nhiêu? Giả sử đầu vào của lớp pooling có kích thước $c\times h\times w$, cửa sổ pooling có dạng $p_\textrm{h}\times p_\textrm{w}$ với padding $(p_\textrm{h}, p_\textrm{w})$ và stride $(s_\textrm{h}, s_\textrm{w})$.
1. Vì sao bạn kỳ vọng max-pooling và average pooling hoạt động khác nhau?
1. Chúng ta có cần một lớp minimum pooling riêng biệt không? Bạn có thể thay thế nó bằng một phép toán khác không?
1. Chúng ta có thể dùng phép toán softmax để pooling. Vì sao nó có thể không phổ biến?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/274)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17999)
:end_tab: