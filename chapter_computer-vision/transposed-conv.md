# Phép tích chập chuyển vị
:label:`sec_transposed_conv`

Các lớp CNN mà chúng ta đã tìm hiểu cho đến nay,
chẳng hạn như các lớp convolution (:numref:`sec_conv_layer`) và các lớp pooling (:numref:`sec_pooling`),
thường làm giảm (downsample) các chiều không gian (chiều cao và chiều rộng) của đầu vào,
hoặc giữ nguyên chúng.
Trong semantic segmentation
phân loại ở cấp độ pixel,
sẽ rất thuận tiện nếu
các chiều không gian của
đầu vào và đầu ra giống nhau.
Ví dụ,
chiều kênh tại một pixel đầu ra
có thể chứa các kết quả phân loại
cho pixel đầu vào tại cùng vị trí không gian.


Để đạt được điều này, đặc biệt là sau khi
các chiều không gian bị giảm bởi các lớp CNN,
chúng ta có thể sử dụng một loại lớp CNN khác
có khả năng tăng (upsample) các chiều không gian
của các feature map trung gian.
Trong phần này,
chúng ta sẽ giới thiệu
*phép tích chập chuyển vị* (transposed convolution), còn được gọi là *phép tích chập có stride phân số* (fractionally-strided convolution) :cite:`Dumoulin.Visin.2016`,
để đảo ngược các phép downsampling
do phép convolution thực hiện.

```{.python .input}
#@tab mxnet
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## Phép toán cơ bản

Tạm thời bỏ qua các kênh,
hãy bắt đầu với
phép tích chập chuyển vị cơ bản
có stride bằng 1 và không padding.
Giả sử rằng
chúng ta có một tensor đầu vào
$n_h \times n_w$
và một kernel $k_h \times k_w$.
Trượt cửa sổ kernel với stride bằng 1
$n_w$ lần trên mỗi hàng
và $n_h$ lần trên mỗi cột
sẽ tạo ra
tổng cộng $n_h n_w$ kết quả trung gian.
Mỗi kết quả trung gian là một tensor
$(n_h + k_h - 1) \times (n_w + k_w - 1)$
được khởi tạo bằng các số 0.
Để tính từng tensor trung gian,
mỗi phần tử trong tensor đầu vào
được nhân với kernel
sao cho tensor $k_h \times k_w$ thu được
thay thế một phần
trong từng tensor trung gian.
Lưu ý rằng
vị trí của phần được thay thế trong mỗi
tensor trung gian tương ứng với vị trí của phần tử
trong tensor đầu vào được sử dụng để tính toán.
Cuối cùng, tất cả các kết quả trung gian
được cộng lại để tạo ra đầu ra.

Ví dụ,
:numref:`fig_trans_conv` minh họa
cách tính phép tích chập chuyển vị với kernel $2\times 2$ cho một tensor đầu vào $2\times 2$.


![Phép tích chập chuyển vị với kernel $2\times 2$. Các phần được tô bóng là một phần của tensor trung gian, đồng thời là các phần tử của tensor đầu vào và kernel được sử dụng để tính toán.](../img/trans_conv.svg)
:label:`fig_trans_conv`


Chúng ta có thể (**triển khai phép tích chập chuyển vị cơ bản này**) `trans_conv` cho một ma trận đầu vào `X` và một ma trận kernel `K`.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

Khác với phép convolution thông thường (trong :numref:`sec_conv_layer`), vốn *làm giảm* các phần tử đầu vào
thông qua kernel,
phép tích chập chuyển vị
*broadcast* các phần tử đầu vào
thông qua kernel, từ đó
tạo ra một đầu ra
lớn hơn đầu vào.
Chúng ta có thể tạo tensor đầu vào `X` và tensor kernel `K` từ :numref:`fig_trans_conv` để [**xác thực đầu ra của phần triển khai ở trên**] của phép tích chập chuyển vị hai chiều cơ bản.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

Ngoài ra,
khi đầu vào `X` và kernel `K` đều là
các tensor bốn chiều,
chúng ta có thể [**sử dụng các API cấp cao để thu được cùng kết quả**].

```{.python .input}
#@tab mxnet
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**Padding, Stride và nhiều kênh**]

Khác với phép convolution thông thường,
trong đó padding được áp dụng cho đầu vào,
padding được áp dụng cho đầu ra
trong phép tích chập chuyển vị.
Ví dụ,
khi chỉ định số padding
ở mỗi phía của chiều cao và chiều rộng
là 1,
hàng và cột đầu tiên cũng như cuối cùng
sẽ bị loại bỏ khỏi đầu ra của phép tích chập chuyển vị.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

Trong phép tích chập chuyển vị,
stride được chỉ định cho các kết quả trung gian (do đó là đầu ra), không phải cho đầu vào.
Sử dụng cùng các tensor đầu vào và kernel
từ :numref:`fig_trans_conv`,
việc thay đổi stride từ 1 thành 2
làm tăng cả chiều cao và chiều rộng
của các tensor trung gian, do đó làm tăng tensor đầu ra
trong :numref:`fig_trans_conv_stride2`.


![Phép tích chập chuyển vị với kernel $2\times 2$ có stride bằng 2. Các phần được tô bóng là một phần của tensor trung gian, đồng thời là các phần tử của tensor đầu vào và kernel được sử dụng để tính toán.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`



Đoạn mã sau đây có thể dùng để xác thực đầu ra của phép tích chập chuyển vị với stride bằng 2 trong :numref:`fig_trans_conv_stride2`.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

Đối với nhiều kênh đầu vào và đầu ra,
phép tích chập chuyển vị
hoạt động giống như phép convolution thông thường.
Giả sử rằng
đầu vào có $c_i$ kênh,
và phép tích chập chuyển vị
gán một tensor kernel $k_h\times k_w$
cho mỗi kênh đầu vào.
Khi chỉ định nhiều kênh đầu ra,
chúng ta sẽ có một kernel $c_i\times k_h\times k_w$ cho mỗi kênh đầu ra.


Nói chung, nếu chúng ta đưa $\mathsf{X}$ vào một lớp convolution $f$ để tạo đầu ra $\mathsf{Y}=f(\mathsf{X})$ và tạo một lớp convolution chuyển vị $g$ với cùng các hyperparameter như $f$, ngoại trừ
số kênh đầu ra
bằng số kênh trong $\mathsf{X}$,
thì $g(Y)$ sẽ có cùng shape với $\mathsf{X}$.
Điều này có thể được minh họa trong ví dụ sau.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**Mối liên hệ với phép chuyển vị ma trận**]
:label:`subsec-connection-to-mat-transposition`

Phép tích chập chuyển vị được đặt tên theo
phép chuyển vị ma trận.
Để giải thích,
trước tiên hãy
xem cách triển khai các phép convolution
bằng phép nhân ma trận.
Trong ví dụ dưới đây, chúng ta định nghĩa một đầu vào `X` $3\times 3$ và một kernel convolution `K` $2\times 2$, sau đó sử dụng hàm `corr2d` để tính đầu ra convolution `Y`.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

Tiếp theo, chúng ta viết lại kernel convolution `K` dưới dạng
một ma trận trọng số thưa `W`
chứa rất nhiều số 0.
Shape của ma trận trọng số là ($4$, $9$),
trong đó các phần tử khác 0 đến từ
kernel convolution `K`.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

Nối đầu vào `X` theo từng hàng để thu được một vector có độ dài 9. Sau đó, phép nhân ma trận giữa `W` và `X` đã vector hóa tạo ra một vector có độ dài 4.
Sau khi reshape, chúng ta có thể thu được cùng kết quả `Y`
từ phép convolution ban đầu ở trên:
chúng ta vừa triển khai phép convolution bằng phép nhân ma trận.

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

Tương tự, chúng ta có thể triển khai phép tích chập chuyển vị bằng
phép nhân ma trận.
Trong ví dụ sau,
chúng ta lấy đầu ra `Y` $2 \times 2$ từ
phép convolution thông thường ở trên
làm đầu vào cho phép tích chập chuyển vị.
Để triển khai phép toán này bằng cách nhân ma trận,
chúng ta chỉ cần chuyển vị ma trận trọng số `W`
thành shape mới $(9, 4)$.

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

Xét việc triển khai phép convolution
bằng phép nhân ma trận.
Với một vector đầu vào $\mathbf{x}$
và một ma trận trọng số $\mathbf{W}$,
hàm forward propagation của phép convolution
có thể được triển khai
bằng cách nhân đầu vào với ma trận trọng số
và xuất ra một vector
$\mathbf{y}=\mathbf{W}\mathbf{x}$.
Vì backpropagation
tuân theo chain rule
và $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$,
hàm backpropagation của phép convolution
có thể được triển khai
bằng cách nhân đầu vào của nó với
ma trận trọng số chuyển vị $\mathbf{W}^\top$.
Do đó,
lớp convolution chuyển vị
chỉ cần hoán đổi hàm forward propagation
và hàm backpropagation của lớp convolution:
các hàm forward propagation
và backpropagation của nó
lần lượt nhân vector đầu vào của chúng với
$\mathbf{W}^\top$ và $\mathbf{W}$.


## Tóm tắt

* Khác với phép convolution thông thường, vốn làm giảm các phần tử đầu vào thông qua kernel, phép tích chập chuyển vị broadcast các phần tử đầu vào thông qua kernel, từ đó tạo ra một đầu ra lớn hơn đầu vào.
* Nếu chúng ta đưa $\mathsf{X}$ vào một lớp convolution $f$ để tạo đầu ra $\mathsf{Y}=f(\mathsf{X})$ và tạo một lớp convolution chuyển vị $g$ với cùng các hyperparameter như $f$, ngoại trừ số kênh đầu ra bằng số kênh trong $\mathsf{X}$, thì $g(Y)$ sẽ có cùng shape với $\mathsf{X}$.
* Chúng ta có thể triển khai phép convolution bằng phép nhân ma trận. Lớp convolution chuyển vị chỉ cần hoán đổi hàm forward propagation và hàm backpropagation của lớp convolution.


## Bài tập

1. Trong :numref:`subsec-connection-to-mat-transposition`, đầu vào của phép convolution `X` và đầu ra của phép tích chập chuyển vị `Z` có cùng shape. Chúng có cùng giá trị không? Tại sao?
1. Có hiệu quả không nếu sử dụng phép nhân ma trận để triển khai phép convolution? Tại sao?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1450)
:end_tab: