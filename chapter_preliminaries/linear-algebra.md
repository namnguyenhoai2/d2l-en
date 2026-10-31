```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Đại số tuyến tính
:label:`sec_linear-algebra`

Đến đây, chúng ta đã có thể tải các bộ dữ liệu vào tensor
và thao tác với các tensor này
bằng những phép toán cơ bản.
Để bắt đầu xây dựng các mô hình phức tạp,
chúng ta cũng sẽ cần một vài công cụ từ đại số tuyến tính.
Phần này giới thiệu nhẹ nhàng
những khái niệm thiết yếu nhất,
bắt đầu từ số học vô hướng
và mở rộng dần đến phép nhân ma trận.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Vô hướng


Phần lớn toán học hằng ngày
bao gồm việc thao tác với
từng số một.
Về mặt hình thức, chúng ta gọi những giá trị này là *vô hướng*.
Ví dụ, nhiệt độ ở Palo Alto
là mức ấm áp $72$ độ Fahrenheit.
Nếu muốn chuyển đổi nhiệt độ sang độ Celsius,
bạn sẽ tính biểu thức
$c = \frac{5}{9}(f - 32)$, với $f$ được đặt bằng $72$.
Trong phương trình này, các giá trị
$5$, $9$, và $32$ là những vô hướng hằng.
Các biến $c$ và $f$
nói chung biểu diễn những vô hướng chưa biết.

Chúng ta ký hiệu các vô hướng
bằng những chữ cái thường thông thường
(ví dụ, $x$, $y$, và $z$)
và không gian của tất cả các vô hướng *có giá trị thực*
(liên tục) bằng $\mathbb{R}$.
Để thuận tiện, chúng ta sẽ bỏ qua
các định nghĩa chặt chẽ về *không gian*:
chỉ cần nhớ rằng biểu thức $x \in \mathbb{R}$
là cách nói hình thức rằng $x$ là một vô hướng có giá trị thực.
Ký hiệu $\in$ (đọc là "thuộc")
biểu thị sự thuộc về một tập hợp.
Ví dụ, $x, y \in \{0, 1\}$
cho biết rằng $x$ và $y$ là các biến
chỉ có thể nhận giá trị $0$ hoặc $1$.

(**Vô hướng được triển khai dưới dạng các tensor
chỉ chứa một phần tử.**)
Bên dưới, chúng ta gán hai vô hướng
và thực hiện các phép cộng, nhân,
chia, và lũy thừa quen thuộc.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## Vector

Trong phạm vi hiện tại, [**bạn có thể hiểu vector là một mảng vô hướng có độ dài cố định.**]
Giống như trong code,
chúng ta gọi các vô hướng này là *phần tử* của vector
(các từ đồng nghĩa bao gồm *mục nhập* và *thành phần*).
Khi vector biểu diễn các ví dụ từ những bộ dữ liệu trong thế giới thực,
các giá trị của chúng mang ý nghĩa thực tế.
Ví dụ, nếu đang huấn luyện một mô hình để dự đoán
rủi ro một khoản vay bị vỡ nợ,
chúng ta có thể gắn mỗi người nộp đơn với một vector
có các thành phần tương ứng với những đại lượng
như thu nhập, thời gian làm việc,
hoặc số lần vỡ nợ trước đây của họ.
Nếu đang nghiên cứu nguy cơ đau tim,
mỗi vector có thể biểu diễn một bệnh nhân
và các thành phần của vector có thể tương ứng với
các dấu hiệu sinh tồn gần đây nhất,
mức cholesterol,
số phút tập thể dục mỗi ngày, v.v.
Chúng ta ký hiệu các vector bằng các chữ cái thường in đậm,
(ví dụ, $\mathbf{x}$, $\mathbf{y}$, và $\mathbf{z}$).

Vector được triển khai dưới dạng các tensor có cấp $1^{\textrm{st}}$.
Nhìn chung, các tensor như vậy có thể có độ dài tùy ý,
phụ thuộc vào giới hạn bộ nhớ. Cẩn thận: trong Python, cũng như trong hầu hết các ngôn ngữ lập trình, chỉ số vector bắt đầu từ $0$, còn được gọi là *đánh chỉ số từ 0*, trong khi các chỉ số dưới trong đại số tuyến tính bắt đầu từ $1` (đánh chỉ số từ 1).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

Chúng ta có thể tham chiếu đến một phần tử của vector bằng cách sử dụng chỉ số dưới.
Ví dụ, $x_2$ biểu thị phần tử thứ hai của $\mathbf{x}$.
Vì $x_2$ là một vô hướng nên chúng ta không in đậm nó.
Theo mặc định, chúng ta trực quan hóa vector
bằng cách xếp các phần tử theo chiều dọc:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

Ở đây, $x_1, \ldots, x_n$ là các phần tử của vector.
Sau này, chúng ta sẽ phân biệt giữa các *vector cột*
như vậy và *vector hàng*, trong đó các phần tử được xếp theo chiều ngang.
Hãy nhớ rằng [**chúng ta truy cập các phần tử của tensor thông qua việc đánh chỉ số.**]

```{.python .input}
%%tab all
x[2]
```

Để biểu thị một vector chứa $n$ phần tử,
chúng ta viết $\mathbf{x} \in \mathbb{R}^n$.
Về mặt hình thức, chúng ta gọi $n$ là *số chiều* của vector.
[**Trong code, điều này tương ứng với độ dài của tensor**],
có thể truy cập thông qua hàm dựng sẵn `len` của Python.

```{.python .input}
%%tab all
len(x)
```

Chúng ta cũng có thể truy cập độ dài thông qua thuộc tính `shape`.
Shape là một tuple cho biết độ dài của tensor dọc theo mỗi trục.
(**Các tensor chỉ có một trục có shape chỉ gồm một phần tử.**)

```{.python .input}
%%tab all
x.shape
```

Thông thường, từ "chiều" bị dùng với nhiều nghĩa
để chỉ cả số lượng trục
và độ dài dọc theo một trục cụ thể.
Để tránh nhầm lẫn này,
chúng ta dùng *cấp* để chỉ số lượng trục
và chỉ dùng *số chiều* để chỉ số lượng thành phần.


## Ma trận

Cũng như vô hướng là các tensor cấp $0^{\textrm{th}}$
và vector là các tensor cấp $1^{\textrm{st}}$,
ma trận là các tensor cấp $2^{\textrm{nd}}$.
Chúng ta ký hiệu ma trận bằng các chữ cái viết hoa in đậm
(ví dụ, $\mathbf{X}$, $\mathbf{Y}$, và $\mathbf{Z}$),
và biểu diễn chúng trong code bằng các tensor có hai trục.
Biểu thức $\mathbf{A} \in \mathbb{R}^{m \times n}$
cho biết rằng ma trận $\mathbf{A}$
chứa $m \times n$ vô hướng có giá trị thực,
được sắp xếp thành $m$ hàng và $n$ cột.
Khi $m = n$, chúng ta nói ma trận là *ma trận vuông*.
Về mặt trực quan, chúng ta có thể minh họa mọi ma trận dưới dạng một bảng.
Để tham chiếu đến một phần tử riêng lẻ,
chúng ta đặt cả chỉ số hàng và cột ở dưới, ví dụ,
$a_{ij}$ là giá trị thuộc về hàng
$i^{\textrm{th}}$ và cột $j^{\textrm{th}}$ của $\mathbf{A}$:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


Trong code, chúng ta biểu diễn ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$
bằng một tensor cấp $2^{\textrm{nd}}$ có shape ($m$, $n$).
[**Chúng ta có thể chuyển đổi mọi tensor $m \times n$ có kích thước phù hợp
thành một ma trận $m \times n$**]
bằng cách truyền shape mong muốn vào `reshape`:

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

Đôi khi chúng ta muốn đảo các trục.
Khi hoán đổi các hàng và cột của một ma trận,
kết quả được gọi là *ma trận chuyển vị*.
Về mặt hình thức, chúng ta biểu thị ma trận chuyển vị của $\mathbf{A}$
bằng $\mathbf{A}^\top$ và nếu $\mathbf{B} = \mathbf{A}^\top$,
thì $b_{ij} = a_{ji}$ với mọi $i$ và $j$.
Do đó, chuyển vị của một ma trận $m \times n$
là một ma trận $n \times m$:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Trong code, chúng ta có thể truy cập (**ma trận chuyển vị**) như sau:

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**Ma trận đối xứng là tập con của các ma trận vuông
bằng chính ma trận chuyển vị của chúng:
$\mathbf{A} = \mathbf{A}^\top$.**]
Ma trận sau đây là ma trận đối xứng:

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

Ma trận rất hữu ích trong việc biểu diễn các bộ dữ liệu.
Thông thường, các hàng tương ứng với từng bản ghi
và các cột tương ứng với những thuộc tính riêng biệt.



## Tensor

Mặc dù bạn có thể tiến xa trong hành trình học máy
chỉ với vô hướng, vector và ma trận,
cuối cùng bạn có thể cần làm việc với
các [**tensor**] cấp cao hơn.
Tensor (**cung cấp cho chúng ta một cách chung để mô tả
các mảng cấp $n^{\textrm{th}}$ mở rộng.**)
Chúng ta gọi các đối tượng phần mềm thuộc *lớp tensor* là "tensor"
chính xác vì chúng cũng có thể có số lượng trục tùy ý.
Mặc dù việc dùng từ
*tensor* cho cả đối tượng toán học
và cách hiện thực hóa nó trong code có thể gây nhầm lẫn,
ý nghĩa của chúng ta thường sẽ rõ ràng dựa trên ngữ cảnh.
Chúng ta ký hiệu các tensor tổng quát bằng các chữ cái viết hoa
với kiểu phông chữ đặc biệt
(ví dụ, $\mathsf{X}$, $\mathsf{Y}$, và $\mathsf{Z}$)
và cơ chế đánh chỉ số của chúng
(ví dụ, $x_{ijk}$ và $[\mathsf{X}]_{1, 2i-1, 3}$)
tuân theo cách tự nhiên từ cơ chế của ma trận.

Tensor sẽ trở nên quan trọng hơn
khi chúng ta bắt đầu làm việc với hình ảnh.
Mỗi hình ảnh được biểu diễn dưới dạng tensor cấp $3^{\textrm{rd}}$
với các trục tương ứng với chiều cao, chiều rộng và *kênh*.
Tại mỗi vị trí không gian, cường độ
của mỗi màu (đỏ, xanh lá và xanh dương)
được xếp dọc theo kênh.
Ngoài ra, một tập hợp hình ảnh được biểu diễn
trong code bằng tensor cấp $4^{\textrm{th}}$,
trong đó các hình ảnh riêng biệt được đánh chỉ số
dọc theo trục đầu tiên.
Các tensor cấp cao hơn được xây dựng, tương tự như vector và ma trận,
bằng cách tăng số lượng thành phần shape.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## Các thuộc tính cơ bản của phép toán tensor

Vô hướng, vector, ma trận
và các tensor cấp cao hơn
đều có một số thuộc tính hữu ích.
Ví dụ, các phép toán theo phần tử
tạo ra những đầu ra có
shape giống với các toán hạng của chúng.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

Tích [**theo từng phần tử của hai ma trận
được gọi là *tích Hadamard* của chúng**] (ký hiệu là $\odot$).
Ta có thể viết rõ các phần tử
của tích Hadamard của hai ma trận
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**Cộng hoặc nhân một scalar với một tensor**] tạo ra một kết quả
có cùng shape với tensor ban đầu.
Ở đây, mỗi phần tử của tensor được cộng với (hoặc nhân với) scalar.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## Phép rút gọn
:label:`subsec_lin-alg-reduction`

Thông thường, chúng ta muốn tính [**tổng các phần tử của một tensor.**]
Để biểu diễn tổng các phần tử trong một vector $\mathbf{x}$ có độ dài $n$,
ta viết $\sum_{i=1}^n x_i$. Có một hàm đơn giản cho phép thực hiện việc này:

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

Để biểu diễn [**tổng các phần tử của tensor có shape tùy ý**],
ta chỉ cần tính tổng theo tất cả các axis của nó.
Ví dụ, tổng các phần tử
của ma trận $m \times n$ $\mathbf{A}$
có thể được viết là $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

Theo mặc định, việc gọi hàm tính tổng
*sẽ rút gọn* một tensor theo tất cả các axis của nó,
cuối cùng tạo ra một scalar.
Các thư viện của chúng ta cũng cho phép [**chỉ định các axis
mà tensor sẽ được rút gọn theo đó.**]
Để tính tổng tất cả các phần tử theo các hàng (axis 0),
ta chỉ định `axis=0` trong `sum`.
Vì ma trận đầu vào được rút gọn theo axis 0
để tạo ra vector đầu ra,
axis này không còn xuất hiện trong shape của đầu ra.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

Việc chỉ định `axis=1` sẽ rút gọn chiều cột (axis 1) bằng cách cộng các phần tử của tất cả các cột.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Rút gọn một ma trận theo cả hàng và cột bằng phép cộng
tương đương với việc cộng tất cả các phần tử của ma trận.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**Một đại lượng liên quan là *mean*, còn gọi là *average*.**]
Ta tính mean bằng cách chia tổng
cho tổng số phần tử.
Vì việc tính mean rất phổ biến,
nó có một hàm thư viện riêng
hoạt động tương tự như `sum`.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Tương tự, hàm tính mean
cũng có thể rút gọn một tensor theo các axis cụ thể.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Tổng không rút gọn
:label:`subsec_lin-alg-non-reduction`

Đôi khi, việc [**giữ nguyên số lượng axis**]
khi gọi hàm tính tổng hoặc mean có thể hữu ích.
Điều này quan trọng khi chúng ta muốn sử dụng cơ chế broadcasting.

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Ví dụ, vì `sum_A` giữ lại hai axis sau khi tính tổng từng hàng,
chúng ta có thể (**chia `A` cho `sum_A` bằng broadcasting**)
để tạo ra một ma trận trong đó tổng của mỗi hàng bằng $1$.

```{.python .input}
%%tab all
A / sum_A
```

Nếu muốn tính [**tổng tích lũy của các phần tử của `A` theo một axis nào đó**],
chẳng hạn `axis=0` (theo từng hàng), ta có thể gọi hàm `cumsum`.
Theo thiết kế, hàm này không rút gọn tensor đầu vào theo bất kỳ axis nào.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Tích vô hướng

Cho đến nay, chúng ta mới chỉ thực hiện các phép toán theo từng phần tử, phép tính tổng và mean.
Và nếu đó là tất cả những gì chúng ta có thể làm, đại số tuyến tính
sẽ không xứng đáng có một phần riêng.
May mắn thay, đây là lúc mọi thứ trở nên thú vị hơn.
Một trong những phép toán nền tảng nhất là tích vô hướng.
Cho hai vector $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
*tích vô hướng* của chúng $\mathbf{x}^\top \mathbf{y}$ (còn gọi là *inner product*, $\langle \mathbf{x}, \mathbf{y}  \rangle$)
là tổng của các tích giữa những phần tử ở cùng vị trí:
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~*Tích vô hướng* của hai vector là tổng các tích giữa những phần tử ở cùng vị trí~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

Tương đương, (**chúng ta có thể tính tích vô hướng của hai vector
bằng cách thực hiện phép nhân theo từng phần tử rồi tính tổng:**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

Tích vô hướng hữu ích trong nhiều ngữ cảnh.
Ví dụ, cho một tập hợp các giá trị,
được biểu diễn bằng vector $\mathbf{x}  \in \mathbb{R}^n$,
và một tập hợp các trọng số, được biểu diễn bằng $\mathbf{w} \in \mathbb{R}^n$,
tổng có trọng số của các giá trị trong $\mathbf{x}$
theo các trọng số $\mathbf{w}$
có thể được biểu diễn bằng tích vô hướng $\mathbf{x}^\top \mathbf{w}$.
Khi các trọng số không âm
và có tổng bằng $1$, tức là $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
tích vô hướng biểu diễn một *weighted average*.
Sau khi chuẩn hóa hai vector để có độ dài bằng đơn vị,
các tích vô hướng biểu diễn cosine của góc giữa chúng.
Ở phần sau của mục này, chúng ta sẽ giới thiệu chính thức khái niệm *độ dài* này.


## Tích ma trận--vector

Giờ đây, khi đã biết cách tính tích vô hướng,
chúng ta có thể bắt đầu tìm hiểu *tích*
giữa một ma trận $m \times n$ $\mathbf{A}$
và một vector $n$ chiều $\mathbf{x}$.
Trước hết, ta biểu diễn ma trận
theo các vector hàng của nó

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

trong đó mỗi $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
là một vector hàng biểu diễn hàng thứ $\mathbf{th}$
của ma trận $\mathbf{A}$.

[**Tích ma trận--vector $\mathbf{A}\mathbf{x}$
chỉ đơn giản là một vector cột có độ dài $m$,
trong đó phần tử thứ $\mathbf{th}$
là tích vô hướng
$\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Ta có thể xem phép nhân với một ma trận
$\mathbf{A}\in \mathbb{R}^{m \times n}$
là một phép biến đổi chiếu các vector
từ $\mathbb{R}^{n}$ sang $\mathbb{R}^{m}$.
Các phép biến đổi này cực kỳ hữu ích.
Ví dụ, chúng ta có thể biểu diễn phép quay
bằng phép nhân với các ma trận vuông nhất định.
Tích ma trận--vector cũng mô tả
phép tính then chốt liên quan đến việc tính
đầu ra của mỗi layer trong một neural network
dựa trên đầu ra từ layer trước đó.

:begin_tab:`mxnet`
Để biểu diễn tích ma trận--vector trong code,
ta sử dụng cùng hàm `dot`.
Phép toán được suy ra
dựa trên kiểu của các đối số.
Lưu ý rằng chiều cột của `A`
(độ dài của nó theo axis 1)
phải bằng chiều của `x` (độ dài của nó).
:end_tab:

:begin_tab:`pytorch`
Để biểu diễn tích ma trận--vector trong code,
ta sử dụng hàm `mv`.
Lưu ý rằng chiều cột của `A`
(độ dài của nó theo axis 1)
phải bằng chiều của `x` (độ dài của nó).
Python có một toán tử tiện lợi `@`
có thể thực hiện cả tích ma trận--vector
và tích ma trận--ma trận
(tùy thuộc vào các đối số).
Do đó, ta có thể viết `A@x`.
:end_tab:

:begin_tab:`tensorflow`
Để biểu diễn tích ma trận--vector trong code,
chúng ta sử dụng hàm `matvec`.
Lưu ý rằng số chiều cột của `A`
(độ dài của nó theo axis 1)
phải bằng số chiều của `x` (độ dài của nó).
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## Phép nhân ma trận--ma trận

Khi đã thành thạo tích vô hướng và tích ma trận--vector,
thì *phép nhân ma trận--ma trận* sẽ trở nên khá đơn giản.

Giả sử chúng ta có hai ma trận
$\mathbf{A} \in \mathbb{R}^{n \times k}$
và $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Gọi $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ là
vector hàng biểu diễn hàng thứ $i^\textrm{th}$
của ma trận $\mathbf{A}$
và gọi $\mathbf{b}_{j} \in \mathbb{R}^k$ là
vector cột từ cột thứ $j^\textrm{th}$
của ma trận $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


Để tạo tích ma trận $\mathbf{C} \in \mathbb{R}^{n \times m}$,
chúng ta chỉ cần tính từng phần tử $c_{ij}$
bằng tích vô hướng giữa
hàng thứ $i^{\textrm{th}}$ của $\mathbf{A}$
và cột thứ $j^{\textrm{th}}$ của $\mathbf{B}$,
tức là $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Chúng ta có thể xem phép nhân ma trận--ma trận $\mathbf{AB}$
là thực hiện $m$ tích ma trận--vector
hoặc $m \times n$ tích vô hướng
rồi ghép các kết quả lại
để tạo thành một ma trận $n \times m$.**]
Trong đoạn mã sau,
chúng ta thực hiện phép nhân ma trận trên `A` và `B`.
Ở đây, `A` là một ma trận có hai hàng và ba cột,
còn `B` là một ma trận có ba hàng và bốn cột.
Sau phép nhân, chúng ta thu được một ma trận có hai hàng và bốn cột.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

Thuật ngữ *phép nhân ma trận--ma trận* thường được rút gọn thành
*phép nhân ma trận*,
và không nên nhầm lẫn với tích Hadamard.


## Chuẩn
:label:`subsec_lin-algebra-norms`

Một số toán tử hữu ích nhất trong đại số tuyến tính là *chuẩn*.
Theo cách hiểu không chính thức, chuẩn của một vector cho biết nó *lớn* đến mức nào.
Chẳng hạn, chuẩn $\ell_2$ đo
độ dài (Euclid) của một vector.
Ở đây, chúng ta sử dụng khái niệm *độ lớn* liên quan đến độ lớn của các thành phần của vector
(chứ không phải số chiều của nó).

Chuẩn là một hàm $\| \cdot \|$ ánh xạ một vector
thành một scalar và thỏa mãn ba tính chất sau:

1. Với mọi vector $\mathbf{x}$, nếu chúng ta scale (tất cả phần tử của) vector
   bằng một scalar $\alpha \in \mathbb{R}$, thì chuẩn của nó cũng scale tương ứng:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Với mọi vector $\mathbf{x}$ và $\mathbf{y}$:
   các chuẩn thỏa mãn bất đẳng thức tam giác:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. Chuẩn của một vector không âm và chỉ bằng không khi vector đó là vector không:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Nhiều hàm là các chuẩn hợp lệ, và các chuẩn khác nhau
biểu đạt những khái niệm khác nhau về độ lớn.
Chuẩn Euclid mà chúng ta đã học trong hình học sơ cấp
khi tính cạnh huyền của một tam giác vuông
là căn bậc hai của tổng bình phương các phần tử của một vector.
Về hình thức, đây được gọi là [** *chuẩn* $\ell_2$ **] và được biểu diễn như sau:

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

Phương thức `norm` tính chuẩn $\ell_2$.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**Chuẩn $\ell_1$**] cũng rất phổ biến
và đại lượng tương ứng được gọi là khoảng cách Manhattan.
Theo định nghĩa, chuẩn $\ell_1$ tính tổng
các giá trị tuyệt đối của các phần tử của một vector:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

So với chuẩn $\ell_2$, nó ít nhạy cảm hơn với các giá trị ngoại lệ.
Để tính chuẩn $\ell_1$,
chúng ta kết hợp giá trị tuyệt đối
với phép toán tổng.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

Cả chuẩn $\ell_2$ và $\ell_1$ đều là các trường hợp đặc biệt
của *các chuẩn* $\ell_p$ tổng quát hơn:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Trong trường hợp ma trận, vấn đề phức tạp hơn.
Xét cho cùng, ma trận có thể được xem vừa là tập hợp các phần tử riêng lẻ
*vừa* là các đối tượng tác động lên vector và biến đổi chúng thành các vector khác.
Chẳng hạn, chúng ta có thể hỏi tích ma trận--vector $\mathbf{X} \mathbf{v}$
có thể dài hơn $\mathbf{v}$ bao nhiêu.
Cách suy nghĩ này dẫn đến khái niệm gọi là chuẩn *phổ*.
Trước mắt, chúng ta giới thiệu [** *chuẩn Frobenius*,
dễ tính hơn nhiều**] và được định nghĩa là
căn bậc hai của tổng bình phương
các phần tử của một ma trận:

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

Chuẩn Frobenius hoạt động như thể nó là
chuẩn $\ell_2$ của một vector có dạng ma trận.
Gọi hàm sau sẽ tính
chuẩn Frobenius của một ma trận.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

Mặc dù chưa muốn đi quá xa,
chúng ta đã có thể hình thành một phần trực giác về lý do các khái niệm này hữu ích.
Trong deep learning, chúng ta thường cố gắng giải các bài toán tối ưu:
*tối đa hóa* xác suất được gán cho dữ liệu quan sát;
*tối đa hóa* doanh thu gắn với một mô hình recommender;
*tối thiểu hóa* khoảng cách giữa các dự đoán
và các quan sát ground truth;
*tối thiểu hóa* khoảng cách giữa các biểu diễn
của ảnh chụp cùng một người
đồng thời *tối đa hóa* khoảng cách giữa các biểu diễn
của ảnh chụp những người khác nhau.
Các khoảng cách này, vốn tạo thành
các mục tiêu của những thuật toán deep learning,
thường được biểu diễn bằng các chuẩn.


## Thảo luận

Trong phần này, chúng ta đã ôn lại toàn bộ đại số tuyến tính
mà bạn cần để hiểu
một phần đáng kể của deep learning hiện đại.
Tuy nhiên, đại số tuyến tính còn nhiều nội dung khác,
và phần lớn trong số đó hữu ích cho machine learning.
Ví dụ, các ma trận có thể được phân rã thành các nhân tử,
và những phép phân rã này có thể làm sáng tỏ
cấu trúc có số chiều thấp trong các tập dữ liệu thực tế.
Có cả những phân ngành machine learning
tập trung vào việc sử dụng các phép phân rã ma trận
và các dạng tổng quát hóa của chúng cho tensor bậc cao
để khám phá cấu trúc trong các tập dữ liệu
và giải quyết các bài toán dự đoán.
Nhưng cuốn sách này tập trung vào deep learning.
Và chúng tôi tin rằng bạn sẽ có động lực hơn
để học thêm toán học
sau khi đã tự mình áp dụng machine learning
vào các tập dữ liệu thực tế.
Vì vậy, mặc dù vẫn có thể
giới thiệu thêm toán học ở những phần sau,
chúng ta kết thúc phần này tại đây.

Nếu muốn tìm hiểu thêm về đại số tuyến tính,
có rất nhiều sách và tài nguyên trực tuyến xuất sắc.
Để tham khảo một khóa học cấp tốc nâng cao hơn, hãy xem
:citet:`Strang.1993`, :citet:`Kolter.2008`, và :citet:`Petersen.Pedersen.ea.2008`.

Tóm tắt:

* Scalar, vector, ma trận và tensor là
  những đối tượng toán học cơ bản được sử dụng trong đại số tuyến tính
  và lần lượt có số trục là không, một, hai và tùy ý.
* Tensor có thể được cắt lát hoặc giảm theo các trục được chỉ định
  thông qua indexing, hoặc các phép toán như `sum` và `mean`, theo thứ tự tương ứng.
* Tích theo từng phần tử được gọi là tích Hadamard.
  Ngược lại, tích vô hướng, tích ma trận--vector và tích ma trận--ma trận
  không phải là các phép toán theo từng phần tử và nhìn chung trả về các đối tượng
  có shape khác với các toán hạng.
* So với tích Hadamard, tích ma trận--ma trận
  mất nhiều thời gian tính toán hơn đáng kể (thời gian bậc ba thay vì bậc hai).
* Norm thể hiện nhiều khái niệm khác nhau về độ lớn của một vector (hoặc ma trận),
  và thường được áp dụng cho hiệu của hai vector
  để đo khoảng cách giữa chúng.
* Các norm vector phổ biến gồm norm $\ell_1$ và $\ell_2$,
   còn các norm ma trận phổ biến gồm norm *spectral* và *Frobenius*.


## Bài tập

1. Chứng minh rằng phép chuyển vị của phép chuyển vị một ma trận chính là ma trận đó: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Cho hai ma trận $\mathbf{A}$ và $\mathbf{B}$, hãy chứng minh rằng phép cộng và phép chuyển vị giao hoán: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Với bất kỳ ma trận vuông nào $\mathbf{A}$, liệu $\mathbf{A} + \mathbf{A}^\top$ có luôn đối xứng không? Bạn có thể chứng minh kết quả chỉ bằng cách sử dụng các kết quả của hai bài tập trước không?
1. Trong phần này, chúng ta đã định nghĩa tensor `X` có shape (2, 3, 4). Kết quả của `len(X)` là gì? Hãy viết câu trả lời mà không triển khai code, sau đó kiểm tra câu trả lời bằng code.
1. Với một tensor `X` có shape tùy ý, liệu `len(X)` có luôn tương ứng với độ dài của một trục nhất định của `X` không? Đó là trục nào?
1. Chạy `A / A.sum(axis=1)` và xem điều gì xảy ra. Bạn có thể phân tích các kết quả không?
1. Khi di chuyển giữa hai điểm ở khu trung tâm Manhattan, khoảng cách cần đi là bao nhiêu nếu biểu diễn theo tọa độ, tức là theo đại lộ và đường phố? Bạn có thể đi theo đường chéo không?
1. Xét một tensor có shape (2, 3, 4). Shape của các kết quả tổng theo các trục 0, 1 và 2 là gì?
1. Truyền một tensor có từ ba trục trở lên vào hàm `linalg.norm` và quan sát kết quả. Hàm này tính toán điều gì đối với các tensor có shape tùy ý?
1. Xét ba ma trận lớn, chẳng hạn $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ và $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, được khởi tạo bằng các biến ngẫu nhiên Gaussian. Bạn muốn tính tích $\mathbf{A} \mathbf{B} \mathbf{C}$. Có sự khác biệt nào về mức sử dụng bộ nhớ và tốc độ tùy thuộc vào việc bạn tính $(\mathbf{A} \mathbf{B}) \mathbf{C}$ hay $\mathbf{A} (\mathbf{B} \mathbf{C})$ không? Tại sao?
1. Xét ba ma trận lớn, chẳng hạn $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ và $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Có sự khác biệt nào về tốc độ tùy thuộc vào việc bạn tính $\mathbf{A} \mathbf{B}$ hay $\mathbf{A} \mathbf{C}^\top$ không? Tại sao? Điều gì thay đổi nếu bạn khởi tạo $\mathbf{C} = \mathbf{B}^\top$ mà không sao chép bộ nhớ? Tại sao?
1. Xét ba ma trận, chẳng hạn $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Tạo một tensor có ba trục bằng cách xếp chồng $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. Tensor này có số chiều là bao nhiêu? Lấy ra tọa độ thứ hai của trục thứ ba để khôi phục $\mathbf{B}$. Hãy kiểm tra để đảm bảo câu trả lời của bạn là chính xác.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17968)
:end_tab: