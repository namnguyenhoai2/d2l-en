```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Thao tác dữ liệu
:label:`sec_ndarray`

Để thực hiện được bất kỳ công việc nào, 
chúng ta cần một cách để lưu trữ và thao tác với dữ liệu.
Nói chung, có hai việc quan trọng 
chúng ta cần làm với dữ liệu: 
(i) thu thập dữ liệu; 
và (ii) xử lý dữ liệu sau khi chúng đã được đưa vào máy tính. 
Việc thu thập dữ liệu sẽ không có ý nghĩa 
nếu không có cách lưu trữ chúng, 
vì vậy, trước tiên, hãy cùng bắt tay vào làm quen
với các mảng $n$ chiều, 
mà chúng ta cũng gọi là *tensor*.
Nếu bạn đã quen thuộc với gói điện toán khoa học NumPy, 
việc này sẽ rất dễ dàng.
Trong tất cả các framework deep learning hiện đại,
*tensor class* (`ndarray` trong MXNet, 
`Tensor` trong PyTorch và TensorFlow) 
có nhiều điểm tương đồng với `ndarray` của NumPy,
nhưng được bổ sung một số tính năng nổi bật.
Thứ nhất, tensor class
hỗ trợ automatic differentiation.
Thứ hai, nó tận dụng GPU
để tăng tốc tính toán số,
trong khi NumPy chỉ chạy trên CPU.
Những thuộc tính này giúp việc lập trình mạng neural
vừa dễ dàng vừa có tốc độ thực thi nhanh.



## Bắt đầu

:begin_tab:`mxnet`
Để bắt đầu, chúng ta import các module `np` (`numpy`) và
`npx` (`numpy_extension`) từ MXNet.
Ở đây, module `np` bao gồm 
các hàm được NumPy hỗ trợ,
trong khi module `npx` chứa một tập hợp các phần mở rộng
được phát triển để hỗ trợ deep learning 
trong một môi trường tương tự NumPy.
Khi sử dụng tensor, chúng ta hầu như luôn 
gọi hàm `set_np`:
điều này nhằm bảo đảm khả năng xử lý tensor 
tương thích với các thành phần khác của MXNet.
:end_tab:

:begin_tab:`pytorch`
(**Để bắt đầu, chúng ta import thư viện PyTorch.
Lưu ý rằng tên package là `torch`.**)
:end_tab:

:begin_tab:`tensorflow`
Để bắt đầu, chúng ta import `tensorflow`. 
Để viết ngắn gọn, các lập trình viên 
thường đặt alias là `tf`.
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**Một tensor biểu diễn một mảng (có thể là đa chiều) gồm các giá trị số.**]
Trong trường hợp một chiều, tức là khi dữ liệu chỉ cần một trục,
tensor được gọi là một *vector*.
Với hai trục, tensor được gọi là một *matrix*.
Với $k > 2$ trục, chúng ta không dùng các tên chuyên biệt
mà chỉ gọi đối tượng đó là một $k^\textrm{th}$-*order tensor*.

:begin_tab:`mxnet`
MXNet cung cấp nhiều hàm 
để tạo các tensor mới 
được điền sẵn các giá trị. 
Ví dụ, bằng cách gọi `arange(n)`,
chúng ta có thể tạo một vector gồm các giá trị cách đều nhau,
bắt đầu từ 0 (được lấy) 
và kết thúc tại `n` (không được lấy).
Theo mặc định, kích thước khoảng cách là $1$.
Trừ khi được chỉ định khác đi, 
các tensor mới được lưu trong bộ nhớ chính 
và được dùng cho việc tính toán trên CPU.
:end_tab:

:begin_tab:`pytorch`
PyTorch cung cấp nhiều hàm 
để tạo các tensor mới 
được điền sẵn các giá trị. 
Ví dụ, bằng cách gọi `arange(n)`,
chúng ta có thể tạo một vector gồm các giá trị cách đều nhau,
bắt đầu từ 0 (được lấy) 
và kết thúc tại `n` (không được lấy).
Theo mặc định, kích thước khoảng cách là $1$.
Trừ khi được chỉ định khác đi, 
các tensor mới được lưu trong bộ nhớ chính 
và được dùng cho việc tính toán trên CPU.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow cung cấp nhiều hàm 
để tạo các tensor mới 
được điền sẵn các giá trị. 
Ví dụ, bằng cách gọi `range(n)`,
chúng ta có thể tạo một vector gồm các giá trị cách đều nhau,
bắt đầu từ 0 (được lấy) 
và kết thúc tại `n` (không được lấy).
Theo mặc định, kích thước khoảng cách là $1$.
Trừ khi được chỉ định khác đi, 
các tensor mới được lưu trong bộ nhớ chính 
và được dùng cho việc tính toán trên CPU.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
Mỗi giá trị này được gọi là
một *phần tử* của tensor.
Tensor `x` chứa 12 phần tử.
Chúng ta có thể kiểm tra tổng số phần tử 
trong một tensor thông qua thuộc tính `size` của nó.
:end_tab:

:begin_tab:`pytorch`
Mỗi giá trị này được gọi là
một *phần tử* của tensor.
Tensor `x` chứa 12 phần tử.
Chúng ta có thể kiểm tra tổng số phần tử 
trong một tensor thông qua phương thức `numel` của nó.
:end_tab:

:begin_tab:`tensorflow`
Mỗi giá trị này được gọi là
một *phần tử* của tensor.
Tensor `x` chứa 12 phần tử.
Chúng ta có thể kiểm tra tổng số phần tử 
trong một tensor thông qua hàm `size`.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**Chúng ta có thể truy cập *shape***) 
(độ dài dọc theo mỗi trục)
bằng cách kiểm tra thuộc tính `shape` của nó.
Vì ở đây chúng ta đang làm việc với một vector,
`shape` chỉ chứa một phần tử
và giống với size.

```{.python .input}
%%tab all
x.shape
```

Chúng ta có thể [**thay đổi shape của một tensor
mà không làm thay đổi size hoặc các giá trị của nó**],
bằng cách gọi `reshape`.
Ví dụ, chúng ta có thể chuyển đổi 
vector `x` có shape là (12,) 
thành một matrix `X` có shape là (3, 4).
Tensor mới này giữ nguyên tất cả các phần tử
nhưng sắp xếp lại chúng thành một matrix.
Hãy lưu ý rằng các phần tử của vector
được sắp xếp lần lượt theo từng hàng và do đó
`x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Lưu ý rằng việc chỉ định mọi thành phần của shape
cho `reshape` là dư thừa.
Vì chúng ta đã biết size của tensor,
chúng ta có thể tính một thành phần của shape dựa trên các thành phần còn lại.
Ví dụ, với một tensor có size $n$
và shape đích là ($h$, $w$),
ta biết rằng $w = n/h$.
Để tự động suy ra một thành phần của shape,
chúng ta có thể đặt `-1` cho thành phần shape
cần được tự động suy ra.
Trong trường hợp này, thay vì gọi `x.reshape(3, 4)`,
chúng ta có thể gọi tương đương `x.reshape(-1, 4)` hoặc `x.reshape(3, -1)`.

Các lập trình viên thường cần làm việc với những tensor
được khởi tạo với toàn bộ phần tử là 0 hoặc 1.
[**Chúng ta có thể tạo một tensor với tất cả các phần tử được đặt thành 0**] (~~hoặc một tensor~~)
và shape là (2, 3, 4) thông qua hàm `zeros`.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

Tương tự, chúng ta có thể tạo một tensor 
với toàn bộ phần tử là 1 bằng cách gọi `ones`.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

Chúng ta thường muốn 
[**lấy mẫu từng phần tử một cách ngẫu nhiên (và độc lập)**] 
từ một phân phối xác suất cho trước.
Ví dụ, các tham số của mạng neural
thường được khởi tạo ngẫu nhiên.
Đoạn mã sau tạo một tensor 
với các phần tử được lấy từ 
phân phối Gaussian (normal) chuẩn
có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

Cuối cùng, chúng ta có thể tạo tensor bằng cách
[**cung cấp chính xác các giá trị cho từng phần tử**] 
thông qua một hoặc nhiều list Python (có thể lồng nhau) 
chứa các literal số.
Ở đây, chúng ta tạo một matrix bằng một list các list,
trong đó list ngoài cùng tương ứng với trục 0,
còn list bên trong tương ứng với trục 1.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indexing và slicing

Tương tự như các list Python,
chúng ta có thể truy cập các phần tử của tensor 
bằng indexing (bắt đầu từ 0).
Để truy cập một phần tử dựa trên vị trí
tính từ cuối list,
chúng ta có thể sử dụng indexing âm.
Cuối cùng, chúng ta có thể truy cập toàn bộ phạm vi các index 
thông qua slicing (ví dụ: `X[start:stop]`), 
trong đó giá trị trả về bao gồm 
index đầu tiên (`start`) *nhưng không bao gồm index cuối cùng* (`stop`).
Cuối cùng, khi chỉ có một index (hoặc slice)
được chỉ định cho một tensor cấp $k^\textrm{th}$,
nó sẽ được áp dụng dọc theo trục 0.
Do đó, trong đoạn mã sau,
[**`[-1]` chọn hàng cuối cùng và `[1:3]`
chọn hàng thứ hai và thứ ba**].

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Ngoài việc đọc các phần tử, (**chúng ta cũng có thể *ghi* các phần tử của một matrix bằng cách chỉ định các index.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` trong TensorFlow là immutable và không thể được gán giá trị.
`Variables` trong TensorFlow là các container trạng thái mutable, hỗ trợ phép gán.
Hãy nhớ rằng gradient trong TensorFlow không truyền ngược
qua các phép gán `Variable`.

Ngoài việc gán một giá trị cho toàn bộ `Variable`, chúng ta có thể ghi các phần tử của một
`Variable` bằng cách chỉ định các index.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

Nếu muốn [**gán cùng một giá trị cho nhiều phần tử,
chúng ta áp dụng indexing ở vế trái 
của phép gán.**]
Ví dụ, `[:2, :]` truy cập 
hàng thứ nhất và thứ hai,
trong khi `:` lấy tất cả các phần tử dọc theo trục 1 (cột).
Mặc dù chúng ta đã thảo luận về indexing cho matrix,
cách này cũng hoạt động với vector
và tensor có nhiều hơn hai chiều.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## Các phép toán

Bây giờ, khi đã biết cách tạo tensor
cũng như cách đọc và ghi các phần tử của chúng,
chúng ta có thể bắt đầu thao tác với chúng
bằng nhiều phép toán toán học khác nhau.
Một trong những phép toán hữu ích nhất 
là các phép toán *elementwise*.
Các phép toán này áp dụng một phép toán vô hướng tiêu chuẩn
cho từng phần tử của tensor.
Đối với các hàm nhận hai tensor làm đầu vào,
các phép toán elementwise áp dụng một toán tử nhị phân tiêu chuẩn
lên từng cặp phần tử tương ứng.
Chúng ta có thể tạo một hàm elementwise 
từ bất kỳ hàm nào ánh xạ 
từ một vô hướng sang một vô hướng.

Trong ký hiệu toán học, chúng ta biểu diễn các toán tử vô hướng *một ngôi* (nhận một đầu vào) như sau:
$f: \mathbb{R} \rightarrow \mathbb{R}$.
Điều này đơn giản có nghĩa là hàm ánh xạ từ một số thực bất kỳ sang một số thực khác.
Hầu hết các toán tử tiêu chuẩn, bao gồm cả các toán tử một ngôi như $e^x$, đều có thể được áp dụng theo từng phần tử.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

Tương tự, chúng ta biểu diễn các toán tử vô hướng *hai ngôi*, ánh xạ các cặp số thực thành một số thực (duy nhất) thông qua chữ ký
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Cho hai vector $\mathbf{u}$ và $\mathbf{v}$ *có cùng shape*, cùng một toán tử hai ngôi $f$, chúng ta có thể tạo ra một vector
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
bằng cách đặt $c_i \gets f(u_i, v_i)$ với mọi $i$,
trong đó $c_i, u_i$ và $v_i$ lần lượt là các phần tử thứ $i^\textrm{th}$ của các vector $\mathbf{c}, \mathbf{u}$ và $\mathbf{v}$.
Ở đây, chúng ta đã tạo ra hàm nhận giá trị vector
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
bằng cách *nâng* hàm vô hướng thành một phép toán vector theo từng phần tử.
Các toán tử số học tiêu chuẩn phổ biến cho phép cộng (`+`), phép trừ (`-`),
phép nhân (`*`), phép chia (`/`)
và phép lũy thừa (`**`)
đều đã được *nâng* thành các phép toán theo từng phần tử
cho các tensor có cùng shape và shape bất kỳ.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

Ngoài các phép tính theo từng phần tử,
chúng ta cũng có thể thực hiện các phép toán đại số tuyến tính,
chẳng hạn như tích vô hướng và phép nhân ma trận.
Chúng ta sẽ trình bày chi tiết hơn về các phép toán này
trong :numref:`sec_linear-algebra`.

Chúng ta cũng có thể [***nối* nhiều tensor,**]
xếp chúng liên tiếp để tạo thành một tensor lớn hơn.
Chúng ta chỉ cần cung cấp một danh sách các tensor
và cho hệ thống biết cần nối theo axis nào.
Ví dụ dưới đây cho thấy điều gì xảy ra khi chúng ta nối
hai ma trận theo các hàng (axis 0)
thay vì theo các cột (axis 1).
Có thể thấy độ dài axis-0 của đầu ra thứ nhất ($6$)
là tổng độ dài axis-0 của hai tensor đầu vào ($3 + 3$);
trong khi độ dài axis-1 của đầu ra thứ hai ($8$)
là tổng độ dài axis-1 của hai tensor đầu vào ($4 + 4$).

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

Đôi khi, chúng ta muốn
[**xây dựng một tensor nhị phân thông qua *các mệnh đề logic*.**]
Lấy `X == Y` làm ví dụ.
Tại mỗi vị trí `i, j`, nếu `X[i, j]` và `Y[i, j]` bằng nhau,
thì phần tử tương ứng trong kết quả nhận giá trị `1`,
ngược lại, nó nhận giá trị `0`.

```{.python .input}
%%tab all
X == Y
```

[**Tính tổng tất cả các phần tử trong tensor**] sẽ tạo ra một tensor chỉ có một phần tử.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Broadcasting
:label:`subsec_broadcasting`

Đến đây, bạn đã biết cách thực hiện
các phép toán hai ngôi theo từng phần tử
trên hai tensor có cùng shape.
Trong một số điều kiện nhất định,
ngay cả khi shape khác nhau,
chúng ta vẫn có thể [**thực hiện các phép toán hai ngôi theo từng phần tử
bằng cách gọi cơ chế *broadcasting*.**]
Broadcasting hoạt động theo
quy trình gồm hai bước sau:
(i) mở rộng một hoặc cả hai mảng
bằng cách sao chép các phần tử dọc theo các axis có độ dài 1
để sau phép biến đổi này,
hai tensor có cùng shape;
(ii) thực hiện phép toán theo từng phần tử
trên các mảng thu được.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

Vì `a` và `b` lần lượt là các ma trận $3\times1$
và $1\times2$,
shape của chúng không khớp nhau.
Broadcasting tạo ra một ma trận lớn hơn có shape $3\times2$
bằng cách lặp lại ma trận `a` dọc theo các cột
và ma trận `b` dọc theo các hàng
trước khi cộng chúng theo từng phần tử.

```{.python .input}
%%tab all
a + b
```

## Tiết kiệm bộ nhớ

[**Việc thực hiện các phép toán có thể khiến bộ nhớ mới được cấp phát cho các kết quả trên host.**]
Ví dụ, nếu chúng ta viết `Y = X + Y`,
chúng ta bỏ tham chiếu tensor mà `Y` từng trỏ tới
và thay vào đó trỏ `Y` tới vùng bộ nhớ vừa được cấp phát.
Chúng ta có thể minh họa vấn đề này bằng hàm `id()` của Python,
hàm này cung cấp cho chúng ta địa chỉ chính xác
của đối tượng được tham chiếu trong bộ nhớ.
Lưu ý rằng sau khi chạy `Y = Y + X`,
`id(Y)` trỏ tới một vị trí khác.
Đó là vì trước tiên Python đánh giá `Y + X`,
cấp phát bộ nhớ mới cho kết quả
rồi trỏ `Y` tới vị trí mới này trong bộ nhớ.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Điều này có thể không mong muốn vì hai lý do.
Thứ nhất, chúng ta không muốn liên tục
cấp phát bộ nhớ một cách không cần thiết.
Trong machine learning, chúng ta thường có
hàng trăm megabyte tham số
và cập nhật tất cả chúng nhiều lần mỗi giây.
Bất cứ khi nào có thể, chúng ta muốn thực hiện các cập nhật này *tại chỗ*.
Thứ hai, nhiều biến có thể cùng trỏ tới
các tham số giống nhau.
Nếu không cập nhật tại chỗ,
chúng ta phải cẩn thận cập nhật tất cả các tham chiếu này,
nếu không có thể gây ra memory leak
hoặc vô tình tham chiếu tới các tham số đã lỗi thời.

:begin_tab:`mxnet, pytorch`
May mắn là (**thực hiện các phép toán tại chỗ**) rất dễ.
Chúng ta có thể gán kết quả của một phép toán
vào một mảng đã được cấp phát trước `Y`
bằng cách sử dụng ký hiệu slice: `Y[:] = <expression>`.
Để minh họa khái niệm này,
chúng ta ghi đè các giá trị của tensor `Z`,
sau khi khởi tạo nó bằng `zeros_like`,
để có cùng shape với `Y`.
:end_tab:

:begin_tab:`tensorflow`
`Variables` là các vùng chứa trạng thái có thể thay đổi trong TensorFlow. Chúng cung cấp
một cách để lưu trữ các tham số của model.
Chúng ta có thể gán kết quả của một phép toán
vào một `Variable` bằng `assign`.
Để minh họa khái niệm này,
chúng ta ghi đè các giá trị của `Variable` `Z`
sau khi khởi tạo nó bằng `zeros_like`,
để có cùng shape với `Y`.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**Nếu giá trị của `X` không được sử dụng lại trong các phép tính tiếp theo,
chúng ta cũng có thể sử dụng `X[:] = X + Y` hoặc `X += Y`
để giảm chi phí bộ nhớ của phép toán.**]
:end_tab:

:begin_tab:`tensorflow`
Ngay cả khi lưu trữ trạng thái lâu dài trong một `Variable`,
bạn vẫn có thể muốn giảm thêm mức sử dụng bộ nhớ bằng cách tránh
cấp phát dư thừa cho các tensor không phải là tham số của model.
Vì `Tensors` của TensorFlow là bất biến
và gradient không lan truyền qua các phép gán `Variable`,
TensorFlow không cung cấp cách rõ ràng để thực hiện
một phép toán riêng lẻ tại chỗ.

Tuy nhiên, TensorFlow cung cấp decorator `tf.function`
để bọc phép tính bên trong một TensorFlow graph
được biên dịch và tối ưu hóa trước khi chạy.
Điều này cho phép TensorFlow loại bỏ các giá trị không được sử dụng
và tái sử dụng các vùng bộ nhớ đã cấp phát trước nhưng không còn cần thiết.
Nhờ đó, chi phí bộ nhớ của các phép tính TensorFlow được giảm thiểu.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Chuyển đổi sang các đối tượng Python khác

:begin_tab:`mxnet, tensorflow`
[**Việc chuyển đổi sang một tensor NumPy (`ndarray`)**], hoặc ngược lại, rất dễ dàng.
Kết quả sau chuyển đổi không dùng chung bộ nhớ.
Sự bất tiện nhỏ này thực ra rất quan trọng:
khi thực hiện các phép toán trên CPU hoặc GPU,
bạn không muốn tạm dừng phép tính để chờ xem
liệu package NumPy của Python
có thể muốn làm việc gì khác
với cùng một vùng bộ nhớ hay không.
:end_tab:

:begin_tab:`pytorch`
[**Việc chuyển đổi sang một tensor NumPy (`ndarray`)**], hoặc ngược lại, rất dễ dàng.
Tensor của torch và mảng NumPy
sẽ dùng chung vùng bộ nhớ bên dưới,
và việc thay đổi một đối tượng thông qua phép toán tại chỗ
cũng sẽ thay đổi đối tượng còn lại.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

Để (**chuyển đổi một tensor có kích thước 1 thành một scalar Python**),
chúng ta có thể gọi hàm `item` hoặc các hàm dựng sẵn của Python.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Tóm tắt

Lớp tensor là giao diện chính để lưu trữ và thao tác với dữ liệu trong các thư viện deep learning.
Tensor cung cấp nhiều chức năng, bao gồm các quy trình khởi tạo; lập chỉ mục và slicing; các phép toán cơ bản; broadcasting; phép gán tiết kiệm bộ nhớ; và chuyển đổi qua lại với các đối tượng Python khác.


## Bài tập

1. Chạy đoạn code trong phần này. Thay đổi mệnh đề điều kiện `X == Y` thành `X < Y` hoặc `X > Y`, rồi xem bạn có thể nhận được loại tensor nào.
1. Thay thế hai tensor hoạt động theo từng phần tử trong cơ chế broadcasting bằng các shape khác, chẳng hạn như tensor 3 chiều. Kết quả có giống như dự kiến không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17966)
:end_tab: