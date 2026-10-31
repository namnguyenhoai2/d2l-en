```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Padding và Stride
:label:`sec_padding`

Hãy nhớ lại ví dụ về một phép tích chập trong :numref:`fig_correlation`. 
Đầu vào có cả chiều cao lẫn chiều rộng bằng 3
và kernel tích chập có cả chiều cao lẫn chiều rộng bằng 2,
tạo ra biểu diễn đầu ra có kích thước $2\times2$.
Giả sử hình dạng đầu vào là $n_\textrm{h}\times n_\textrm{w}$
và hình dạng kernel tích chập là $k_\textrm{h}\times k_\textrm{w}$,
thì hình dạng đầu ra sẽ là $(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1)$: 
ta chỉ có thể dịch kernel tích chập đến một mức nhất định trước khi không còn
pixel nào để áp dụng phép tích chập. 

Sau đây, chúng ta sẽ tìm hiểu một số kỹ thuật, 
bao gồm padding và tích chập có stride,
giúp kiểm soát tốt hơn kích thước đầu ra. 
Để làm rõ động cơ, hãy lưu ý rằng vì kernel thường
có chiều rộng và chiều cao lớn hơn $1$,
sau khi áp dụng nhiều phép tích chập liên tiếp,
đầu ra thường trở nên
nhỏ hơn đáng kể so với đầu vào.
Nếu bắt đầu với một ảnh $240 \times 240$ pixel,
mười lớp tích chập $5 \times 5$
sẽ giảm ảnh xuống còn $200 \times 200$ pixel,
cắt bỏ $30 \%$ ảnh và cùng với đó
xóa mất mọi thông tin thú vị
ở các biên của ảnh gốc.
*Padding* là công cụ phổ biến nhất để xử lý vấn đề này.
Trong các trường hợp khác, ta có thể muốn giảm số chiều một cách mạnh mẽ,
ví dụ, nếu thấy độ phân giải đầu vào ban đầu quá cồng kềnh.
*Tích chập có stride* là một kỹ thuật phổ biến có thể hữu ích trong những trường hợp này.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Padding

Như đã mô tả ở trên, một vấn đề khó khi áp dụng các lớp tích chập
là ta thường làm mất pixel ở chu vi của ảnh. Hãy xem :numref:`img_conv_reuse`, minh họa mức độ sử dụng pixel theo kích thước kernel tích chập và vị trí trong ảnh. Các pixel ở góc hầu như không được sử dụng. 

![Mức độ sử dụng pixel cho các phép tích chập có kích thước lần lượt là $1 \times 1$, $2 \times 2$ và $3 \times 3$.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

Vì ta thường sử dụng các kernel nhỏ,
trong mỗi phép tích chập nhất định
ta có thể chỉ mất vài pixel,
nhưng điều này có thể tích lũy khi áp dụng
nhiều lớp tích chập liên tiếp.
Một giải pháp trực tiếp cho vấn đề này
là thêm các pixel đệm xung quanh biên của ảnh đầu vào,
qua đó tăng kích thước hiệu dụng của ảnh.
Thông thường, ta đặt giá trị của các pixel bổ sung bằng không.
Trong :numref:`img_conv_pad`, ta padding đầu vào $3 \times 3$,
tăng kích thước của nó lên $5 \times 5$.
Đầu ra tương ứng sau đó tăng thành ma trận $4 \times 4$.
Các phần được tô bóng là phần tử đầu ra đầu tiên, cũng như các phần tử tensor đầu vào và kernel được dùng để tính đầu ra: $0\times0+0\times1+0\times2+0\times3=0$.

![Tương quan chéo hai chiều với padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

Nói chung, nếu ta thêm tổng cộng $p_\textrm{h}$ hàng padding
(xấp xỉ một nửa ở trên và một nửa ở dưới)
và tổng cộng $p_\textrm{w}$ cột padding
(xấp xỉ một nửa ở bên trái và một nửa ở bên phải),
thì hình dạng đầu ra sẽ là

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$

Điều này có nghĩa là chiều cao và chiều rộng của đầu ra
sẽ lần lượt tăng thêm $p_\textrm{h}$ và $p_\textrm{w}$.

Trong nhiều trường hợp, ta sẽ muốn đặt $p_\textrm{h}=k_\textrm{h}-1$ và $p_\textrm{w}=k_\textrm{w}-1$
để đầu vào và đầu ra có cùng chiều cao và chiều rộng.
Điều này giúp dễ dự đoán hình dạng đầu ra của mỗi lớp hơn
khi xây dựng mạng.
Giả sử $k_\textrm{h}$ là số lẻ ở đây,
ta sẽ padding $p_\textrm{h}/2$ hàng ở cả hai phía của chiều cao.
Nếu $k_\textrm{h}$ là số chẵn, một khả năng là
padding $\lceil p_\textrm{h}/2\rceil$ hàng ở phía trên đầu vào
và $\lfloor p_\textrm{h}/2\rfloor$ hàng ở phía dưới.
Ta sẽ padding cả hai phía của chiều rộng theo cùng cách.

CNN thường sử dụng các kernel tích chập
có giá trị chiều cao và chiều rộng lẻ, chẳng hạn như 1, 3, 5 hoặc 7.
Việc chọn kích thước kernel lẻ có lợi ích
là ta có thể bảo toàn số chiều
khi padding cùng số hàng ở trên và dưới,
cũng như cùng số cột ở bên trái và bên phải.

Hơn nữa, cách sử dụng kernel lẻ này
và padding để bảo toàn chính xác số chiều
còn mang lại một lợi ích về mặt thao tác.
Với bất kỳ tensor hai chiều `X` nào,
khi kích thước kernel là số lẻ
và số hàng, cột padding
ở mọi phía là như nhau,
từ đó tạo ra đầu ra có cùng chiều cao và chiều rộng với đầu vào,
ta biết rằng đầu ra `Y[i, j]` được tính
bằng tương quan chéo của đầu vào và kernel tích chập
với cửa sổ được căn giữa tại `X[i, j]`.

Trong ví dụ sau, ta tạo một lớp tích chập hai chiều
có chiều cao và chiều rộng bằng 3
và (**áp dụng 1 pixel padding ở mọi phía.**)
Với đầu vào có chiều cao và chiều rộng bằng 8,
ta nhận thấy chiều cao và chiều rộng của đầu ra cũng bằng 8.

```{.python .input}
%%tab mxnet
# We define a helper function to calculate convolutions. It initializes 
# the convolutional layer weights and performs corresponding dimensionality 
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# We define a helper function to calculate convolutions. It initializes the
# convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# We define a helper function to calculate convolutions. It initializes
# the convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return tf.reshape(Y, Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# We define a helper function to calculate convolutions. It initializes
# the convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, X.shape, 1) indicates that batch size and the number of channels are both 1
    key = jax.random.PRNGKey(d2l.get_seed())
    X = X.reshape((1,) + X.shape + (1,))
    Y, _ = conv2d.init_with_output(key, X)
    # Strip the dimensions: examples and channels
    return Y.reshape(Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

Khi chiều cao và chiều rộng của kernel tích chập khác nhau,
ta có thể làm cho đầu ra và đầu vào có cùng chiều cao và chiều rộng
bằng cách [**đặt số lượng padding khác nhau cho chiều cao và chiều rộng.**]

```{.python .input}
%%tab mxnet
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# We use a convolution kernel with height 5 and width 3. The padding on either
# side of the height and width are 2 and 1, respectively
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = nn.Conv(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## Stride

Khi tính tương quan chéo,
ta bắt đầu với cửa sổ tích chập
ở góc trên bên trái của tensor đầu vào,
sau đó trượt nó qua tất cả vị trí theo cả hướng xuống và sang phải.
Trong các ví dụ trước, mặc định ta trượt mỗi lần một phần tử.
Tuy nhiên, đôi khi, để đạt hiệu quả tính toán
hoặc vì muốn downsample,
ta di chuyển cửa sổ nhiều hơn một phần tử mỗi lần,
bỏ qua các vị trí ở giữa. Điều này đặc biệt hữu ích nếu kernel tích chập 
lớn vì nó bao quát một vùng rộng của ảnh bên dưới.

Ta gọi số hàng và cột đi qua trong mỗi lần trượt là *stride*.
Cho đến nay, ta đã sử dụng stride bằng 1 cho cả chiều cao lẫn chiều rộng.
Đôi khi, ta có thể muốn sử dụng stride lớn hơn.
:numref:`img_conv_stride` cho thấy một phép tương quan chéo hai chiều
với stride bằng 3 theo chiều dọc và 2 theo chiều ngang.
Các phần được tô bóng là các phần tử đầu ra, cũng như các phần tử tensor đầu vào và kernel được dùng để tính đầu ra: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
Ta có thể thấy rằng khi phần tử thứ hai của cột đầu tiên được tạo ra,
cửa sổ tích chập trượt xuống ba hàng.
Cửa sổ tích chập trượt sang phải hai cột
khi phần tử thứ hai của hàng đầu tiên được tạo ra.
Khi cửa sổ tích chập tiếp tục trượt sang phải hai cột trên đầu vào,
sẽ không có đầu ra vì phần tử đầu vào không thể lấp đầy cửa sổ
(trừ khi ta thêm một cột padding khác).

![Tương quan chéo với stride lần lượt bằng 3 và 2 cho chiều cao và chiều rộng.](../img/conv-stride.svg)
:label:`img_conv_stride`

Nói chung, khi stride cho chiều cao là $s_\textrm{h}$
và stride cho chiều rộng là $s_\textrm{w}$, hình dạng đầu ra là

$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$

Nếu đặt $p_\textrm{h}=k_\textrm{h}-1$ và $p_\textrm{w}=k_\textrm{w}-1$,
thì hình dạng đầu ra có thể được đơn giản hóa thành
$\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$.
Đi xa hơn một bước, nếu chiều cao và chiều rộng đầu vào
chia hết cho stride của chiều cao và chiều rộng,
thì hình dạng đầu ra sẽ là $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$.

Dưới đây, ta [**đặt stride cho cả chiều cao và chiều rộng bằng 2**],
do đó giảm một nửa chiều cao và chiều rộng đầu vào.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 3), padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

Hãy xem (**một ví dụ phức tạp hơn một chút**).

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## Tóm tắt và Thảo luận

Padding có thể làm tăng chiều cao và chiều rộng của đầu ra. Điều này thường được dùng để làm cho đầu ra có cùng chiều cao và chiều rộng với đầu vào, nhằm tránh việc đầu ra bị thu nhỏ không mong muốn. Hơn nữa, nó đảm bảo rằng tất cả các pixel được sử dụng với tần suất như nhau. Thông thường, ta chọn padding đối xứng ở cả hai phía của chiều cao và chiều rộng đầu vào. Trong trường hợp này, ta gọi là padding $(p_\textrm{h}, p_\textrm{w})$. Phổ biến nhất, ta đặt $p_\textrm{h} = p_\textrm{w}$, trong trường hợp đó ta chỉ đơn giản nói rằng ta chọn padding $p$. 

Một quy ước tương tự cũng áp dụng cho stride. Khi stride ngang $s_\textrm{h}$ và stride dọc $s_\textrm{w}$ giống nhau, ta chỉ nói về stride $s$. Stride có thể làm giảm độ phân giải của đầu ra, chẳng hạn giảm chiều cao và chiều rộng của đầu ra xuống chỉ còn $1/n$ chiều cao và chiều rộng của đầu vào với $n > 1$. Theo mặc định, padding là 0 và stride là 1. 

Cho đến nay, mọi padding mà ta đã thảo luận đều chỉ mở rộng ảnh bằng các số 0. Điều này mang lại lợi ích tính toán đáng kể vì rất dễ thực hiện. Hơn nữa, các toán tử có thể được thiết kế để tận dụng padding này một cách ngầm định mà không cần cấp phát thêm bộ nhớ. Đồng thời, nó cho phép CNN mã hóa thông tin vị trí ngầm định trong ảnh, chỉ bằng cách học vị trí của "khoảng trắng". Có nhiều lựa chọn thay thế cho zero-padding. :citet:`Alsallakh.Kokhlikyan.Miglani.ea.2020` đã cung cấp một tổng quan rộng rãi về các lựa chọn đó (dù chưa đưa ra trường hợp rõ ràng về thời điểm nên dùng padding khác 0, trừ khi xuất hiện artifacts). 


## Bài tập

1. Với ví dụ code cuối cùng trong phần này có kernel size $(3, 5)$, padding $(0, 1)$ và stride $(3, 4)$, 
   hãy tính output shape để kiểm tra xem nó có nhất quán với kết quả thực nghiệm hay không.
1. Đối với tín hiệu âm thanh, stride bằng 2 tương ứng với điều gì?
1. Hãy triển khai mirror padding, tức là padding trong đó các giá trị biên được phản chiếu đơn giản để mở rộng tensor. 
1. Lợi ích tính toán của stride lớn hơn 1 là gì?
1. Lợi ích thống kê của stride lớn hơn 1 có thể là gì?
1. Bạn sẽ triển khai stride $\frac{1}{2}$ như thế nào? Nó tương ứng với điều gì? Khi nào điều này hữu ích?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/272)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17997)
:end_tab: