```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Nhiều Kênh Đầu vào và Nhiều Kênh Đầu ra
:label:`sec_channels`

Mặc dù chúng ta đã mô tả các kênh đa dạng
cấu thành mỗi hình ảnh (ví dụ, ảnh màu có các kênh RGB tiêu chuẩn
để biểu thị lượng màu đỏ, xanh lục và xanh lam) và các lớp tích chập dành cho nhiều kênh trong :numref:`subsec_why-conv-channels`,
cho đến nay, chúng ta đã đơn giản hóa tất cả các ví dụ số
bằng cách chỉ làm việc với một kênh đầu vào và một kênh đầu ra.
Điều này cho phép chúng ta xem đầu vào, kernel tích chập,
và đầu ra đều là các tensor hai chiều.

Khi thêm các kênh vào,
đầu vào và biểu diễn ẩn của chúng ta
đều trở thành các tensor ba chiều.
Ví dụ, mỗi ảnh đầu vào RGB có hình dạng $3\times h\times w$.
Chúng ta gọi trục này, có kích thước 3, là chiều *kênh* (channel). Khái niệm về
kênh đã có từ thời CNNs xuất hiện: ví dụ, LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995` sử dụng chúng.
Trong phần này, chúng ta sẽ xem xét kỹ hơn
các kernel tích chập có nhiều kênh đầu vào và nhiều kênh đầu ra.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Nhiều Kênh Đầu vào

Khi dữ liệu đầu vào chứa nhiều kênh,
chúng ta cần xây dựng một kernel tích chập
có cùng số lượng kênh đầu vào với dữ liệu đầu vào,
để có thể thực hiện cross-correlation với dữ liệu đầu vào.
Giả sử số lượng kênh của dữ liệu đầu vào là $c_\textrm{i}$,
số lượng kênh đầu vào của kernel tích chập cũng phải là $c_\textrm{i}$. Nếu hình dạng cửa sổ của kernel tích chập là $k_\textrm{h}\times k_\textrm{w}$,
thì khi $c_\textrm{i}=1$, chúng ta có thể xem kernel tích chập
đơn giản là một tensor hai chiều có hình dạng $k_\textrm{h}\times k_\textrm{w}$.

Tuy nhiên, khi $c_\textrm{i}>1$, chúng ta cần một kernel
chứa một tensor có hình dạng $k_\textrm{h}\times k_\textrm{w}$ cho *mỗi* kênh đầu vào. Nối $c_\textrm{i}$ tensor này lại với nhau
sẽ tạo ra một kernel tích chập có hình dạng $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$.
Vì đầu vào và kernel tích chập đều có $c_\textrm{i}$ kênh,
chúng ta có thể thực hiện phép cross-correlation
trên tensor hai chiều của đầu vào
và tensor hai chiều của kernel tích chập
cho từng kênh, sau đó cộng $c_\textrm{i}$ kết quả lại với nhau
(cộng theo các kênh)
để tạo ra một tensor hai chiều.
Đây là kết quả của phép cross-correlation hai chiều
giữa đầu vào đa kênh và
kernel tích chập có nhiều kênh đầu vào.

:numref:`fig_conv_multi_in` cung cấp một ví dụ
về phép cross-correlation hai chiều với hai kênh đầu vào.
Các phần được tô bóng là phần tử đầu ra đầu tiên
cũng như các phần tử tensor đầu vào và kernel được dùng để tính đầu ra:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Tính toán cross-correlation với hai kênh đầu vào.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


Để bảo đảm rằng chúng ta thực sự hiểu điều đang diễn ra ở đây,
chúng ta có thể tự (**triển khai các phép toán cross-correlation với nhiều kênh đầu vào**).
Lưu ý rằng tất cả những gì chúng ta làm là thực hiện phép cross-correlation
cho mỗi kênh rồi cộng các kết quả lại.

```{.python .input}
%%tab mxnet, pytorch, jax
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
%%tab tensorflow
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

Chúng ta có thể xây dựng tensor đầu vào `X` và tensor kernel `K`
tương ứng với các giá trị trong :numref:`fig_conv_multi_in`
để (**xác thực đầu ra**) của phép toán cross-correlation.

```{.python .input}
%%tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Nhiều Kênh Đầu ra
:label:`subsec_multi-output-channels`

Bất kể số lượng kênh đầu vào là bao nhiêu,
cho đến nay chúng ta luôn chỉ thu được một kênh đầu ra.
Tuy nhiên, như đã thảo luận trong :numref:`subsec_why-conv-channels`,
việc có nhiều kênh ở mỗi lớp hóa ra là thiết yếu.
Trong các kiến trúc mạng nơ-ron phổ biến nhất,
chúng ta thực tế tăng chiều kênh
khi đi sâu hơn vào mạng nơ-ron,
thường thực hiện downsampling để đánh đổi độ phân giải không gian
lấy *độ sâu kênh* lớn hơn.
Theo trực giác, bạn có thể xem mỗi kênh
là phản hồi với một tập đặc trưng khác nhau.
Thực tế phức tạp hơn một chút. Một cách diễn giải ngây thơ có thể cho rằng
các biểu diễn được học độc lập theo từng pixel hoặc từng kênh.
Thay vào đó, các kênh được tối ưu để cùng hữu ích.
Điều này có nghĩa là thay vì ánh xạ một kênh đơn lẻ tới một bộ phát hiện cạnh, nó có thể đơn giản có nghĩa
rằng một hướng nào đó trong không gian kênh tương ứng với việc phát hiện cạnh.

Ký hiệu $c_\textrm{i}$ và $c_\textrm{o}$ lần lượt là số lượng
kênh đầu vào và đầu ra,
và $k_\textrm{h}$ và $k_\textrm{w}$ là chiều cao và chiều rộng của kernel.
Để có đầu ra với nhiều kênh,
chúng ta có thể tạo một tensor kernel
có hình dạng $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$
cho *mỗi* kênh đầu ra.
Chúng ta nối chúng theo chiều kênh đầu ra,
để hình dạng của kernel tích chập
là $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$.
Trong các phép toán cross-correlation,
kết quả trên mỗi kênh đầu ra được tính
từ kernel tích chập tương ứng với kênh đầu ra đó
và nhận đầu vào từ tất cả các kênh trong tensor đầu vào.

Chúng ta triển khai một hàm cross-correlation
để [**tính đầu ra của nhiều kênh**] như dưới đây.

```{.python .input}
%%tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Chúng ta xây dựng một kernel tích chập đơn giản với ba kênh đầu ra
bằng cách nối tensor kernel cho `K` với `K+1` và `K+2`.

```{.python .input}
%%tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Dưới đây, chúng ta thực hiện các phép toán cross-correlation
trên tensor đầu vào `X` với tensor kernel `K`.
Bây giờ đầu ra chứa ba kênh.
Kết quả của kênh đầu tiên nhất quán
với kết quả của tensor đầu vào `X` trước đó
và kernel nhiều kênh đầu vào,
một kênh đầu ra.

```{.python .input}
%%tab all
corr2d_multi_in_out(X, K)
```

## Lớp Tích chập $1\times 1$
:label:`subsec_1x1`

Thoạt đầu, một [**phép tích chập $1 \times 1$**], tức là $k_\textrm{h} = k_\textrm{w} = 1$,
dường như không có nhiều ý nghĩa.
Xét cho cùng, một phép tích chập tương quan các pixel lân cận.
Rõ ràng phép tích chập $1 \times 1$ thì không.
Tuy nhiên, đây là các phép toán phổ biến đôi khi được đưa vào
thiết kế của các mạng sâu phức tạp :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`.
Hãy xem chi tiết nó thực sự làm gì.

Vì sử dụng cửa sổ nhỏ nhất,
phép tích chập $1\times 1$ mất đi khả năng
của các lớp tích chập lớn hơn trong việc nhận diện
các mẫu gồm tương tác
giữa các phần tử lân cận theo chiều cao và chiều rộng.
Phép tính duy nhất của tích chập $1\times 1$ diễn ra
trên chiều kênh.

:numref:`fig_conv_1x1` minh họa phép tính cross-correlation
sử dụng kernel tích chập $1\times 1$
với 3 kênh đầu vào và 2 kênh đầu ra.
Lưu ý rằng đầu vào và đầu ra có cùng chiều cao và chiều rộng.
Mỗi phần tử trong đầu ra được suy ra
từ một tổ hợp tuyến tính của các phần tử *tại cùng vị trí*
trong ảnh đầu vào.
Bạn có thể xem lớp tích chập $1\times 1$
như một lớp fully connected được áp dụng tại từng vị trí pixel
để biến đổi $c_\textrm{i}$ giá trị đầu vào tương ứng thành $c_\textrm{o}$ giá trị đầu ra.
Vì đây vẫn là một lớp tích chập,
các trọng số được chia sẻ giữa các vị trí pixel.
Do đó, lớp tích chập $1\times 1$ cần $c_\textrm{o}\times c_\textrm{i}$ trọng số
(cộng thêm bias). Cũng lưu ý rằng các lớp tích chập thường được theo sau
bởi các hàm phi tuyến. Điều này đảm bảo rằng các phép tích chập $1 \times 1$ không thể đơn giản
được gộp vào các phép tích chập khác.

![Phép tính cross-correlation sử dụng kernel tích chập $1\times 1$ với ba kênh đầu vào và hai kênh đầu ra. Đầu vào và đầu ra có cùng chiều cao và chiều rộng.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Hãy kiểm tra xem điều này có hoạt động trong thực tế không:
chúng ta triển khai phép tích chập $1 \times 1$
bằng một lớp fully connected.
Điều duy nhất là chúng ta cần thực hiện một số điều chỉnh
đối với hình dạng dữ liệu trước và sau phép nhân ma trận.

```{.python .input}
%%tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

Khi thực hiện các phép tích chập $1\times 1$,
hàm trên tương đương với hàm cross-correlation `corr2d_multi_in_out` đã được triển khai trước đó.
Hãy kiểm tra điều này với một số dữ liệu mẫu.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab jax
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (3, 3, 3)) + 0 * 1
K = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 3, 1, 1)) + 0 * 1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Thảo luận

Các kênh cho phép chúng ta kết hợp những ưu điểm tốt nhất của cả hai thế giới: MLPs cho phép các phi tuyến tính đáng kể và tích chập cho phép phân tích đặc trưng *cục bộ* (localized). Đặc biệt, các kênh cho phép CNN suy luận với nhiều đặc trưng, chẳng hạn như bộ phát hiện cạnh và hình dạng, cùng lúc. Chúng cũng mang lại sự đánh đổi thực tế giữa việc giảm mạnh số lượng tham số nhờ tính bất biến theo phép tịnh tiến và tính cục bộ, với nhu cầu về các mô hình biểu đạt và đa dạng trong thị giác máy tính. 

Tuy nhiên, lưu ý rằng tính linh hoạt này phải đánh đổi bằng chi phí. Với một ảnh có kích thước $(h \times w)$, chi phí để tính một phép tích chập $k \times k$ là $\mathcal{O}(h \cdot w \cdot k^2)$. Với $c_\textrm{i}$ và $c_\textrm{o}$ là số kênh đầu vào và đầu ra tương ứng, chi phí này tăng lên thành $\mathcal{O}(h \cdot w \cdot k^2 \cdot c_\textrm{i} \cdot c_\textrm{o})$. Đối với một ảnh $256 \times 256$ pixel với kernel $5 \times 5$ và lần lượt $128$ kênh đầu vào và đầu ra, con số này lên tới hơn 53 tỷ phép toán (chúng ta tính riêng phép nhân và phép cộng). Sau này, chúng ta sẽ gặp các chiến lược hiệu quả để giảm chi phí, ví dụ bằng cách yêu cầu các phép toán theo kênh có dạng block-diagonal, dẫn đến các kiến trúc như ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`. 

## Bài tập

1. Giả sử chúng ta có hai kernel tích chập với kích thước lần lượt là $k_1$ và $k_2$ 
   (không có phi tuyến nào ở giữa).
    1. Chứng minh rằng kết quả của phép toán có thể được biểu diễn bằng một phép tích chập duy nhất.
    1. Số chiều của phép tích chập duy nhất tương đương là bao nhiêu?
    1. Điều ngược lại có đúng không, tức là bạn có luôn có thể phân rã một phép tích chập thành hai phép nhỏ hơn không?
1. Giả sử đầu vào có dạng $c_\textrm{i}\times h\times w$ và kernel tích chập có dạng 
   $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$, padding là $(p_\textrm{h}, p_\textrm{w})$, và stride là $(s_\textrm{h}, s_\textrm{w})$.
    1. Chi phí tính toán (phép nhân và phép cộng) cho forward propagation là bao nhiêu?
    1. Memory footprint là bao nhiêu?
    1. Memory footprint cho phép tính backward là bao nhiêu?
    1. Chi phí tính toán cho backpropagation là bao nhiêu?
1. Số lượng phép tính tăng theo hệ số bao nhiêu nếu ta tăng gấp đôi cả số kênh đầu vào $c_\textrm{i}$ lẫn số kênh đầu ra $c_\textrm{o}$? Điều gì xảy ra nếu ta tăng gấp đôi padding?
1. Các biến `Y1` và `Y2` trong ví dụ cuối cùng của phần này có hoàn toàn giống nhau không? Tại sao?
1. Biểu diễn các phép tích chập dưới dạng phép nhân ma trận, kể cả khi cửa sổ tích chập không phải là $1 \times 1$. 
1. Nhiệm vụ của bạn là triển khai các phép tích chập nhanh với kernel $k \times k$. Một trong các thuật toán khả thi 
   là quét theo chiều ngang trên nguồn, đọc một dải có chiều rộng $k$ và tính dải đầu ra rộng $1$ 
   từng giá trị một. Phương án thay thế là đọc một dải có chiều rộng $k + \Delta$ và tính một dải 
   đầu ra rộng $\Delta$. Tại sao phương án sau tốt hơn? Có giới hạn nào cho độ lớn của $\Delta$ mà bạn nên chọn không?
1. Giả sử chúng ta có một ma trận $c \times c$. 
    1. Việc nhân với một ma trận block-diagonal nhanh hơn bao nhiêu nếu ma trận được chia thành $b$ block?
    1. Nhược điểm của việc có $b$ block là gì? Bạn có thể khắc phục điều đó, ít nhất là một phần, như thế nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/273)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17998)
:end_tab: