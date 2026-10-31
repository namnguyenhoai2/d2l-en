```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Perceptron đa tầng
:label:`sec_mlp`

Trong :numref:`sec_softmax`, chúng ta đã giới thiệu
hồi quy softmax,
triển khai thuật toán từ đầu
(:numref:`sec_softmax_scratch`) và sử dụng các API cấp cao
(:numref:`sec_softmax_concise`). Điều này cho phép chúng ta
huấn luyện các bộ phân loại có khả năng nhận diện
10 loại trang phục từ các hình ảnh độ phân giải thấp.
Trong quá trình đó, chúng ta đã học cách xử lý dữ liệu,
ép đầu ra thành một phân phối xác suất hợp lệ,
áp dụng một hàm mất mát phù hợp,
và tối thiểu hóa hàm này theo các tham số của mô hình.
Giờ đây, sau khi đã nắm vững các cơ chế này
trong bối cảnh của các mô hình tuyến tính đơn giản,
chúng ta có thể bắt đầu khám phá mạng nơ-ron sâu,
lớp mô hình phong phú hơn đáng kể
mà cuốn sách này chủ yếu tập trung vào.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Lớp ẩn

Trong :numref:`subsec_linear_model`, chúng ta đã mô tả
các phép biến đổi affine là
các phép biến đổi tuyến tính có thêm bias.
Để bắt đầu, hãy nhắc lại kiến trúc mô hình
tương ứng với ví dụ hồi quy softmax của chúng ta,
được minh họa trong :numref:`fig_softmaxreg`.
Mô hình này ánh xạ trực tiếp đầu vào sang đầu ra
thông qua một phép biến đổi affine duy nhất,
tiếp theo là một phép toán softmax.
Nếu nhãn thực sự có quan hệ với
dữ liệu đầu vào qua một phép biến đổi affine đơn giản,
thì cách tiếp cận này là đủ.
Tuy nhiên, tính tuyến tính (trong các phép biến đổi affine) là một giả định *mạnh*.

### Các hạn chế của mô hình tuyến tính

Ví dụ, tính tuyến tính hàm ý giả định *yếu hơn*
về *tính đơn điệu*, tức là
bất kỳ sự tăng nào của feature cũng phải
hoặc luôn làm tăng đầu ra của mô hình
(nếu trọng số tương ứng dương),
hoặc luôn làm giảm đầu ra của mô hình
(nếu trọng số tương ứng âm).
Đôi khi điều này hợp lý.
Ví dụ, nếu chúng ta cố gắng dự đoán
liệu một cá nhân có hoàn trả khoản vay hay không,
ta có thể hợp lý khi giả định rằng, nếu các yếu tố khác không đổi,
một người nộp đơn có thu nhập cao hơn
sẽ luôn có khả năng hoàn trả cao hơn
người có thu nhập thấp hơn.
Dù đơn điệu, mối quan hệ này có lẽ
không liên hệ tuyến tính với xác suất
hoàn trả. Việc tăng thu nhập từ \$0 lên \$50,000
có lẽ tương ứng với mức tăng lớn hơn
về khả năng hoàn trả
so với việc tăng từ \$1 triệu lên \$1.05 triệu.
Một cách để xử lý điều này có thể là hậu xử lý kết quả
để tính tuyến tính trở nên hợp lý hơn,
bằng cách sử dụng ánh xạ logistic (và do đó là logarit của xác suất kết quả).

Lưu ý rằng chúng ta có thể dễ dàng đưa ra các ví dụ
vi phạm tính đơn điệu.
Chẳng hạn, giả sử chúng ta muốn dự đoán sức khỏe theo
nhiệt độ cơ thể.
Với những người có nhiệt độ cơ thể bình thường
trên 37°C (98.6°F),
nhiệt độ cao hơn cho thấy rủi ro lớn hơn.
Tuy nhiên, nếu nhiệt độ cơ thể giảm
xuống dưới 37°C, nhiệt độ thấp hơn lại cho thấy rủi ro lớn hơn!
Một lần nữa, chúng ta có thể giải quyết vấn đề
bằng một số tiền xử lý khéo léo, chẳng hạn dùng khoảng cách đến 37°C
làm feature.


Nhưng còn việc phân loại hình ảnh mèo và chó thì sao?
Việc tăng cường độ
của pixel tại vị trí (13, 17)
có phải luôn làm tăng (hoặc luôn làm giảm)
khả năng hình ảnh đó mô tả một con chó không?
Việc dựa vào mô hình tuyến tính tương ứng với giả định ngầm
rằng yêu cầu duy nhất
để phân biệt mèo và chó là đánh giá
độ sáng của từng pixel riêng lẻ.
Cách tiếp cận này chắc chắn thất bại trong một thế giới
mà việc đảo ngược hình ảnh vẫn giữ nguyên danh mục.

Tuy vậy, mặc dù tính tuyến tính trong trường hợp này có vẻ phi lý
so với các ví dụ trước đó của chúng ta,
không rõ ràng rằng ta có thể xử lý vấn đề
bằng một cách sửa tiền xử lý đơn giản.
Đó là vì ý nghĩa của bất kỳ pixel nào
phụ thuộc phức tạp vào ngữ cảnh của nó
(các giá trị của những pixel xung quanh).
Mặc dù có thể tồn tại một cách biểu diễn dữ liệu
có tính đến
các tương tác liên quan giữa các feature,
mà trên đó mô hình tuyến tính sẽ phù hợp,
chúng ta đơn giản là không biết cách tính nó thủ công.
Với mạng nơ-ron sâu, chúng ta sử dụng dữ liệu quan sát
để cùng lúc học cả biểu diễn thông qua các lớp ẩn
và một bộ dự đoán tuyến tính hoạt động trên biểu diễn đó.

Vấn đề phi tuyến tính này đã được nghiên cứu ít nhất trong
một thế kỷ :cite:`Fisher.1928`. Ví dụ, cây quyết định
ở dạng cơ bản nhất sử dụng một chuỗi quyết định nhị phân để
quyết định việc thuộc về một lớp :cite:`quinlan2014c4`. Tương tự, các phương pháp kernel
đã được sử dụng trong nhiều thập kỷ để mô hình hóa các phụ thuộc phi tuyến
:cite:`Aronszajn.1950`. Điều này đã được đưa vào
các mô hình spline phi tham số :cite:`Wahba.1990` và các phương pháp kernel
:cite:`Scholkopf.Smola.2002`. Đây cũng là điều mà não bộ giải quyết
khá tự nhiên. Xét cho cùng, các nơ-ron truyền tín hiệu đến các nơ-ron khác,
rồi các nơ-ron này lại truyền tín hiệu đến các nơ-ron khác nữa :cite:`Cajal.Azoulay.1894`.
Do đó, chúng ta có một chuỗi các phép biến đổi tương đối đơn giản.

### Tích hợp các lớp ẩn

Chúng ta có thể vượt qua các hạn chế của mô hình tuyến tính
bằng cách tích hợp một hoặc nhiều lớp ẩn.
Cách dễ nhất để làm điều này là xếp chồng
nhiều lớp kết nối đầy đủ lên nhau.
Mỗi lớp truyền đầu ra vào lớp phía trên,
cho đến khi tạo ra đầu ra cuối cùng.
Ta có thể xem $L-1$ lớp đầu tiên
là biểu diễn của mình và lớp cuối cùng
là bộ dự đoán tuyến tính.
Kiến trúc này thường được gọi là
*mạng perceptron đa tầng*,
thường viết tắt là *MLP* (:numref:`fig_mlp`).

![Một MLP có một lớp ẩn gồm năm đơn vị ẩn.](../img/mlp.svg)
:label:`fig_mlp`

MLP này có bốn đầu vào, ba đầu ra,
và lớp ẩn của nó chứa năm đơn vị ẩn.
Vì lớp đầu vào không thực hiện bất kỳ phép tính nào,
việc tạo đầu ra bằng mạng này
đòi hỏi triển khai các phép tính
cho cả lớp ẩn lẫn lớp đầu ra;
do đó, số lớp trong MLP này là hai.
Lưu ý rằng cả hai lớp đều được kết nối đầy đủ.
Mọi đầu vào đều ảnh hưởng đến mọi nơ-ron trong lớp ẩn,
và mỗi nơ-ron trong số này lại ảnh hưởng đến
mọi nơ-ron trong lớp đầu ra. Tuy nhiên, chúng ta vẫn chưa
hoàn tất.

### Từ tuyến tính sang phi tuyến

Như trước đây, chúng ta ký hiệu ma trận $\mathbf{X} \in \mathbb{R}^{n \times d}$
là một minibatch gồm $n$ mẫu, trong đó mỗi mẫu có $d$ đầu vào (feature).
Với một MLP có một lớp ẩn mà lớp ẩn có $h$ đơn vị ẩn,
chúng ta ký hiệu $\mathbf{H} \in \mathbb{R}^{n \times h}$
là đầu ra của lớp ẩn, tức là
*các biểu diễn ẩn*.
Vì cả lớp ẩn và lớp đầu ra đều được kết nối đầy đủ,
ta có trọng số lớp ẩn $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ và bias $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
cùng trọng số lớp đầu ra $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ và bias $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.
Điều này cho phép chúng ta tính đầu ra $\mathbf{O} \in \mathbb{R}^{n \times q}$
của MLP một lớp ẩn như sau:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Lưu ý rằng sau khi thêm lớp ẩn,
mô hình của chúng ta giờ đây yêu cầu theo dõi và cập nhật
các tập tham số bổ sung.
Vậy đổi lại chúng ta thu được gì?
Bạn có thể ngạc nhiên khi biết rằng
---trong mô hình được định nghĩa ở trên---*chúng ta
không thu được gì cho công sức của mình*!
Lý do rất rõ ràng.
Các đơn vị ẩn ở trên được xác định bởi
một hàm affine của đầu vào,
và đầu ra (trước softmax) chỉ là
một hàm affine của các đơn vị ẩn.
Một hàm affine của một hàm affine
vẫn là một hàm affine.
Hơn nữa, mô hình tuyến tính của chúng ta vốn đã
có thể biểu diễn bất kỳ hàm affine nào.

Để thấy điều này một cách hình thức, ta chỉ cần triệt tiêu lớp ẩn trong định nghĩa trên,
thu được một mô hình một lớp tương đương với các tham số
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ và $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

Để hiện thực hóa tiềm năng của các kiến trúc nhiều tầng,
chúng ta cần thêm một thành phần then chốt: một
*hàm kích hoạt* (activation function) phi tuyến $\sigma$
được áp dụng cho mỗi đơn vị ẩn
sau phép biến đổi affine. Chẳng hạn, một lựa chọn
phổ biến là hàm kích hoạt ReLU (rectified linear unit) :cite:`Nair.Hinton.2010`
$\sigma(x) = \mathrm{max}(0, x)$ hoạt động trên từng phần tử của các đối số.
Đầu ra của các hàm kích hoạt $\sigma(\cdot)$
được gọi là *activation*.
Nhìn chung, khi có các hàm kích hoạt,
chúng ta không còn có thể rút gọn MLP thành một mô hình tuyến tính:

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Vì mỗi hàng trong $\mathbf{X}$ tương ứng với một ví dụ trong minibatch,
với một chút lạm dụng ký hiệu, chúng ta định nghĩa hàm phi tuyến
$\sigma$ áp dụng lên các đầu vào của nó theo từng hàng,
tức là mỗi lần một ví dụ.
Lưu ý rằng chúng ta đã dùng cùng ký hiệu cho softmax
khi biểu thị một phép toán theo từng hàng trong :numref:`subsec_softmax_vectorization`.
Rất thường xuyên, các hàm kích hoạt mà chúng ta sử dụng không chỉ áp dụng theo từng hàng mà còn
theo từng phần tử. Điều đó có nghĩa là sau khi tính phần tuyến tính của tầng,
chúng ta có thể tính từng activation
mà không cần xem các giá trị của những đơn vị ẩn khác.

Để xây dựng các MLP tổng quát hơn, chúng ta có thể tiếp tục chồng
các tầng ẩn như vậy,
ví dụ: $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
và $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$,
tầng này trên tầng kia, tạo ra các mô hình ngày càng có khả năng biểu đạt cao hơn.

### Bộ xấp xỉ phổ quát

Chúng ta biết rằng não bộ có khả năng phân tích thống kê rất tinh vi. Do đó,
đáng để đặt câu hỏi: một mạng sâu có thể *mạnh đến mức nào*. Câu hỏi này
đã được trả lời nhiều lần, chẳng hạn trong :citet:`Cybenko.1989` trong bối cảnh
MLP, và trong :citet:`micchelli1984interpolation` trong bối cảnh tái tạo các không gian
Hilbert nhân tái sinh theo cách có thể xem như các mạng hàm cơ sở xuyên tâm (RBF) với một tầng ẩn duy nhất.
Các kết quả này (và các kết quả liên quan) cho thấy rằng ngay cả với một mạng chỉ có một tầng ẩn,
nếu có đủ node (có thể là cực kỳ nhiều),
và bộ trọng số phù hợp,
chúng ta có thể mô hình hóa bất kỳ hàm nào.
Tuy vậy, thực sự học được hàm đó mới là phần khó.
Bạn có thể xem mạng nơ-ron của mình
hơi giống ngôn ngữ lập trình C.
Ngôn ngữ này, như bất kỳ ngôn ngữ hiện đại nào khác,
có thể biểu đạt bất kỳ chương trình tính toán được nào.
Nhưng thực sự nghĩ ra một chương trình
đáp ứng các yêu cầu của bạn mới là phần khó.

Hơn nữa, chỉ vì một mạng có một tầng ẩn
*có thể* học bất kỳ hàm nào
không có nghĩa là bạn nên cố gắng
giải quyết mọi vấn đề của mình
bằng một mạng như vậy. Thực tế, trong trường hợp này, các phương pháp kernel
hiệu quả hơn rất nhiều, vì chúng có thể giải quyết bài toán
*một cách chính xác* ngay cả trong các không gian vô hạn chiều :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`.
Thực tế, chúng ta có thể xấp xỉ nhiều hàm
gọn hơn nhiều bằng cách dùng các mạng sâu hơn (thay vì rộng hơn) :cite:`Simonyan.Zisserman.2014`.
Chúng ta sẽ đề cập đến các lập luận chặt chẽ hơn trong các chương sau.


## Hàm kích hoạt
:label:`subsec_activation-functions`

Các hàm kích hoạt quyết định liệu một nơ-ron có nên được kích hoạt hay không bằng cách
tính tổng có trọng số và sau đó cộng thêm bias vào đó.
Chúng là các toán tử khả vi biến đổi tín hiệu đầu vào thành đầu ra,
trong đó phần lớn bổ sung tính phi tuyến.
Vì các hàm kích hoạt là nền tảng của deep learning,
(**hãy cùng khảo sát ngắn gọn một số hàm phổ biến**).

### Hàm ReLU

Lựa chọn phổ biến nhất,
nhờ cả sự đơn giản trong triển khai lẫn
hiệu năng tốt trên nhiều tác vụ dự đoán,
là *đơn vị tuyến tính chỉnh lưu* (*ReLU*) :cite:`Nair.Hinton.2010`.
[**ReLU cung cấp một phép biến đổi phi tuyến rất đơn giản**].
Với một phần tử $x$, hàm này được định nghĩa
là giá trị lớn nhất giữa phần tử đó và $0$:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Một cách trực quan, hàm ReLU chỉ giữ lại các
phần tử dương và loại bỏ toàn bộ phần tử âm
bằng cách đặt các activation tương ứng thành 0.
Để có thêm trực giác, chúng ta có thể vẽ đồ thị hàm này.
Như bạn có thể thấy, hàm kích hoạt này tuyến tính từng đoạn.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

Khi đầu vào âm,
đạo hàm của hàm ReLU bằng 0,
và khi đầu vào dương,
đạo hàm của hàm ReLU bằng 1.
Lưu ý rằng hàm ReLU không khả vi
khi đầu vào có giá trị chính xác bằng 0.
Trong các trường hợp này, chúng ta mặc định dùng đạo hàm
phía bên trái và nói rằng đạo hàm bằng 0 khi đầu vào bằng 0.
Chúng ta có thể chấp nhận điều này vì
đầu vào có thể thực tế không bao giờ bằng 0 (các nhà toán học sẽ
nói rằng nó không khả vi trên một tập có độ đo bằng không).
Có một câu ngạn ngữ cũ rằng nếu các điều kiện biên tinh vi là quan trọng,
có lẽ chúng ta đang làm toán học (*thực sự*), chứ không phải kỹ thuật.
Quan niệm thông thường đó có thể áp dụng ở đây, hoặc ít nhất là vì
chúng ta không thực hiện tối ưu hóa có ràng buộc :cite:`Mangasarian.1965,Rockafellar.1970`.
Chúng ta vẽ đồ thị đạo hàm của hàm ReLU bên dưới.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

Lý do sử dụng ReLU là vì
các đạo hàm của nó có tính chất đặc biệt tốt:
hoặc chúng triệt tiêu, hoặc chỉ đơn giản truyền đối số đi.
Điều này giúp quá trình tối ưu hóa ổn định hơn
và giảm nhẹ vấn đề đã được ghi nhận rõ ràng
về gradient biến mất, vốn từng gây khó khăn cho
các phiên bản mạng nơ-ron trước đây (sẽ nói thêm về điều này sau).

Lưu ý rằng có nhiều biến thể của hàm ReLU,
bao gồm hàm *ReLU có tham số* (*pReLU*) :cite:`He.Zhang.Ren.ea.2015`.
Biến thể này thêm một hạng tử tuyến tính vào ReLU,
để một phần thông tin vẫn đi qua,
ngay cả khi đối số âm:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Hàm sigmoid

[**Hàm *sigmoid* biến đổi các đầu vào**]
có giá trị thuộc miền $\mathbb{R}$
(**thành các đầu ra nằm trong khoảng (0, 1).**)
Vì lý do đó, sigmoid
thường được gọi là một *hàm nén*:
nó nén mọi đầu vào trong khoảng (-inf, inf)
thành một giá trị nào đó trong khoảng (0, 1):

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

Trong những mạng nơ-ron đầu tiên, các nhà khoa học
quan tâm đến việc mô hình hóa các nơ-ron sinh học
hoặc *phát xung* hoặc *không phát xung*.
Do đó, những người tiên phong trong lĩnh vực này,
từ tận thời McCulloch và Pitts,
những người phát minh ra nơ-ron nhân tạo,
đã tập trung vào các đơn vị ngưỡng :cite:`McCulloch.Pitts.1943`.
Một activation ngưỡng nhận giá trị 0
khi đầu vào của nó thấp hơn một ngưỡng nào đó
và giá trị 1 khi đầu vào vượt quá ngưỡng đó.

Khi trọng tâm chuyển sang học dựa trên gradient,
hàm sigmoid là một lựa chọn tự nhiên
vì nó là một phép xấp xỉ trơn, khả vi
cho một đơn vị ngưỡng.
Sigmoid vẫn được sử dụng rộng rãi làm
hàm kích hoạt trên các đơn vị đầu ra
khi chúng ta muốn diễn giải đầu ra là xác suất
cho các bài toán phân loại nhị phân: bạn có thể xem sigmoid là một trường hợp đặc biệt của softmax.
Tuy nhiên, sigmoid phần lớn đã được thay thế
bởi ReLU đơn giản hơn và dễ huấn luyện hơn
trong hầu hết các ứng dụng ở tầng ẩn. Phần lớn nguyên nhân là do
sigmoid gây ra thách thức cho tối ưu hóa
:cite:`LeCun.Bottou.Orr.ea.1998` vì gradient của nó biến mất với các đối số dương *và* âm lớn.
Điều này có thể dẫn đến các vùng bằng phẳng khó thoát ra.
Tuy vậy, sigmoid vẫn quan trọng. Trong các chương sau (ví dụ, :numref:`sec_lstm`) về mạng nơ-ron hồi quy,
chúng ta sẽ mô tả các kiến trúc tận dụng các đơn vị sigmoid
để kiểm soát luồng thông tin theo thời gian.

Bên dưới, chúng ta vẽ đồ thị hàm sigmoid.
Lưu ý rằng khi đầu vào gần 0,
hàm sigmoid tiến gần đến
một phép biến đổi tuyến tính.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

Đạo hàm của hàm sigmoid được cho bởi phương trình sau:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


Đạo hàm của hàm sigmoid được vẽ bên dưới.
Lưu ý rằng khi đầu vào bằng 0,
đạo hàm của hàm sigmoid
đạt giá trị cực đại là 0.25.
Khi đầu vào lệch khỏi 0 theo bất kỳ hướng nào,
đạo hàm tiến về 0.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### Hàm tanh
:label:`subsec_tanh`

Tương tự hàm sigmoid, [**hàm tanh (tangent hyperbolic)
cũng nén các đầu vào của nó**],
biến đổi chúng thành các phần tử trong khoảng (**từ $-1$ đến $1$**):

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

Chúng ta vẽ đồ thị hàm tanh bên dưới. Lưu ý rằng khi đầu vào tiến gần 0, hàm tanh tiến gần đến một phép biến đổi tuyến tính. Mặc dù hình dạng của hàm tương tự hàm sigmoid, hàm tanh thể hiện tính đối xứng tâm qua gốc tọa độ :cite:`Kalman.Kwasny.1992`.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

Đạo hàm của hàm tanh là:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

Được vẽ bên dưới.
Khi đầu vào tiến gần 0,
đạo hàm của hàm tanh tiến đến giá trị cực đại là 1.
Và như chúng ta đã thấy với hàm sigmoid,
khi đầu vào dịch chuyển ra xa 0 theo bất kỳ hướng nào,
đạo hàm của hàm tanh tiến đến 0.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## Tóm tắt và Thảo luận

Giờ đây, chúng ta đã biết cách kết hợp các phi tuyến tính
để xây dựng các kiến trúc mạng nơ-ron nhiều lớp có khả năng biểu diễn phong phú.
Nhân tiện, kiến thức của bạn hiện tại
đã giúp bạn làm chủ một bộ công cụ tương tự
với một người thực hành vào khoảng năm 1990.
Theo một số khía cạnh, bạn có lợi thế
so với bất kỳ ai làm việc vào thời điểm đó,
bởi vì bạn có thể tận dụng các framework deep learning mã nguồn mở mạnh mẽ
để xây dựng mô hình nhanh chóng, chỉ bằng vài dòng mã.
Trước đây, việc huấn luyện các mạng này
yêu cầu các nhà nghiên cứu phải viết mã cho các layer và đạo hàm
một cách tường minh bằng C, Fortran, hoặc thậm chí Lisp (trong trường hợp của LeNet).

Một lợi ích khác là ReLU phù hợp với
tối ưu hóa hơn đáng kể so với hàm sigmoid hoặc tanh. Có thể lập luận
rằng đây là một trong những đổi mới then chốt đã góp phần vào sự hồi sinh
của deep learning trong thập kỷ qua. Tuy nhiên, lưu ý rằng nghiên cứu về
các hàm kích hoạt vẫn chưa dừng lại.
Ví dụ,
hàm kích hoạt GELU (Gaussian error linear unit)
$x \Phi(x)$ của :citet:`Hendrycks.Gimpel.2016` ($\Phi(x)$
là hàm phân phối tích lũy Gaussian chuẩn)
và
hàm kích hoạt Swish
$\sigma(x) = x \operatorname{sigmoid}(\beta x)$ như được đề xuất trong :citet:`Ramachandran.Zoph.Le.2017` có thể mang lại độ chính xác tốt hơn
trong nhiều trường hợp.

## Bài tập

1. Chứng minh rằng việc thêm layer vào một mạng sâu *tuyến tính*, tức là một mạng không có
   tính phi tuyến $\sigma$, không bao giờ có thể tăng khả năng biểu diễn của mạng.
   Hãy đưa ra một ví dụ mà việc này thực sự làm giảm khả năng đó.
1. Tính đạo hàm của hàm kích hoạt pReLU.
1. Tính đạo hàm của hàm kích hoạt Swish $x \operatorname{sigmoid}(\beta x)$.
1. Chứng minh rằng một MLP chỉ sử dụng ReLU (hoặc pReLU) xây dựng một
   hàm tuyến tính từng đoạn liên tục.
1. Sigmoid và tanh rất tương tự nhau.
    1. Chứng minh rằng $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
    1. Chứng minh rằng các lớp hàm được tham số hóa bởi cả hai tính phi tuyến là giống hệt nhau. Gợi ý: các layer affine cũng có các hạng tử bias.
1. Giả sử rằng chúng ta có một tính phi tuyến áp dụng cho từng minibatch một, chẳng hạn như batch normalization :cite:`Ioffe.Szegedy.2015`. Bạn dự đoán điều này sẽ gây ra những loại vấn đề nào?
1. Đưa ra một ví dụ mà gradient triệt tiêu đối với hàm kích hoạt sigmoid.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/226)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17984)
:end_tab: