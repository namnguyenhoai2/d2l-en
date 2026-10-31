# Trình biên dịch và trình thông dịch
:label:`sec_hybridize`

Cho đến nay, cuốn sách này tập trung vào lập trình mệnh lệnh (imperative programming), sử dụng các câu lệnh như `print`, `+` và `if` để thay đổi trạng thái của chương trình. Hãy xem ví dụ sau về một chương trình mệnh lệnh đơn giản.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python là một *ngôn ngữ thông dịch*. Khi đánh giá hàm `fancy_func` ở trên, nó thực hiện các thao tác tạo nên phần thân của hàm *theo thứ tự*. Nghĩa là, nó sẽ đánh giá `e = add(a, b)` và lưu kết quả vào biến `e`, từ đó thay đổi trạng thái của chương trình. Hai câu lệnh tiếp theo, `f = add(c, d)` và `g = add(e, f)`, cũng sẽ được thực thi tương tự, thực hiện phép cộng và lưu kết quả vào các biến. :numref:`fig_compute_graph` minh họa luồng dữ liệu.

![Luồng dữ liệu trong một chương trình mệnh lệnh.](../img/computegraph.svg)
:label:`fig_compute_graph`

Mặc dù lập trình mệnh lệnh thuận tiện, nhưng nó có thể không hiệu quả. Một mặt, ngay cả khi hàm `add` được gọi lặp lại trong `fancy_func`, Python vẫn thực thi riêng lẻ ba lần gọi hàm. Nếu các thao tác này được thực hiện trên GPU (hoặc thậm chí trên nhiều GPU), chi phí phát sinh từ trình thông dịch Python có thể trở nên quá lớn. Hơn nữa, Python sẽ cần lưu các giá trị của biến `e` và `f` cho đến khi tất cả câu lệnh trong `fancy_func` được thực thi. Điều này là do chúng ta không biết liệu các biến `e` và `f` có được những phần khác của chương trình sử dụng sau khi các câu lệnh `e = add(a, b)` và `f = add(c, d)` được thực thi hay không.

## Lập trình ký hiệu

Hãy xem xét một lựa chọn khác là *lập trình ký hiệu* (symbolic programming), trong đó phép tính thường chỉ được thực hiện sau khi toàn bộ quy trình đã được định nghĩa. Chiến lược này được nhiều framework deep learning sử dụng, bao gồm Theano và TensorFlow (framework sau này đã bổ sung các phần mở rộng mệnh lệnh). Chiến lược này thường bao gồm các bước sau:

1. Xác định các thao tác cần thực hiện.
1. Biên dịch các thao tác thành một chương trình có thể thực thi.
1. Cung cấp các đầu vào cần thiết và gọi chương trình đã biên dịch để thực thi.

Cách này cho phép tối ưu hóa đáng kể. Trước hết, trong nhiều trường hợp, chúng ta có thể bỏ qua trình thông dịch Python, qua đó loại bỏ một nút thắt hiệu năng có thể trở nên đáng kể khi sử dụng nhiều GPU nhanh kết hợp với một thread Python duy nhất trên CPU.

Thứ hai, một trình biên dịch có thể tối ưu hóa và viết lại đoạn mã trên thành `print((1 + 2) + (3 + 4))` hoặc thậm chí `print(10)`. Điều này khả thi vì trình biên dịch có thể xem toàn bộ mã trước khi chuyển nó thành các chỉ thị máy. Chẳng hạn, trình biên dịch có thể giải phóng bộ nhớ (hoặc không bao giờ cấp phát bộ nhớ) khi một biến không còn cần thiết. Hoặc nó có thể biến đổi hoàn toàn đoạn mã thành một đoạn mã tương đương.
Để hiểu rõ hơn, hãy xem mô phỏng sau đây về lập trình mệnh lệnh (dù sao thì đây vẫn là Python).

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

Sự khác biệt giữa lập trình mệnh lệnh (được thông dịch) và lập trình ký hiệu như sau:

* Lập trình mệnh lệnh dễ hơn. Khi sử dụng lập trình mệnh lệnh trong Python, phần lớn mã nguồn rõ ràng và dễ viết. Việc debug mã lập trình mệnh lệnh cũng dễ hơn. Đó là vì việc lấy và in tất cả giá trị biến trung gian liên quan, hoặc sử dụng các công cụ debug tích hợp sẵn của Python, dễ thực hiện hơn.
* Lập trình ký hiệu hiệu quả hơn và dễ chuyển đổi hơn. Lập trình ký hiệu giúp tối ưu hóa mã trong quá trình biên dịch dễ dàng hơn, đồng thời có khả năng chuyển chương trình sang một định dạng không phụ thuộc vào Python. Nhờ đó, chương trình có thể chạy trong môi trường không sử dụng Python, tránh các vấn đề hiệu năng tiềm ẩn liên quan đến trình thông dịch Python.


## Lập trình lai

Trong lịch sử, hầu hết framework deep learning đều lựa chọn giữa cách tiếp cận mệnh lệnh và cách tiếp cận ký hiệu. Ví dụ, Theano, TensorFlow (lấy cảm hứng từ Theano), Keras và CNTK biểu diễn các model bằng ký hiệu. Ngược lại, Chainer và PyTorch sử dụng cách tiếp cận mệnh lệnh. TensorFlow 2.0 đã bổ sung chế độ mệnh lệnh, và Keras cũng bổ sung chế độ này trong các phiên bản sau.

:begin_tab:`mxnet`
Khi thiết kế Gluon, các nhà phát triển đã cân nhắc liệu có thể kết hợp những lợi ích của cả hai mô hình lập trình hay không. Điều này dẫn đến một model lai, cho phép người dùng phát triển và debug bằng lập trình mệnh lệnh thuần túy, đồng thời có khả năng chuyển đổi hầu hết chương trình thành chương trình ký hiệu để chạy khi cần hiệu năng tính toán và khả năng deploy ở cấp độ sản phẩm.

Trên thực tế, điều này có nghĩa là chúng ta xây dựng model bằng class `HybridBlock` hoặc `HybridSequential`. Theo mặc định, cả hai class đều được thực thi theo cách giống như class `Block` hoặc `Sequential` khi lập trình mệnh lệnh.
Class `HybridSequential` là một subclass của `HybridBlock` (giống như `Sequential` là subclass của `Block`). Khi gọi hàm `hybridize`, Gluon biên dịch model sang dạng được sử dụng trong lập trình ký hiệu. Điều này cho phép tối ưu hóa các thành phần cần nhiều tính toán mà không phải hy sinh cách triển khai model. Chúng ta sẽ minh họa các lợi ích bên dưới, tập trung vào các model tuần tự và block.
:end_tab:

:begin_tab:`pytorch`
Như đã đề cập ở trên, PyTorch dựa trên lập trình mệnh lệnh và sử dụng computational graph động. Để tận dụng tính khả chuyển và hiệu quả của lập trình ký hiệu, các nhà phát triển đã cân nhắc liệu có thể kết hợp những lợi ích của cả hai mô hình lập trình hay không. Điều này dẫn đến torchscript, cho phép người dùng phát triển và debug bằng lập trình mệnh lệnh thuần túy, đồng thời có khả năng chuyển đổi hầu hết chương trình thành chương trình ký hiệu để chạy khi cần hiệu năng tính toán và khả năng deploy ở cấp độ sản phẩm.
:end_tab:

:begin_tab:`tensorflow`
Mô hình lập trình mệnh lệnh hiện là mặc định trong Tensorflow 2, một thay đổi đáng hoan nghênh đối với những người mới làm quen với ngôn ngữ này. Tuy nhiên, các kỹ thuật lập trình ký hiệu và computational graph tương ứng vẫn tồn tại trong TensorFlow và có thể được truy cập thông qua decorator `tf.function` dễ sử dụng. Điều này đã đưa mô hình lập trình mệnh lệnh vào TensorFlow, cho phép người dùng định nghĩa các hàm trực quan hơn, sau đó bọc chúng và tự động biên dịch thành computational graph bằng tính năng mà nhóm TensorFlow gọi là [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
:end_tab:

## Hybrid hóa class `Sequential`

Cách dễ nhất để hình dung cách hybridization hoạt động là xem xét các deep network có nhiều layer. Theo cách thông thường, trình thông dịch Python cần thực thi mã cho tất cả layer để tạo ra một chỉ thị, sau đó chuyển chỉ thị này đến CPU hoặc GPU. Với một thiết bị tính toán đơn lẻ (nhanh), điều này không gây ra vấn đề đáng kể. Mặt khác, nếu sử dụng một server 8-GPU hiện đại như instance AWS P3dn.24xlarge, Python sẽ khó giữ cho tất cả GPU luôn bận. Trình thông dịch Python đơn thread trở thành nút thắt ở đây. Hãy xem cách giải quyết vấn đề này đối với các phần đáng kể của mã bằng cách thay thế `Sequential` bằng `HybridSequential`. Trước tiên, chúng ta định nghĩa một MLP đơn giản.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
Bằng cách gọi hàm `hybridize`, chúng ta có thể biên dịch và tối ưu hóa phép tính trong MLP. Kết quả tính toán của model không thay đổi.
:end_tab:

:begin_tab:`pytorch`
Bằng cách chuyển đổi model bằng hàm `torch.jit.script`, chúng ta có thể biên dịch và tối ưu hóa phép tính trong MLP. Kết quả tính toán của model không thay đổi.
:end_tab:

:begin_tab:`tensorflow`
Trước đây, tất cả hàm được xây dựng trong TensorFlow đều được tạo dưới dạng computational graph và do đó mặc định được JIT compile. Tuy nhiên, với sự ra mắt của TensorFlow 2.X và EagerTensor, đây không còn là hành vi mặc định.
Chúng ta có thể bật lại chức năng này bằng tf.function. tf.function thường được sử dụng dưới dạng function decorator, nhưng cũng có thể được gọi trực tiếp như một hàm Python thông thường, như minh họa bên dưới. Kết quả tính toán của model không thay đổi.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Điều này dường như gần như quá tốt để là sự thật: chỉ cần chỉ định một block là `HybridSequential`, viết cùng đoạn mã như trước và gọi `hybridize`. Sau đó, network sẽ được tối ưu hóa (chúng ta sẽ benchmark hiệu năng bên dưới). Đáng tiếc là điều này không tự động hoạt động với mọi layer. Tuy vậy, một layer sẽ không được tối ưu hóa nếu nó kế thừa từ class `Block` thay vì class `HybridBlock`.
:end_tab:

:begin_tab:`pytorch`
Điều này dường như gần như quá tốt để là sự thật: viết cùng đoạn mã như trước và chỉ cần chuyển đổi model bằng `torch.jit.script`. Sau đó, network sẽ được tối ưu hóa (chúng ta sẽ benchmark hiệu năng bên dưới).
:end_tab:

:begin_tab:`tensorflow`
Điều này dường như gần như quá tốt để trở thành sự thật: viết cùng đoạn mã như trước và chỉ cần chuyển đổi mô hình bằng `tf.function`. Sau khi hoàn tất, mạng được xây dựng dưới dạng một computational graph trong biểu diễn trung gian MLIR của TensorFlow và được tối ưu hóa mạnh ở cấp compiler để thực thi nhanh (chúng ta sẽ đánh giá hiệu năng bên dưới).
Việc thêm tường minh cờ `jit_compile = True` vào lời gọi `tf.function()` sẽ bật chức năng XLA (Accelerated Linear Algebra) trong TensorFlow. Trong một số trường hợp, XLA có thể tối ưu thêm mã được biên dịch JIT. Chế độ thực thi graph được bật mà không cần định nghĩa tường minh này; tuy nhiên, XLA có thể làm cho một số phép toán đại số tuyến tính lớn (tương tự những phép toán thường thấy trong các ứng dụng deep learning) nhanh hơn đáng kể, đặc biệt trong môi trường GPU.
:end_tab:

### Tăng tốc bằng Hybridization

Để minh họa mức cải thiện hiệu năng đạt được nhờ compilation, chúng ta sẽ so sánh thời gian cần để đánh giá `net(x)` trước và sau hybridization. Trước tiên, hãy định nghĩa một class để đo thời gian này. Class này sẽ hữu ích trong suốt chương khi chúng ta tiến hành đo lường (và cải thiện) hiệu năng.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
Bây giờ chúng ta có thể gọi network hai lần, một lần có hybridization và một lần không có hybridization.
:end_tab:

:begin_tab:`pytorch`
Bây giờ chúng ta có thể gọi network hai lần, một lần có torchscript và một lần không có torchscript.
:end_tab:

:begin_tab:`tensorflow`
Bây giờ chúng ta có thể gọi network ba lần: một lần thực thi eager, một lần với graph-mode execution và một lần nữa bằng XLA được biên dịch JIT.
:end_tab:

```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
Như có thể thấy từ các kết quả ở trên, sau khi một instance `HybridSequential` gọi hàm `hybridize`, hiệu năng tính toán được cải thiện nhờ sử dụng symbolic programming.
:end_tab:

:begin_tab:`pytorch`
Như có thể thấy từ các kết quả ở trên, sau khi một instance `nn.Sequential` được script hóa bằng hàm `torch.jit.script`, hiệu năng tính toán được cải thiện nhờ sử dụng symbolic programming.
:end_tab:

:begin_tab:`tensorflow`
Như có thể thấy từ các kết quả ở trên, sau khi một instance `tf.keras.Sequential` được script hóa bằng hàm `tf.function`, hiệu năng tính toán được cải thiện nhờ sử dụng symbolic programming thông qua graph-mode execution trong tensorflow.
:end_tab:

### Serialization

:begin_tab:`mxnet`
Một trong những lợi ích của việc compile các model là chúng ta có thể serialize (lưu) model và các parameter của nó vào ổ đĩa. Điều này cho phép chúng ta lưu trữ một model theo cách độc lập với ngôn ngữ front-end được lựa chọn. Nhờ đó, chúng ta có thể deploy các model đã train lên những thiết bị khác và dễ dàng sử dụng các ngôn ngữ lập trình front-end khác. Đồng thời, code thường nhanh hơn so với những gì có thể đạt được bằng imperative programming. Hãy xem hàm `export` hoạt động.
:end_tab:

:begin_tab:`pytorch`
Một trong những lợi ích của việc compile các model là chúng ta có thể serialize (lưu) model và các parameter của nó vào ổ đĩa. Điều này cho phép chúng ta lưu trữ một model theo cách độc lập với ngôn ngữ front-end được lựa chọn. Nhờ đó, chúng ta có thể deploy các model đã train lên những thiết bị khác và dễ dàng sử dụng các ngôn ngữ lập trình front-end khác. Đồng thời, code thường nhanh hơn so với những gì có thể đạt được bằng imperative programming. Hãy xem hàm `save` hoạt động.
:end_tab:

:begin_tab:`tensorflow`
Một trong những lợi ích của việc compile các model là chúng ta có thể serialize (lưu) model và các parameter của nó vào ổ đĩa. Điều này cho phép chúng ta lưu trữ một model theo cách độc lập với ngôn ngữ front-end được lựa chọn. Nhờ đó, chúng ta có thể deploy các model đã train lên những thiết bị khác, dễ dàng sử dụng các ngôn ngữ lập trình front-end khác hoặc thực thi một model đã train trên server. Đồng thời, code thường nhanh hơn so với những gì có thể đạt được bằng imperative programming.
API cấp thấp cho phép chúng ta lưu trong tensorflow là `tf.saved_model`.
Hãy xem instance `saved_model` hoạt động.
:end_tab:

```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
Model được phân tách thành một file parameter (binary lớn) và một mô tả JSON về chương trình cần thiết để thực thi phép tính của model. Các file này có thể được đọc bởi những ngôn ngữ front-end khác được Python hoặc MXNet hỗ trợ, chẳng hạn như C++, R, Scala và Perl. Hãy xem vài dòng đầu tiên trong phần mô tả model.
:end_tab:

```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
Trước đó, chúng ta đã chứng minh rằng sau khi gọi hàm `hybridize`, model có thể đạt được hiệu năng tính toán và khả năng portability vượt trội. Tuy nhiên, cần lưu ý rằng hybridization có thể ảnh hưởng đến tính linh hoạt của model, đặc biệt là về control flow.

Ngoài ra, trái với instance `Block`, vốn cần sử dụng hàm `forward`, với một instance `HybridBlock`, chúng ta cần sử dụng hàm `hybrid_forward`.
:end_tab:

```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
Đoạn code trên triển khai một network đơn giản với 4 hidden unit và 2 output. Hàm `hybrid_forward` nhận thêm một argument là `F`. Điều này cần thiết vì tùy thuộc vào việc code đã được hybridize hay chưa, nó sẽ sử dụng một library hơi khác (`ndarray` hoặc `symbol`) để xử lý. Cả hai class đều thực hiện những chức năng rất tương tự và MXNet tự động xác định argument này. Để hiểu điều gì đang xảy ra, chúng ta in các argument trong lời gọi hàm.
:end_tab:

```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
Việc lặp lại phép tính forward sẽ cho cùng một output (chúng ta lược bỏ chi tiết). Bây giờ hãy xem điều gì xảy ra nếu chúng ta gọi hàm `hybridize`.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Thay vì sử dụng `ndarray`, giờ đây chúng ta sử dụng module `symbol` cho `F`. Hơn nữa, mặc dù input có kiểu `ndarray`, dữ liệu chạy qua network giờ đây được chuyển đổi thành kiểu `symbol` trong quá trình compilation. Việc lặp lại lời gọi hàm dẫn đến một kết quả đáng ngạc nhiên:
:end_tab:

```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
Điều này khá khác so với những gì chúng ta đã thấy trước đó. Tất cả câu lệnh print, như được định nghĩa trong `hybrid_forward`, đều bị lược bỏ. Thật vậy, sau hybridization, việc thực thi `net(x)` không còn liên quan đến Python interpreter nữa. Điều này có nghĩa là mọi đoạn Python code thừa đều bị lược bỏ (chẳng hạn như các câu lệnh print) để ưu tiên việc thực thi hợp lý hơn và hiệu năng tốt hơn. Thay vào đó, MXNet gọi trực tiếp C++ backend. Cũng lưu ý rằng một số hàm không được module `symbol` hỗ trợ (ví dụ: `asnumpy`) và các operation in-place như `a += b` và `a[:] = a + b` phải được viết lại thành `a = a + b`. Tuy vậy, compilation model vẫn đáng để thực hiện bất cứ khi nào tốc độ là yếu tố quan trọng. Lợi ích có thể dao động từ vài phần trăm nhỏ đến tốc độ nhanh hơn hơn hai lần, tùy thuộc vào độ phức tạp của model, tốc độ CPU, cũng như tốc độ và số lượng GPU.
:end_tab:

## Tóm tắt


* Imperative programming giúp dễ dàng thiết kế các model mới vì có thể viết code với control flow và sử dụng một lượng lớn hệ sinh thái phần mềm Python.
* Symbolic programming yêu cầu chúng ta chỉ định chương trình và compile chương trình trước khi thực thi. Lợi ích là hiệu năng được cải thiện.

:begin_tab:`mxnet` 
* MXNet có thể kết hợp các ưu điểm của cả hai phương pháp khi cần.
* Các model được xây dựng bằng các class `HybridSequential` và `HybridBlock` có thể chuyển đổi imperative program thành symbolic program bằng cách gọi hàm `hybridize`.
:end_tab:


## Bài tập


:begin_tab:`mxnet` 
1. Thêm `x.asnumpy()` vào dòng đầu tiên của hàm `hybrid_forward` thuộc class `HybridNet` trong phần này. Thực thi code và quan sát các lỗi gặp phải. Tại sao chúng xảy ra?
1. Điều gì xảy ra nếu chúng ta thêm control flow, tức là các câu lệnh Python `if` và `for` vào hàm `hybrid_forward`?
1. Xem lại các model mà bạn quan tâm trong những chương trước. Bạn có thể cải thiện hiệu năng tính toán của chúng bằng cách triển khai lại không?
:end_tab:

:begin_tab:`pytorch,tensorflow` 
1. Xem lại các model mà bạn quan tâm trong những chương trước. Bạn có thể cải thiện hiệu năng tính toán của chúng bằng cách triển khai lại không?
:end_tab:




:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/2492)
:end_tab: