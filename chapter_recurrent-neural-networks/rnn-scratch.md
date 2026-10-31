# Triển khai Recurrent Neural Network từ đầu
:label:`sec_rnn-scratch`

Bây giờ chúng ta đã sẵn sàng triển khai một RNN từ đầu.
Cụ thể, chúng ta sẽ huấn luyện RNN này hoạt động
như một mô hình ngôn ngữ cấp ký tự
(xem :numref:`sec_rnn`)
và huấn luyện nó trên một corpus bao gồm
toàn bộ văn bản *The Time Machine* của H. G. Wells,
theo các bước xử lý dữ liệu
được trình bày trong :numref:`sec_text-sequence`.
Chúng ta bắt đầu bằng cách tải tập dữ liệu.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## Mô hình RNN

Chúng ta bắt đầu bằng cách định nghĩa một lớp
để triển khai mô hình RNN
(:numref:`subsec_rnn_w_hidden_states`).
Lưu ý rằng số lượng đơn vị ẩn `num_hiddens`
là một hyperparameter có thể điều chỉnh.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**Phương thức `forward` bên dưới định nghĩa cách tính
đầu ra và trạng thái ẩn tại bất kỳ bước thời gian nào,
dựa trên đầu vào hiện tại và trạng thái của mô hình
ở bước thời gian trước đó.**]
Lưu ý rằng mô hình RNN lặp qua
chiều ngoài cùng của `inputs`,
cập nhật trạng thái ẩn
mỗi lần một bước thời gian.
Mô hình ở đây sử dụng hàm kích hoạt $\tanh$ (:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

Chúng ta có thể đưa một minibatch các chuỗi đầu vào vào mô hình RNN như sau.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

Hãy kiểm tra xem mô hình RNN
có tạo ra kết quả với hình dạng chính xác hay không
để đảm bảo rằng số chiều
của trạng thái ẩn không thay đổi.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## Mô hình ngôn ngữ dựa trên RNN

Lớp `RNNLMScratch` sau đây định nghĩa
một mô hình ngôn ngữ dựa trên RNN,
trong đó chúng ta truyền RNN của mình
thông qua đối số `rnn`
của phương thức `__init__`.
Khi huấn luyện mô hình ngôn ngữ,
đầu vào và đầu ra
đến từ cùng một vocabulary.
Vì vậy, chúng có cùng số chiều,
bằng với kích thước vocabulary.
Lưu ý rằng chúng ta sử dụng perplexity để đánh giá mô hình.
Như đã thảo luận trong :numref:`subsec_perplexity`, điều này đảm bảo
các chuỗi có độ dài khác nhau có thể so sánh được.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**One-Hot Encoding**]

Hãy nhớ rằng mỗi token được biểu diễn
bằng một chỉ số số học cho biết
vị trí trong vocabulary của
từ/ký tự/mảnh từ tương ứng.
Bạn có thể bị cám dỗ xây dựng một neural network
với một nút đầu vào duy nhất (tại mỗi bước thời gian),
trong đó chỉ số có thể được đưa vào dưới dạng giá trị vô hướng.
Điều này hiệu quả khi chúng ta xử lý các đầu vào số
như giá hoặc nhiệt độ, nơi hai giá trị
đủ gần nhau
nên được xử lý tương tự nhau.
Nhưng điều này không thực sự hợp lý.
Các từ thứ $45^{\textrm{th}}$ và $46^{\textrm{th}}$
trong vocabulary của chúng ta lần lượt là "their" và "said",
mà ý nghĩa của chúng hoàn toàn không tương tự nhau.

Khi xử lý dữ liệu phân loại như vậy,
chiến lược phổ biến nhất là biểu diễn
mỗi mục bằng một *one-hot encoding*
(nhắc lại từ :numref:`subsec_classification-problem`).
One-hot encoding là một vector có độ dài
được xác định bởi kích thước vocabulary $N$,
trong đó tất cả phần tử đều được đặt thành $0$,
ngoại trừ phần tử tương ứng
với token của chúng ta, được đặt thành $1$.
Ví dụ, nếu vocabulary có năm phần tử,
thì các vector one-hot tương ứng
với chỉ số 0 và 2 sẽ như sau.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**Các minibatch mà chúng ta lấy mẫu tại mỗi iteration
sẽ có hình dạng (batch size, number of time steps).
Sau khi biểu diễn mỗi đầu vào dưới dạng vector one-hot,
chúng ta có thể xem mỗi minibatch như một tensor ba chiều,
trong đó độ dài theo trục thứ ba
được xác định bởi kích thước vocabulary (`len(vocab)`).**)
Chúng ta thường chuyển vị đầu vào để nhận được đầu ra
có hình dạng (number of time steps, batch size, vocabulary size).
Điều này sẽ cho phép chúng ta lặp thuận tiện hơn qua chiều ngoài cùng
để cập nhật trạng thái ẩn của một minibatch,
mỗi lần một bước thời gian
(ví dụ, trong phương thức `forward` ở trên).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### Biến đổi đầu ra RNN

Mô hình ngôn ngữ sử dụng một output layer fully connected
để biến đổi đầu ra RNN thành các dự đoán token tại mỗi bước thời gian.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Hãy [**kiểm tra xem phép tính forward
có tạo ra đầu ra với hình dạng chính xác hay không.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradient Clipping**]

Mặc dù bạn đã quen với việc xem neural network
là "sâu" theo nghĩa có nhiều layer
ngăn cách đầu vào và đầu ra
ngay cả trong một bước thời gian duy nhất,
độ dài của chuỗi lại đưa ra
một khái niệm mới về độ sâu.
Ngoài việc truyền qua network
theo hướng từ đầu vào đến đầu ra,
đầu vào ở bước thời gian đầu tiên
phải đi qua một chuỗi gồm $T$ layer
dọc theo các bước thời gian để
ảnh hưởng đến đầu ra của mô hình
tại bước thời gian cuối cùng.
Xét theo chiều ngược lại, trong mỗi iteration,
chúng ta backpropagate gradient theo thời gian,
dẫn đến một chuỗi tích ma trận
có độ dài $\mathcal{O}(T)$.
Như đã đề cập trong :numref:`sec_numerical_stability`,
điều này có thể gây ra bất ổn định số,
khiến gradient bùng nổ hoặc biến mất,
tùy thuộc vào đặc tính của các ma trận trọng số.

Xử lý gradient biến mất và bùng nổ
là một vấn đề nền tảng khi thiết kế RNN
và đã truyền cảm hứng cho một số tiến bộ lớn nhất
trong các kiến trúc neural network hiện đại.
Trong chương tiếp theo, chúng ta sẽ nói về
các kiến trúc chuyên biệt được thiết kế
với hy vọng giảm nhẹ vấn đề gradient biến mất.
Tuy nhiên, ngay cả các RNN hiện đại cũng thường gặp
gradient bùng nổ.
Một giải pháp không thanh nhã nhưng phổ biến
là đơn giản clip gradient,
buộc các gradient "đã clip" thu được
có giá trị nhỏ hơn.

Nói chung, khi tối ưu một objective
bằng gradient descent, chúng ta lặp lại việc cập nhật
parameter cần quan tâm, chẳng hạn một vector $\mathbf{x}$,
bằng cách đẩy nó theo hướng của
gradient âm $\mathbf{g}$
(trong stochastic gradient descent,
chúng ta tính gradient này
trên một minibatch được lấy mẫu ngẫu nhiên).
Ví dụ, với learning rate $\eta > 0$,
mỗi lần cập nhật có dạng
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$.
Giả sử thêm rằng hàm objective $f$
đủ trơn.
Về mặt hình thức, ta nói rằng objective
là *Lipschitz continuous* với hằng số $L$,
nghĩa là với mọi $\mathbf{x}$ và $\mathbf{y}$, ta có

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Như bạn có thể thấy, khi chúng ta cập nhật vector parameter bằng cách trừ $\eta \mathbf{g}$,
sự thay đổi trong giá trị của objective
phụ thuộc vào learning rate,
chuẩn của gradient và $L$ như sau:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

Nói cách khác, objective không thể
thay đổi nhiều hơn $L \eta \|\mathbf{g}\|$.
Việc có giá trị nhỏ cho cận trên này
có thể được xem là tốt hoặc xấu.
Về mặt bất lợi, chúng ta đang giới hạn tốc độ
mà tại đó có thể giảm giá trị của objective.
Về mặt tích cực, điều này giới hạn mức độ
chúng ta có thể sai lệch trong một bước gradient.

Khi nói gradient bùng nổ,
chúng ta muốn nói rằng $\|\mathbf{g}\|$
trở nên quá lớn.
Trong trường hợp xấu nhất, chúng ta có thể gây ra quá nhiều
thiệt hại chỉ trong một bước gradient đến mức
có thể xóa bỏ toàn bộ tiến bộ đạt được
trong hàng nghìn iteration huấn luyện.
Khi gradient có thể lớn đến vậy,
việc huấn luyện neural network thường phân kỳ,
không thể giảm giá trị của objective.
Trong các trường hợp khác, việc huấn luyện cuối cùng hội tụ
nhưng không ổn định do các đột biến lớn trong loss.

Một cách để giới hạn độ lớn của $L \eta \|\mathbf{g}\|$
là giảm learning rate $\eta$ xuống các giá trị rất nhỏ.
Cách này có ưu điểm là không làm lệch các cập nhật.
Nhưng nếu chúng ta chỉ *hiếm khi* nhận được gradient lớn thì sao?
Biện pháp quyết liệt này làm chậm tiến độ ở mọi bước,
chỉ để xử lý các sự kiện gradient bùng nổ hiếm gặp.
Một phương án thay thế phổ biến là áp dụng heuristic *gradient clipping*
chiếu các gradient $\mathbf{g}$ lên một quả cầu
có bán kính cho trước $\theta$ như sau:

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

Điều này đảm bảo chuẩn gradient không bao giờ vượt quá $\theta$
và gradient được cập nhật hoàn toàn cùng hướng
với hướng ban đầu của $\mathbf{g}$.
Nó cũng có tác dụng phụ mong muốn
là giới hạn mức ảnh hưởng mà bất kỳ minibatch nào
(và trong đó là bất kỳ sample nào)
có thể tác động lên vector parameter.
Điều này mang lại một mức độ robustness nhất định cho mô hình.
Nói rõ hơn, đây là một hack.
Gradient clipping có nghĩa là chúng ta không phải lúc nào cũng
đi theo gradient thực và rất khó
phân tích về các tác dụng phụ có thể có.
Tuy nhiên, đây là một hack rất hữu ích,
và được áp dụng rộng rãi trong các triển khai RNN
của hầu hết deep learning framework.

Bên dưới, chúng ta định nghĩa một phương thức để clip gradient,
được gọi bởi phương thức `fit_epoch` của
lớp `d2l.Trainer` (xem :numref:`sec_linear_scratch`).
Lưu ý rằng khi tính chuẩn gradient,
chúng ta nối tất cả parameter của mô hình,
coi chúng là một vector parameter khổng lồ duy nhất.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Huấn luyện

Sử dụng bộ dữ liệu *The Time Machine* (`data`),
chúng ta huấn luyện một mô hình ngôn ngữ ở cấp độ ký tự (`model`)
dựa trên RNN (`rnn`) được hiện thực từ đầu.
Lưu ý rằng trước hết chúng ta tính các gradient,
sau đó cắt ngưỡng chúng, và cuối cùng
cập nhật các tham số mô hình
bằng các gradient đã được cắt ngưỡng.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Giải mã

Khi một mô hình ngôn ngữ đã được học,
chúng ta không chỉ có thể dùng nó để dự đoán token tiếp theo
mà còn để tiếp tục dự đoán từng token kế tiếp,
coi token đã được dự đoán trước đó như thể
nó là token tiếp theo trong đầu vào.
Đôi khi chúng ta chỉ muốn sinh văn bản
như thể đang bắt đầu từ đầu
một tài liệu.
Tuy nhiên, thường sẽ hữu ích khi điều kiện hóa
mô hình ngôn ngữ dựa trên một tiền tố do người dùng cung cấp.
Ví dụ, nếu chúng ta đang phát triển một
tính năng autocomplete cho công cụ tìm kiếm
hoặc để hỗ trợ người dùng viết email,
chúng ta sẽ muốn đưa vào những gì họ
đã viết cho đến thời điểm đó (tiền tố),
rồi sinh ra một phần tiếp nối có khả năng xảy ra.


[**Phương thức `predict` sau đây
sinh một phần tiếp nối, từng ký tự một,
sau khi tiếp nhận một `prefix` do người dùng cung cấp**].
Khi lặp qua các ký tự trong `prefix`,
chúng ta tiếp tục truyền hidden state
sang time step tiếp theo
nhưng không sinh bất kỳ đầu ra nào.
Điều này được gọi là giai đoạn *warm-up*.
Sau khi tiếp nhận tiền tố, giờ đây chúng ta
đã sẵn sàng bắt đầu tạo ra các ký tự tiếp theo,
mỗi ký tự trong số đó sẽ được đưa trở lại mô hình
làm đầu vào ở time step kế tiếp.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Sau đây, chúng ta chỉ định tiền tố
và yêu cầu nó sinh thêm 20 ký tự.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

Mặc dù việc hiện thực mô hình RNN ở trên từ đầu mang tính hướng dẫn, nhưng không thuận tiện.
Trong phần tiếp theo, chúng ta sẽ thấy cách tận dụng các framework deep learning để nhanh chóng tạo RNN
bằng các kiến trúc chuẩn, đồng thời đạt được hiệu năng cao hơn
nhờ dựa vào các hàm thư viện được tối ưu hóa cao.


## Tóm tắt

Chúng ta có thể huấn luyện các mô hình ngôn ngữ dựa trên RNN để sinh văn bản tiếp nối tiền tố văn bản do người dùng cung cấp.
Một mô hình ngôn ngữ RNN đơn giản gồm mã hóa đầu vào, mô hình hóa RNN và sinh đầu ra.
Trong quá trình huấn luyện, gradient clipping có thể giảm nhẹ vấn đề exploding gradients nhưng không giải quyết vấn đề vanishing gradients. Trong thí nghiệm, chúng ta đã hiện thực một mô hình ngôn ngữ RNN đơn giản và huấn luyện nó với gradient clipping trên các chuỗi văn bản được token hóa ở cấp độ ký tự. Bằng cách điều kiện hóa theo một tiền tố, chúng ta có thể dùng mô hình ngôn ngữ để sinh các phần tiếp nối có khả năng xảy ra, điều này hữu ích trong nhiều ứng dụng, ví dụ như các tính năng autocomplete.


## Bài tập

1. Mô hình ngôn ngữ đã hiện thực có dự đoán token tiếp theo dựa trên tất cả các token trong quá khứ, tính đến token đầu tiên trong *The Time Machine*, không?
1. Hyperparameter nào kiểm soát độ dài lịch sử được dùng cho dự đoán?
1. Chứng minh rằng one-hot encoding tương đương với việc chọn một embedding khác nhau cho mỗi đối tượng.
1. Điều chỉnh các hyperparameter (ví dụ: số epoch, số hidden unit, số time step trong một minibatch và learning rate) để cải thiện perplexity. Với kiến trúc đơn giản này, bạn có thể giảm xuống mức thấp đến đâu?
1. Thay one-hot encoding bằng các embedding có thể học được. Điều này có dẫn đến hiệu năng tốt hơn không?
1. Tiến hành một thí nghiệm để xác định mức độ hiệu quả của mô hình ngôn ngữ này
   khi được huấn luyện trên *The Time Machine* đối với các sách khác của H. G. Wells,
   ví dụ, *The War of the Worlds*.
1. Tiến hành một thí nghiệm khác để đánh giá perplexity của mô hình này
   trên các sách được viết bởi những tác giả khác.
1. Sửa đổi phương thức dự đoán để sử dụng sampling
   thay vì chọn ký tự tiếp theo có xác suất cao nhất.
    * Điều gì xảy ra?
    * Thiên lệch mô hình về phía các đầu ra có khả năng xảy ra cao hơn, ví dụ,
    bằng cách lấy mẫu từ $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ với $\alpha > 1$.
1. Chạy mã trong phần này mà không gradient clipping. Điều gì xảy ra?
1. Thay activation function được sử dụng trong phần này bằng ReLU
   và lặp lại các thí nghiệm trong phần này. Chúng ta vẫn cần gradient clipping không? Tại sao?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18014)
:end_tab: