```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Triển khai Hồi quy tuyến tính từ đầu
:label:`sec_linear_scratch`

Bây giờ chúng ta đã sẵn sàng tìm hiểu từng bước một triển khai hồi quy tuyến tính hoàn chỉnh và có thể hoạt động. Trong phần này, (**chúng ta sẽ triển khai toàn bộ phương thức từ đầu, bao gồm (i) mô hình; (ii) hàm mất mát; (iii) một optimizer stochastic gradient descent theo minibatch; và (iv) hàm huấn luyện kết nối tất cả các thành phần này với nhau.**)
Cuối cùng, chúng ta sẽ chạy bộ tạo dữ liệu tổng hợp từ :numref:`sec_synthetic-regression-data` và áp dụng mô hình của mình trên tập dữ liệu thu được. Mặc dù các framework deep learning hiện đại có thể tự động hóa gần như toàn bộ công việc này, việc tự triển khai từ đầu là cách duy nhất để chắc chắn rằng bạn thực sự hiểu mình đang làm gì. Hơn nữa, khi cần tùy chỉnh mô hình, chẳng hạn như định nghĩa các layer hoặc hàm mất mát của riêng mình, việc hiểu cách mọi thứ hoạt động bên trong sẽ rất hữu ích. Trong phần này, chúng ta sẽ chỉ dựa vào tensor và automatic differentiation. Về sau, chúng ta sẽ giới thiệu một cách triển khai ngắn gọn hơn, tận dụng các tính năng phong phú của framework deep learning nhưng vẫn giữ nguyên cấu trúc được trình bày dưới đây.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Định nghĩa mô hình

[**Trước khi có thể bắt đầu tối ưu các tham số của mô hình**] bằng minibatch SGD, (**trước hết chúng ta cần có một số tham số.**)
Trong phần sau, chúng ta khởi tạo các trọng số bằng cách lấy mẫu ngẫu nhiên từ phân phối chuẩn có trung bình bằng 0 và độ lệch chuẩn bằng 0.01. Con số kỳ diệu 0.01 thường hoạt động tốt trong thực tế, nhưng bạn có thể chỉ định một giá trị khác thông qua đối số `sigma`. Ngoài ra, chúng ta đặt bias bằng 0. Lưu ý rằng theo thiết kế hướng đối tượng, chúng ta thêm đoạn mã vào phương thức `__init__` của một lớp con của `d2l.Module` (được giới thiệu trong :numref:`subsec_oo-design-models`).

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

Tiếp theo, chúng ta phải [**định nghĩa mô hình, liên hệ đầu vào và các tham số của mô hình với đầu ra của nó.**]
Sử dụng cùng ký hiệu như trong :eqref:`eq_linreg-y-vec` cho mô hình tuyến tính, chúng ta chỉ cần lấy tích ma trận--vector của các đặc trưng đầu vào $\mathbf{X}$ và các trọng số mô hình $\mathbf{w}$, rồi cộng offset $b$ vào từng mẫu. Tích $\mathbf{Xw}$ là một vector còn $b$ là một scalar. Nhờ cơ chế broadcasting (xem :numref:`subsec_broadcasting`), khi cộng một vector với một scalar, scalar sẽ được cộng vào từng thành phần của vector. Phương thức `forward` thu được được đăng ký trong lớp `LinearRegressionScratch` thông qua `add_to_class` (được giới thiệu trong :numref:`oo-design-utilities`).

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Định nghĩa hàm mất mát

Vì [**việc cập nhật mô hình đòi hỏi phải lấy gradient của hàm mất mát,**] nên (**trước tiên chúng ta cần định nghĩa hàm mất mát.**)
Ở đây, chúng ta sử dụng hàm mất mát bình phương trong :eqref:`eq_mse`. Trong phần triển khai, chúng ta cần chuyển đổi giá trị thực `y` sang shape của giá trị dự đoán `y_hat`. Kết quả được phương thức sau trả về cũng sẽ có cùng shape với `y_hat`. Chúng ta cũng trả về giá trị mất mát trung bình trên tất cả các mẫu trong minibatch.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Định nghĩa thuật toán tối ưu hóa

Như đã thảo luận trong :numref:`sec_linear_regression`, hồi quy tuyến tính có một nghiệm dạng đóng. Tuy nhiên, mục tiêu của chúng ta ở đây là minh họa cách huấn luyện các neural network tổng quát hơn, và điều đó đòi hỏi chúng ta phải hướng dẫn bạn cách sử dụng minibatch SGD. Vì vậy, chúng ta sẽ tận dụng cơ hội này để giới thiệu ví dụ SGD hoạt động đầu tiên của bạn. Ở mỗi bước, sử dụng một minibatch được lấy ngẫu nhiên từ tập dữ liệu, chúng ta ước lượng gradient của hàm mất mát theo các tham số. Tiếp theo, chúng ta cập nhật các tham số theo hướng có thể làm giảm mất mát.

Đoạn mã sau áp dụng phép cập nhật với một tập tham số và learning rate `lr`. Vì hàm mất mát của chúng ta được tính dưới dạng trung bình trên minibatch, chúng ta không cần điều chỉnh learning rate theo kích thước batch. Trong các chương sau, chúng ta sẽ tìm hiểu cách điều chỉnh learning rate cho các minibatch rất lớn, như thường gặp trong quá trình học phân tán quy mô lớn. Hiện tại, chúng ta có thể bỏ qua sự phụ thuộc này.

:begin_tab:`mxnet`
Chúng ta định nghĩa lớp `SGD`, một lớp con của `d2l.HyperParameters` (được giới thiệu trong :numref:`oo-design-utilities`), để có API tương tự optimizer SGD tích hợp sẵn. Chúng ta cập nhật các tham số trong phương thức `step`. Phương thức này nhận một đối số `batch_size` mà chúng ta có thể bỏ qua.
:end_tab:

:begin_tab:`pytorch`
Chúng ta định nghĩa lớp `SGD`, một lớp con của `d2l.HyperParameters` (được giới thiệu trong :numref:`oo-design-utilities`), để có API tương tự optimizer SGD tích hợp sẵn. Chúng ta cập nhật các tham số trong phương thức `step`. Phương thức `zero_grad` đặt tất cả gradient về 0, và phải được chạy trước một bước backpropagation.
:end_tab:

:begin_tab:`tensorflow`
Chúng ta định nghĩa lớp `SGD`, một lớp con của `d2l.HyperParameters` (được giới thiệu trong :numref:`oo-design-utilities`), để có API tương tự optimizer SGD tích hợp sẵn. Chúng ta cập nhật các tham số trong phương thức `apply_gradients`. Phương thức này nhận một danh sách các cặp tham số và gradient.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

Tiếp theo, chúng ta định nghĩa phương thức `configure_optimizers`, phương thức này trả về một instance của lớp `SGD`.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Huấn luyện

Bây giờ, khi đã có đầy đủ các thành phần (tham số, hàm mất mát, mô hình và optimizer), chúng ta sẵn sàng [**triển khai vòng lặp huấn luyện chính.**]
Điều quan trọng là bạn phải hiểu đầy đủ đoạn mã này, vì bạn sẽ sử dụng các vòng lặp huấn luyện tương tự cho mọi mô hình deep learning khác được trình bày trong cuốn sách này. Trong mỗi *epoch*, chúng ta lặp qua toàn bộ tập dữ liệu huấn luyện, đi qua mỗi mẫu đúng một lần (với giả định rằng số lượng mẫu chia hết cho kích thước batch). Trong mỗi *iteration*, chúng ta lấy một minibatch gồm các mẫu huấn luyện và tính loss của nó thông qua phương thức `training_step` của mô hình. Sau đó, chúng ta tính gradient theo từng tham số. Cuối cùng, chúng ta gọi thuật toán tối ưu hóa để cập nhật các tham số của mô hình. Tóm lại, chúng ta sẽ thực hiện vòng lặp sau:

* Khởi tạo các tham số $(\mathbf{w}, b)$
* Lặp lại cho đến khi hoàn tất
    * Tính gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Cập nhật các tham số $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
Hãy nhớ rằng tập dữ liệu hồi quy tổng hợp mà chúng ta đã tạo trong :numref:``sec_synthetic-regression-data`` không cung cấp tập dữ liệu validation. Tuy nhiên, trong hầu hết trường hợp, chúng ta sẽ muốn có một tập dữ liệu validation để đo lường chất lượng mô hình. Ở đây, chúng ta chạy qua validation dataloader một lần trong mỗi epoch để đo hiệu năng của mô hình. Theo thiết kế hướng đối tượng, các phương thức `prepare_batch` và `fit_epoch` được đăng ký trong lớp `d2l.Trainer` (được giới thiệu trong :numref:`oo-design-training`).

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

Chúng ta gần như đã sẵn sàng để huấn luyện mô hình, nhưng trước tiên cần có dữ liệu huấn luyện. Ở đây, chúng ta sử dụng lớp `SyntheticRegressionData` và truyền vào một số tham số ground truth. Sau đó, chúng ta huấn luyện mô hình với learning rate `lr=0.03` và đặt `max_epochs=3`. Lưu ý rằng nhìn chung, cả số epoch và learning rate đều là hyperparameter. Nhìn chung, việc thiết lập hyperparameter khá khó khăn, và chúng ta thường muốn sử dụng phép chia ba chiều: một tập cho huấn luyện, tập thứ hai để lựa chọn hyperparameter, và tập thứ ba dành riêng cho việc đánh giá cuối cùng. Hiện tại, chúng ta lược bỏ các chi tiết này và sẽ xem xét lại chúng sau.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Vì chính chúng ta đã tổng hợp tập dữ liệu, nên biết chính xác các tham số thực là gì. Do đó, chúng ta có thể [**đánh giá mức độ thành công của quá trình huấn luyện bằng cách so sánh các tham số thực với những tham số đã học được**] thông qua vòng lặp huấn luyện. Quả thực, chúng rất gần nhau.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

Chúng ta không nên mặc nhiên cho rằng có thể khôi phục chính xác các tham số ground truth. Nhìn chung, đối với các mô hình deep, nghiệm duy nhất cho các tham số không tồn tại; và ngay cả với các mô hình tuyến tính, việc khôi phục chính xác các tham số chỉ khả thi khi không có đặc trưng nào phụ thuộc tuyến tính vào các đặc trưng khác. Tuy nhiên, trong machine learning, chúng ta thường ít quan tâm đến việc khôi phục các tham số nền tảng thực sự hơn, mà quan tâm đến các tham số dẫn đến dự đoán có độ chính xác cao :cite:`Vapnik.1992`. May mắn là ngay cả với các bài toán tối ưu hóa khó, stochastic gradient descent thường có thể tìm được các nghiệm tốt đáng kinh ngạc, một phần nhờ thực tế rằng đối với các mạng deep, tồn tại nhiều cấu hình tham số khác nhau cùng dẫn đến dự đoán có độ chính xác cao.


## Tóm tắt

Trong phần này, chúng ta đã thực hiện một bước tiến đáng kể  
trong việc thiết kế các hệ thống deep learning  
bằng cách triển khai một mô hình mạng neuron  
và vòng lặp huấn luyện hoàn chỉnh.  
Trong quá trình này, chúng ta đã xây dựng một data loader,  
một mô hình, một hàm loss, một quy trình tối ưu hóa  
cùng một công cụ trực quan hóa và giám sát.  
Chúng ta thực hiện điều này bằng cách kết hợp các đối tượng Python  
chứa tất cả thành phần liên quan đến việc huấn luyện một mô hình.  
Mặc dù đây vẫn chưa phải là một triển khai ở cấp độ chuyên nghiệp,  
nó hoàn toàn hoạt động tốt và mã như vậy  
đã có thể giúp bạn giải quyết nhanh các bài toán nhỏ.  
Trong các phần tiếp theo, chúng ta sẽ tìm hiểu cách thực hiện điều này  
vừa *cô đọng hơn* (tránh mã boilerplate)  
vừa *hiệu quả hơn* (tận dụng GPU ở mức tối đa).



## Bài tập

1. Điều gì sẽ xảy ra nếu chúng ta khởi tạo các trọng số bằng không? Thuật toán có còn hoạt động không? Điều gì xảy ra nếu chúng ta
   khởi tạo các tham số với phương sai $1000$ thay vì $0.01$?
1. Giả sử bạn là [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) đang cố gắng xây dựng
   một mô hình về điện trở liên hệ điện áp và dòng điện. Bạn có thể sử dụng automatic
   differentiation để học các tham số của mô hình không?
1. Bạn có thể sử dụng [Định luật Planck](https://en.wikipedia.org/wiki/Planck%27s_law) để xác định nhiệt độ của một vật thể
   bằng cách sử dụng mật độ năng lượng phổ không? Để tham khảo, mật độ phổ $B$ của bức xạ phát ra từ một vật đen là
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Trong đó
   $\lambda$ là bước sóng, $T$ là nhiệt độ, $c$ là tốc độ ánh sáng, $h$ là hằng số Planck và $k$ là
   hằng số Boltzmann. Bạn đo năng lượng ở các bước sóng $\lambda$ khác nhau và giờ cần khớp
   đường cong mật độ phổ với định luật Planck.
1. Bạn có thể gặp những vấn đề gì nếu muốn tính các đạo hàm bậc hai của loss? Bạn sẽ
   khắc phục chúng như thế nào?
1. Tại sao cần có phương thức `reshape` trong hàm `loss`?
1. Hãy thử nghiệm với các learning rate khác nhau để tìm hiểu tốc độ giảm của giá trị hàm loss. Bạn có thể giảm
   lỗi bằng cách tăng số epoch huấn luyện không?
1. Nếu số lượng mẫu không chia hết cho batch size thì điều gì xảy ra với `data_iter` ở cuối một epoch?
1. Hãy thử triển khai một hàm loss khác, chẳng hạn như absolute value loss `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Kiểm tra điều gì xảy ra với dữ liệu thông thường.
    1. Kiểm tra xem hành vi có khác biệt không nếu bạn chủ động làm nhiễu một số phần tử, chẳng hạn như $y_5 = 10000$, trong $\mathbf{y}$.
    1. Bạn có nghĩ ra một giải pháp ít tốn kém để kết hợp những ưu điểm của squared loss và absolute value loss không?
       Gợi ý: làm thế nào để tránh các giá trị gradient thực sự lớn?
1. Tại sao chúng ta cần xáo trộn lại dataset? Bạn có thể thiết kế một trường hợp trong đó một dataset được tạo ra một cách có chủ ý sẽ làm hỏng thuật toán tối ưu hóa nếu không xáo trộn không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17976)
:end_tab: