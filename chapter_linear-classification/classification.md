```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mô hình Phân loại Cơ sở
:label:`sec_classification`

Bạn có thể đã nhận thấy rằng các phần triển khai từ đầu và phần triển khai ngắn gọn sử dụng chức năng của framework khá giống nhau trong trường hợp hồi quy. Điều này cũng đúng với phân loại. Vì nhiều mô hình trong cuốn sách này xử lý bài toán phân loại, nên việc bổ sung các chức năng để hỗ trợ riêng cho thiết lập này là đáng giá. Phần này cung cấp một lớp cơ sở cho các mô hình phân loại nhằm đơn giản hóa mã nguồn sau này.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Lớp `Classifier`

:begin_tab:`pytorch, mxnet, tensorflow`
Chúng ta định nghĩa lớp `Classifier` bên dưới. Trong `validation_step`, chúng ta báo cáo cả giá trị loss và độ chính xác phân loại trên một batch xác thực. Chúng ta vẽ một bản cập nhật cho mỗi `num_val_batches` batch. Điều này có lợi ích là tạo ra loss và độ chính xác trung bình trên toàn bộ dữ liệu xác thực. Các con số trung bình này không hoàn toàn chính xác nếu batch cuối chứa ít ví dụ hơn, nhưng chúng ta bỏ qua khác biệt nhỏ này để giữ cho mã đơn giản.
:end_tab:


:begin_tab:`jax`
Chúng ta định nghĩa lớp `Classifier` bên dưới. Trong `validation_step`, chúng ta báo cáo cả giá trị loss và độ chính xác phân loại trên một batch xác thực. Chúng ta vẽ một bản cập nhật cho mỗi `num_val_batches` batch. Điều này có lợi ích là tạo ra loss và độ chính xác trung bình trên toàn bộ dữ liệu xác thực. Các con số trung bình này không hoàn toàn chính xác nếu batch cuối chứa ít ví dụ hơn, nhưng chúng ta bỏ qua khác biệt nhỏ này để giữ cho mã đơn giản.

Chúng ta cũng định nghĩa lại phương thức `training_step` cho JAX vì tất cả các mô hình sau này kế thừa từ `Classifier` sẽ có một loss trả về dữ liệu phụ trợ. Dữ liệu phụ trợ này có thể được dùng cho các mô hình có batch normalization (sẽ được giải thích trong :numref:`sec_batch_norm`), trong khi ở mọi trường hợp khác, chúng ta cũng sẽ để loss trả về một placeholder (dictionary rỗng) để biểu diễn dữ liệu phụ trợ.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

Theo mặc định, chúng ta sử dụng optimizer stochastic gradient descent, hoạt động trên các minibatch, giống như đã làm trong ngữ cảnh hồi quy tuyến tính.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Độ chính xác

Với phân phối xác suất dự đoán `y_hat`, chúng ta thường chọn lớp có xác suất dự đoán cao nhất bất cứ khi nào cần đưa ra một dự đoán cứng.
Thật vậy, nhiều ứng dụng yêu cầu chúng ta phải đưa ra một lựa chọn.
Ví dụ, Gmail phải phân loại một email vào "Primary", "Social", "Updates", "Forums", hoặc "Spam".
Nó có thể ước lượng xác suất nội bộ,
nhưng cuối cùng vẫn phải chọn một trong các lớp.

Khi dự đoán khớp với lớp nhãn `y`, chúng là đúng.
Độ chính xác phân loại là tỷ lệ của tất cả dự đoán đúng.
Mặc dù việc tối ưu trực tiếp độ chính xác có thể khó khăn (nó không khả vi),
đây thường là thước đo hiệu năng mà chúng ta quan tâm nhất. Nó thường là đại lượng *then chốt*
trong các benchmark. Do đó, chúng ta gần như luôn báo cáo nó khi huấn luyện các classifier.

Độ chính xác được tính như sau.
Trước hết, nếu `y_hat` là một ma trận,
chúng ta giả định rằng chiều thứ hai lưu điểm dự đoán cho từng lớp.
Chúng ta dùng `argmax` để lấy lớp dự đoán bằng chỉ số của phần tử lớn nhất trong mỗi hàng.
Sau đó, chúng ta [**so sánh từng phần tử giữa lớp dự đoán và ground truth `y`.**]
Vì toán tử so sánh bằng `==` nhạy cảm với kiểu dữ liệu,
chúng ta chuyển kiểu dữ liệu của `y_hat` để khớp với kiểu dữ liệu của `y`.
Kết quả là một tensor chứa các phần tử 0 (false) và 1 (true).
Lấy tổng sẽ cho số lượng dự đoán đúng.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Tóm tắt

Phân loại là một bài toán đủ phổ biến để cần có các hàm tiện ích riêng. Điều có tầm quan trọng trung tâm trong phân loại là *độ chính xác* của classifier. Lưu ý rằng mặc dù chúng ta thường chủ yếu quan tâm đến độ chính xác, chúng ta huấn luyện classifier để tối ưu nhiều mục tiêu khác nhau vì các lý do thống kê và tính toán. Tuy nhiên, bất kể hàm loss nào được tối thiểu hóa trong quá trình huấn luyện, việc có một phương thức tiện ích để đánh giá thực nghiệm độ chính xác của classifier vẫn hữu ích.


## Bài tập

1. Ký hiệu $L_\textrm{v}$ là loss xác thực, và gọi $L_\textrm{v}^\textrm{q}$ là ước lượng nhanh và đơn giản của nó được tính bằng cách lấy trung bình hàm loss trong phần này. Cuối cùng, ký hiệu $l_\textrm{v}^\textrm{b}$ là loss trên minibatch cuối. Hãy biểu diễn $L_\textrm{v}$ theo $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, cùng kích thước mẫu và minibatch.
1. Chứng minh rằng ước lượng nhanh và đơn giản $L_\textrm{v}^\textrm{q}$ là không chệch. Nghĩa là, chứng minh rằng $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Vì sao bạn vẫn muốn sử dụng $L_\textrm{v}$ thay vào đó?
1. Với một loss phân loại đa lớp, ký hiệu $l(y,y')$ là mức phạt khi ước lượng $y'$ trong khi quan sát $y$, và cho xác suất $p(y \mid x)$, hãy xây dựng quy tắc để chọn $y'$ tối ưu. Gợi ý: biểu diễn loss kỳ vọng bằng $l$ và $p(y \mid x)$.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17981)
:end_tab: