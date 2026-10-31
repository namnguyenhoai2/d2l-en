```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Chuẩn hóa theo batch
:label:`sec_batch_norm`

Việc huấn luyện các mạng nơ-ron sâu rất khó.
Để chúng hội tụ trong một khoảng thời gian hợp lý có thể là một thách thức.
Trong phần này, chúng ta mô tả *chuẩn hóa theo batch* (batch normalization), một kỹ thuật phổ biến và hiệu quả
giúp tăng tốc quá trình hội tụ của các mạng sâu một cách nhất quán :cite:`Ioffe.Szegedy.2015`.
Cùng với các residual block---được trình bày sau trong :numref:`sec_resnet`---chuẩn hóa theo batch
đã giúp các chuyên gia có thể thường xuyên huấn luyện những mạng có hơn 100 lớp.
Một lợi ích phụ (tình cờ) của chuẩn hóa theo batch là khả năng regularization vốn có của nó.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Huấn luyện mạng sâu

Khi làm việc với dữ liệu, chúng ta thường tiền xử lý trước khi huấn luyện.
Các lựa chọn về tiền xử lý dữ liệu thường tạo ra khác biệt rất lớn trong kết quả cuối cùng.
Hãy nhớ lại việc chúng ta áp dụng MLP để dự đoán giá nhà (:numref:`sec_kaggle_house`).
Bước đầu tiên khi làm việc với dữ liệu thực
là chuẩn hóa các đặc trưng đầu vào để chúng có
trung bình bằng không $\boldsymbol{\mu} = 0$ và phương sai đơn vị $\boldsymbol{\Sigma} = \boldsymbol{1}$ trên nhiều quan sát :cite:`friedman1987exploratory`, thường điều chỉnh lại đại lượng sau để đường chéo bằng một, tức là $\Sigma_{ii} = 1$.
Một chiến lược khác là điều chỉnh các vector về độ dài đơn vị, có thể kèm trung bình bằng không *cho mỗi quan sát*.
Điều này có thể hoạt động tốt, chẳng hạn với dữ liệu cảm biến không gian. Các kỹ thuật tiền xử lý này cùng nhiều kỹ thuật khác
hữu ích trong việc giữ cho bài toán ước lượng được kiểm soát tốt.
Ví dụ, để xem tổng quan về lựa chọn và trích xuất đặc trưng, hãy xem bài viết của :citet:`guyon2008feature`.
Chuẩn hóa vector cũng có tác dụng phụ hữu ích là hạn chế độ phức tạp của các hàm tác động lên chúng. Chẳng hạn, cận radius-margin nổi tiếng :cite:`Vapnik95` trong máy vector hỗ trợ và Định lý Hội tụ Perceptron :cite:`Novikoff62` đều dựa vào đầu vào có chuẩn bị chặn.

Về trực giác, việc chuẩn hóa này phù hợp với các optimizer của chúng ta
vì nó đặt các tham số *a priori* trên một thang đo tương tự.
Do đó, tự nhiên ta sẽ đặt câu hỏi liệu một bước chuẩn hóa tương ứng *bên trong* mạng sâu
có thể mang lại lợi ích hay không. Mặc dù đây không hẳn là lập luận dẫn đến sự ra đời của chuẩn hóa theo batch :cite:`Ioffe.Szegedy.2015`, nhưng đây là cách hữu ích để hiểu nó và họ hàng của nó, chuẩn hóa theo lớp (layer normalization) :cite:`Ba.Kiros.Hinton.2016`, trong một khuôn khổ thống nhất.

Thứ hai, với một MLP hoặc CNN điển hình, trong quá trình huấn luyện,
các biến
ở các lớp trung gian (ví dụ, đầu ra của phép biến đổi affine trong MLP)
có thể nhận các giá trị với độ lớn rất khác nhau:
theo chiều các lớp từ đầu vào đến đầu ra, giữa các unit trong cùng một lớp,
và theo thời gian do chúng ta cập nhật các tham số mô hình.
Các nhà phát minh chuẩn hóa theo batch đã không chính thức giả định
rằng sự trôi dạt trong phân phối của các biến như vậy có thể cản trở sự hội tụ của mạng.
Về trực giác, chúng ta có thể phỏng đoán rằng nếu một
lớp có activation của biến lớn gấp 100 lần lớp khác,
điều này có thể đòi hỏi các điều chỉnh bù trừ đối với learning rate. Các solver thích nghi
như AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`, hoặc Distributed Shampoo :cite:`anil2020scalable` nhằm giải quyết vấn đề này từ góc độ tối ưu hóa, chẳng hạn bằng cách bổ sung các khía cạnh của phương pháp bậc hai.
Giải pháp thay thế là ngăn vấn đề xảy ra, đơn giản bằng cách chuẩn hóa thích nghi.

Thứ ba, các mạng sâu hơn phức tạp và thường dễ overfitting hơn.
Điều này khiến regularization trở nên quan trọng hơn. Một kỹ thuật regularization phổ biến là chèn nhiễu.
Điều này đã được biết đến từ lâu, chẳng hạn đối với việc chèn nhiễu vào
đầu vào :cite:`Bishop.1995`. Nó cũng là cơ sở của dropout trong :numref:`sec_dropout`. Hóa ra, khá tình cờ, chuẩn hóa theo batch mang lại cả ba lợi ích: tiền xử lý, ổn định số học và regularization.

Chuẩn hóa theo batch được áp dụng cho từng lớp, hoặc tùy chọn cho tất cả các lớp:
Trong mỗi vòng lặp huấn luyện,
trước tiên chúng ta chuẩn hóa các đầu vào (của chuẩn hóa theo batch)
bằng cách trừ đi trung bình của chúng và
chia cho độ lệch chuẩn của chúng,
trong đó cả hai được ước lượng dựa trên thống kê của minibatch hiện tại.
Tiếp theo, chúng ta áp dụng một hệ số scale và một offset để khôi phục các bậc
tự do đã mất. Chính do việc *chuẩn hóa* này dựa trên thống kê của *batch*
mà *chuẩn hóa theo batch* có tên gọi như vậy.

Lưu ý rằng nếu cố áp dụng chuẩn hóa theo batch với minibatch có kích thước 1,
chúng ta sẽ không thể học được gì.
Đó là vì sau khi trừ đi các giá trị trung bình,
mỗi hidden unit sẽ nhận giá trị 0.
Như bạn có thể đoán, vì chúng ta dành cả một phần cho chuẩn hóa theo batch,
với minibatch đủ lớn, phương pháp này tỏ ra hiệu quả và ổn định.
Một điểm cần rút ra ở đây là khi áp dụng chuẩn hóa theo batch,
việc chọn kích thước batch
còn quan trọng hơn so với khi không dùng chuẩn hóa theo batch, hoặc ít nhất,
cần hiệu chỉnh phù hợp khi chúng ta có thể điều chỉnh kích thước batch.

Ký hiệu $\mathcal{B}$ là một minibatch và đặt $\mathbf{x} \in \mathcal{B}$ là một đầu vào cho
chuẩn hóa theo batch ($\textrm{BN}$). Trong trường hợp này, chuẩn hóa theo batch được định nghĩa như sau:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

Trong :eqref:`eq_batchnorm`,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ là trung bình mẫu
và $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ là độ lệch chuẩn mẫu của minibatch $\mathcal{B}$.
Sau khi áp dụng chuẩn hóa,
minibatch thu được
có trung bình bằng không và phương sai đơn vị.
Việc chọn phương sai đơn vị
(thay vì một con số đặc biệt nào khác) là tùy ý. Chúng ta khôi phục bậc tự do này
bằng cách thêm một
*tham số scale* theo từng phần tử $\boldsymbol{\gamma}$ và *tham số shift* $\boldsymbol{\beta}$
có cùng hình dạng với $\mathbf{x}$. Cả hai đều là các tham số
cần được học trong quá trình huấn luyện mô hình.

Độ lớn của các biến
ở các lớp trung gian không thể phân kỳ trong quá trình huấn luyện
vì chuẩn hóa theo batch chủ động căn giữa và điều chỉnh lại tỷ lệ của chúng
về một trung bình và kích thước nhất định (thông qua $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
Kinh nghiệm thực tế xác nhận rằng, như đã đề cập khi thảo luận về việc điều chỉnh tỷ lệ đặc trưng, chuẩn hóa theo batch dường như cho phép dùng learning rate mạnh hơn.
Chúng ta tính $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ trong :eqref:`eq_batchnorm` như sau:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Lưu ý rằng chúng ta thêm một hằng số nhỏ $\epsilon > 0$
vào ước lượng phương sai
để đảm bảo rằng chúng ta không bao giờ cố chia cho không,
ngay cả trong những trường hợp mà ước lượng phương sai thực nghiệm có thể rất nhỏ hoặc bằng không.
Các ước lượng $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ khắc phục vấn đề scale
bằng cách sử dụng các ước lượng nhiễu của trung bình và phương sai.
Bạn có thể nghĩ rằng tính nhiễu này sẽ là một vấn đề.
Ngược lại, nó thực sự có lợi.

Đây hóa ra là một chủ đề lặp lại trong deep learning.
Vì những lý do chưa được mô tả đầy đủ về mặt lý thuyết,
nhiều nguồn nhiễu khác nhau trong quá trình tối ưu hóa
thường dẫn đến huấn luyện nhanh hơn và ít overfitting hơn:
sự biến thiên này dường như hoạt động như một dạng regularization.
:citet:`Teye.Azizpour.Smith.2018` và :citet:`Luo.Wang.Shao.ea.2018`
đã liên hệ các đặc tính của chuẩn hóa theo batch với Bayesian prior và penalty tương ứng.
Đặc biệt, điều này làm sáng tỏ phần nào câu hỏi
vì sao chuẩn hóa theo batch hoạt động tốt nhất với các minibatch có kích thước vừa phải trong khoảng 50--100.
Kích thước minibatch cụ thể này dường như chèn đúng "lượng" nhiễu vào mỗi lớp, cả về scale thông qua $\hat{\boldsymbol{\sigma}}$ lẫn về offset thông qua $\hat{\boldsymbol{\mu}}$: một
minibatch lớn hơn regularize ít hơn do các ước lượng ổn định hơn, trong khi các minibatch quá nhỏ
phá hủy tín hiệu hữu ích do phương sai cao. Khám phá sâu hơn hướng này, bằng cách xem xét các kiểu
tiền xử lý và lọc thay thế, có thể dẫn đến những kiểu regularization hiệu quả khác.

Với một mô hình đã được huấn luyện, bạn có thể nghĩ rằng
chúng ta sẽ ưu tiên sử dụng toàn bộ tập dữ liệu
để ước lượng trung bình và phương sai.
Sau khi huấn luyện hoàn tất, tại sao chúng ta lại muốn
cùng một hình ảnh được phân loại khác nhau,
tùy thuộc vào batch mà nó tình cờ thuộc về?
Trong quá trình huấn luyện, việc tính toán chính xác như vậy là không khả thi
vì các biến trung gian
cho tất cả các ví dụ dữ liệu
thay đổi mỗi khi chúng ta cập nhật mô hình.
Tuy nhiên, khi mô hình đã được huấn luyện,
chúng ta có thể tính trung bình và phương sai
của các biến ở từng layer dựa trên toàn bộ tập dữ liệu.
Thật vậy, đây là thực hành tiêu chuẩn cho
các mô hình sử dụng batch normalization;
do đó, các layer batch normalization hoạt động khác nhau
ở *chế độ huấn luyện* (chuẩn hóa theo thống kê minibatch)
so với ở *chế độ dự đoán* (chuẩn hóa theo thống kê tập dữ liệu).
Ở dạng này, chúng rất giống với hành vi của regularization dropout trong :numref:`sec_dropout`,
trong đó nhiễu chỉ được đưa vào khi huấn luyện.


## Các Layer Batch Normalization

Các cách triển khai batch normalization cho fully connected layer
và convolutional layer có đôi chút khác nhau.
Một khác biệt quan trọng giữa batch normalization và các layer khác
là vì batch normalization xử lý toàn bộ một minibatch cùng lúc,
chúng ta không thể đơn giản bỏ qua chiều batch
như đã làm trước đây khi giới thiệu các layer khác.

### Fully Connected Layers

Khi áp dụng batch normalization cho fully connected layer,
:citet:`Ioffe.Szegedy.2015`, trong bài báo gốc của họ đã chèn batch normalization sau phép biến đổi affine
và *trước* hàm kích hoạt phi tuyến. Các ứng dụng sau này đã thử nghiệm
chèn batch normalization ngay *sau* các hàm kích hoạt.
Ký hiệu đầu vào của fully connected layer là $\mathbf{x}$,
phép biến đổi affine
là $\mathbf{W}\mathbf{x} + \mathbf{b}$ (với tham số trọng số $\mathbf{W}$ và tham số bias $\mathbf{b}$),
và hàm kích hoạt là $\phi$,
ta có thể biểu diễn phép tính của đầu ra $\mathbf{h}$ của một fully connected layer
có hỗ trợ batch normalization như sau:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Hãy nhớ rằng trung bình và phương sai được tính
trên *cùng* minibatch
mà phép biến đổi được áp dụng.

### Convolutional Layers

Tương tự, với convolutional layer,
chúng ta có thể áp dụng batch normalization sau phép tích chập
nhưng trước hàm kích hoạt phi tuyến. Khác biệt chính so với batch normalization
trong fully connected layer là chúng ta áp dụng phép toán theo từng channel
*trên tất cả các vị trí*. Điều này tương thích với giả định về tính bất biến
theo phép tịnh tiến đã dẫn đến convolution: chúng ta giả định rằng vị trí cụ thể của một mẫu
trong ảnh không quan trọng đối với mục đích thấu hiểu.

Giả sử các minibatch của chúng ta chứa $m$ ví dụ
và với mỗi channel,
đầu ra của phép tích chập có chiều cao $p$ và chiều rộng $q$.
Đối với convolutional layer, chúng ta thực hiện mỗi batch normalization
đồng thời trên $m \cdot p \cdot q$ phần tử cho mỗi output channel.
Do đó, chúng ta thu thập các giá trị tại mọi vị trí không gian
khi tính trung bình và phương sai
và vì thế
áp dụng cùng một trung bình và phương sai
trong một channel nhất định
để chuẩn hóa giá trị tại từng vị trí không gian.
Mỗi channel có các tham số scale và shift riêng,
cả hai đều là scalar.

### Layer Normalization
:label:`subsec_layer-normalization-in-bn`

Lưu ý rằng trong ngữ cảnh của convolution, batch normalization vẫn được xác định rõ ngay cả với
minibatch có kích thước 1: xét cho cùng, chúng ta có tất cả các vị trí trên một ảnh để lấy trung bình. Do đó,
trung bình và phương sai được xác định rõ, ngay cả khi chúng chỉ nằm trong một quan sát duy nhất. Điều này
đã dẫn :citet:`Ba.Kiros.Hinton.2016` đến việc giới thiệu khái niệm *layer normalization*. Nó hoạt động giống
batch norm, chỉ khác là được áp dụng cho từng quan sát một. Do đó, cả offset lẫn hệ số scale đều là scalar. Với vector $\mathbf{x}$ có $n$ chiều, layer norm được cho bởi 

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

trong đó scale và offset được áp dụng theo từng hệ số
và được cho bởi 

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

Như trước đây, chúng ta thêm một offset nhỏ $\epsilon > 0$ để tránh chia cho không. Một trong những lợi ích lớn của việc sử dụng layer normalization là nó ngăn sự phân kỳ. Xét cho cùng, nếu bỏ qua $\epsilon$, đầu ra của layer normalization không phụ thuộc vào scale. Nghĩa là, ta có $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ với mọi lựa chọn $\alpha \neq 0$. Điều này trở thành đẳng thức khi $|\alpha| \to \infty$ (đẳng thức gần đúng là do offset $\epsilon$ cho phương sai). 

Một ưu điểm khác của layer normalization là nó không phụ thuộc vào kích thước minibatch. Nó cũng độc lập với việc chúng ta đang ở chế độ huấn luyện hay kiểm tra. Nói cách khác, đây đơn giản là một phép biến đổi xác định chuẩn hóa các activation về một scale nhất định. Điều này có thể rất hữu ích để ngăn sự phân kỳ trong tối ưu hóa. Chúng ta bỏ qua các chi tiết thêm và khuyến nghị độc giả quan tâm tham khảo bài báo gốc.

### Batch Normalization Trong Khi Dự Đoán

Như đã đề cập trước đó, batch normalization thường hoạt động khác nhau
ở chế độ huấn luyện so với chế độ dự đoán.
Thứ nhất, nhiễu trong trung bình mẫu và phương sai mẫu
phát sinh do ước lượng chúng trên các minibatch
không còn mong muốn sau khi chúng ta đã huấn luyện mô hình.
Thứ hai, chúng ta có thể không có điều kiện
để tính các thống kê chuẩn hóa theo từng batch.
Ví dụ,
chúng ta có thể cần áp dụng mô hình để đưa ra từng dự đoán một.

Thông thường, sau khi huấn luyện, chúng ta sử dụng toàn bộ tập dữ liệu
để tính các ước lượng ổn định cho thống kê của biến
và sau đó cố định chúng tại thời điểm dự đoán.
Do đó, batch normalization hoạt động khác nhau trong quá trình huấn luyện so với lúc kiểm tra.
Hãy nhớ rằng dropout cũng có đặc điểm này.

## (**Triển Khai Từ Đầu**)

Để xem batch normalization hoạt động như thế nào trong thực tế, dưới đây chúng ta tự triển khai một phiên bản từ đầu.

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        # `linen.Module.variables` have a `value` attribute containing the array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

Bây giờ chúng ta có thể [**tạo một layer `BatchNorm` đúng chuẩn.**]
Layer của chúng ta sẽ duy trì các tham số phù hợp
cho scale `gamma` và shift `beta`,
cả hai sẽ được cập nhật trong quá trình huấn luyện.
Ngoài ra, layer của chúng ta sẽ duy trì
các trung bình động của trung bình và phương sai
để sử dụng sau này trong quá trình dự đoán của mô hình.

Tạm gác các chi tiết thuật toán sang một bên,
hãy lưu ý design pattern làm nền tảng cho cách triển khai layer của chúng ta.
Thông thường, chúng ta định nghĩa phần toán học trong một hàm riêng, chẳng hạn `batch_norm`.
Sau đó, chúng ta tích hợp chức năng này vào một custom layer,
mà mã của nó chủ yếu xử lý các công việc bookkeeping,
chẳng hạn như chuyển dữ liệu đến đúng device context,
cấp phát và khởi tạo mọi biến cần thiết,
theo dõi các trung bình động (ở đây là cho trung bình và phương sai), v.v.
Pattern này cho phép tách biệt rõ ràng phần toán học khỏi boilerplate code.
Cũng lưu ý rằng để thuận tiện,
ở đây chúng ta không lo việc tự động suy luận shape đầu vào;
do đó, chúng ta cần chỉ định số lượng feature xuyên suốt.
Đến nay, mọi framework deep learning hiện đại đều cung cấp khả năng tự động phát hiện kích thước và shape trong các
API batch normalization cấp cao (trong thực tế chúng ta sẽ dùng cách này).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

Chúng ta đã sử dụng `momentum` để điều khiển việc tổng hợp các ước lượng trung bình và phương sai trước đó. Đây phần nào là một cách gọi sai vì nó hoàn toàn không liên quan đến hạng *momentum* trong tối ưu hóa. Tuy vậy, đây là tên gọi thường được chấp nhận cho hạng này và để tuân theo quy ước đặt tên của API, chúng ta dùng cùng tên biến đó trong mã.

## [**LeNet với Batch Normalization**]

Để xem cách áp dụng `BatchNorm` trong ngữ cảnh,
dưới đây chúng ta áp dụng nó cho một mô hình LeNet truyền thống (:numref:`sec_lenet`).
Hãy nhớ rằng batch normalization được áp dụng
sau convolutional layer hoặc fully connected layer
nhưng trước các hàm kích hoạt tương ứng.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
Vì các layer `BatchNorm` cần tính các thống kê batch
(trung bình và phương sai), Flax theo dõi dictionary `batch_stats`, cập nhật
chúng với mỗi minibatch. Các collection như `batch_stats` có thể được lưu trong
đối tượng `TrainState` (trong class `d2l.Trainer` được định nghĩa tại
:numref:`oo-design-training`) như một thuộc tính và trong quá trình forward pass của mô hình,
chúng cần được truyền đến đối số `mutable`, để Flax trả về các
biến đã bị thay đổi.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

Như trước, chúng ta sẽ [**huấn luyện mạng trên tập dữ liệu Fashion-MNIST**].
Mã này gần như giống hệt với khi chúng ta lần đầu huấn luyện LeNet.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

Hãy [**xem xét tham số scale `gamma`
và tham số shift `beta`**] đã học được
từ layer batch normalization đầu tiên.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**Cách triển khai ngắn gọn**]

So với lớp `BatchNorm`
mà chúng ta vừa tự định nghĩa,
chúng ta có thể sử dụng trực tiếp lớp `BatchNorm` được định nghĩa trong các API cấp cao của framework deep learning.
Mã gần như giống hệt
với cách triển khai ở trên của chúng ta, ngoại trừ việc chúng ta không còn cần cung cấp các đối số bổ sung để nó xác định đúng các chiều.

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

Dưới đây, chúng ta [**sử dụng cùng các siêu tham số để huấn luyện mô hình.**]
Lưu ý rằng như thường lệ, biến thể API cấp cao chạy nhanh hơn nhiều
vì mã của nó đã được biên dịch sang C++ hoặc CUDA,
trong khi cách triển khai tùy chỉnh của chúng ta phải được Python thông dịch.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Thảo luận

Theo trực giác, batch normalization được cho là
làm cho bề mặt tối ưu hóa mượt hơn.
Tuy nhiên, chúng ta phải cẩn thận phân biệt giữa
những trực giác mang tính suy đoán và các lời giải thích thực sự
cho những hiện tượng quan sát được khi huấn luyện các mô hình sâu.
Hãy nhớ rằng ngay từ đầu chúng ta còn không biết vì sao
các mạng nơ-ron sâu đơn giản hơn (MLP và CNN thông thường)
lại tổng quát hóa tốt.
Ngay cả với dropout và weight decay,
chúng vẫn linh hoạt đến mức khả năng tổng quát hóa sang dữ liệu chưa thấy
có lẽ cần các đảm bảo tổng quát hóa theo lý thuyết học tinh vi hơn đáng kể.

Bài báo gốc đề xuất batch normalization :cite:`Ioffe.Szegedy.2015`, ngoài việc giới thiệu một công cụ mạnh mẽ và hữu ích,
còn đưa ra lời giải thích cho lý do nó hoạt động:
bằng cách giảm *internal covariate shift*.
Có lẽ *internal covariate shift* được hiểu là
một điều gì đó giống với trực giác được trình bày ở trên---ý niệm
rằng phân phối của các giá trị biến thay đổi
trong quá trình huấn luyện.
Tuy nhiên, lời giải thích này có hai vấn đề:
i) Sự trôi dạt này rất khác với *covariate shift*,
khiến tên gọi trở thành không chính xác. Nếu có gì tương tự, nó gần với concept drift hơn.
ii) Lời giải thích đưa ra một trực giác chưa được đặc tả đầy đủ
nhưng để lại câu hỏi *chính xác vì sao kỹ thuật này hoạt động*
vẫn còn bỏ ngỏ và cần một lời giải thích chặt chẽ.
Xuyên suốt cuốn sách này, chúng tôi hướng tới việc truyền đạt những trực giác mà các practitioner
dùng để định hướng quá trình phát triển mạng nơ-ron sâu.
Tuy nhiên, chúng tôi tin rằng điều quan trọng
là phải tách biệt các trực giác định hướng này
khỏi các sự thật khoa học đã được xác lập.
Cuối cùng, khi bạn nắm vững tài liệu này
và bắt đầu viết các bài báo nghiên cứu của riêng mình,
bạn sẽ muốn phân định rõ ràng
giữa các tuyên bố kỹ thuật và những phỏng đoán.

Sau thành công của batch normalization,
lời giải thích của nó theo hướng *internal covariate shift*
đã nhiều lần xuất hiện trong các cuộc tranh luận trong tài liệu kỹ thuật
và các thảo luận rộng hơn về cách trình bày nghiên cứu machine learning.
Trong một bài phát biểu đáng nhớ khi nhận Test of Time Award
tại hội nghị NeurIPS năm 2017,
Ali Rahimi đã sử dụng *internal covariate shift*
làm trọng tâm cho lập luận ví
thực hành deep learning hiện đại với thuật giả kim.
Sau đó, ví dụ này được xem xét lại chi tiết
trong một position paper nêu ra
các xu hướng đáng lo ngại trong machine learning :cite:`Lipton.Steinhardt.2018`.
Các tác giả khác
đã đề xuất những lời giải thích thay thế cho thành công của batch normalization,
trong đó một số người :cite:`Santurkar.Tsipras.Ilyas.ea.2018`
cho rằng thành công của batch normalization có được bất chấp việc nó thể hiện hành vi
theo một số khía cạnh trái ngược với những gì bài báo gốc tuyên bố.


Chúng tôi lưu ý rằng *internal covariate shift*
không đáng bị phê phán hơn bất kỳ
tuyên bố mơ hồ tương tự nào trong hàng nghìn tuyên bố
được đưa ra mỗi năm trong tài liệu kỹ thuật về machine learning.
Có lẽ việc nó trở thành tâm điểm của những cuộc tranh luận này
là nhờ mức độ dễ nhận biết rộng rãi đối với đối tượng mục tiêu.
Batch normalization đã chứng tỏ là một phương pháp không thể thiếu,
được áp dụng trong gần như mọi bộ phân loại ảnh đã deploy,
mang về cho bài báo giới thiệu kỹ thuật này
hàng chục nghìn lượt trích dẫn. Tuy nhiên, chúng tôi phỏng đoán rằng các nguyên tắc định hướng
của regularization thông qua việc thêm nhiễu, tăng tốc thông qua rescaling, và cuối cùng là preprocessing
có thể sẽ dẫn đến nhiều phát minh hơn nữa về layer và kỹ thuật trong tương lai.

Ở khía cạnh thực tiễn hơn, có một số điểm cần nhớ về batch normalization:

* Trong quá trình huấn luyện mô hình, batch normalization liên tục điều chỉnh đầu ra trung gian của
  mạng bằng cách sử dụng giá trị trung bình và độ lệch chuẩn của minibatch, để
  các giá trị đầu ra trung gian ở mỗi layer trong toàn bộ mạng nơ-ron ổn định hơn.
* Batch normalization có đôi chút khác biệt đối với fully connected layer so với convolutional layer. Trên thực tế,
  đối với convolutional layer, layer normalization đôi khi có thể được dùng thay thế.
* Giống như dropout layer, các layer batch normalization có hành vi khác nhau
  ở chế độ huấn luyện so với chế độ dự đoán.
* Batch normalization hữu ích cho regularization và cải thiện sự hội tụ trong tối ưu hóa. Ngược lại,
  động cơ ban đầu là giảm internal covariate shift dường như không phải là một lời giải thích hợp lệ.
* Để có các mô hình robust hơn và ít nhạy cảm hơn với nhiễu đầu vào, hãy cân nhắc loại bỏ batch normalization :cite:`wang2022removing`.

## Bài tập

1. Chúng ta có nên loại bỏ tham số bias khỏi fully connected layer hoặc convolutional layer trước batch normalization không? Vì sao?
1. So sánh learning rate của LeNet có và không có batch normalization.
    1. Vẽ biểu đồ mức tăng của validation accuracy.
    1. Bạn có thể đặt learning rate lớn đến mức nào trước khi tối ưu hóa thất bại trong cả hai trường hợp?
1. Chúng ta có cần batch normalization ở mọi layer không? Hãy thử nghiệm.
1. Triển khai một phiên bản "lite" của batch normalization chỉ loại bỏ giá trị trung bình, hoặc một phiên bản khác
   chỉ loại bỏ phương sai. Nó hoạt động như thế nào?
1. Cố định các tham số `beta` và `gamma`. Quan sát và phân tích kết quả.
1. Bạn có thể thay dropout bằng batch normalization không? Hành vi thay đổi như thế nào?
1. Ý tưởng nghiên cứu: hãy nghĩ về các phép biến đổi normalization khác mà bạn có thể áp dụng:
    1. Bạn có thể áp dụng probability integral transform không?
    1. Bạn có thể dùng một ước lượng covariance full-rank không? Vì sao có lẽ bạn không nên làm vậy?
    1. Bạn có thể dùng các biến thể ma trận compact khác (block-diagonal, low-displacement rank, Monarch, v.v.) không?
    1. Liệu nén sparsification có hoạt động như một regularizer không?
    1. Có các phép chiếu khác (ví dụ: convex cone, các phép biến đổi dành riêng cho symmetry group) mà bạn có thể sử dụng không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18005)
:end_tab: