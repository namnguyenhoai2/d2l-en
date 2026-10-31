```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Tài liệu
:begin_tab:`mxnet`
Mặc dù không thể giới thiệu từng hàm và lớp MXNet 
(và thông tin có thể nhanh chóng trở nên lỗi thời), 
[tài liệu API](https://mxnet.apache.org/versions/1.8.0/api) 
cùng với các [bài hướng dẫn](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) và ví dụ bổ sung 
cung cấp những tài liệu như vậy. 
Phần này đưa ra một số hướng dẫn về cách khám phá API MXNet.
:end_tab:

:begin_tab:`pytorch`
Mặc dù không thể giới thiệu từng hàm và lớp PyTorch 
(và thông tin có thể nhanh chóng trở nên lỗi thời), 
[tài liệu API](https://pytorch.org/docs/stable/index.html) cùng với các [bài hướng dẫn](https://pytorch.org/tutorials/beginner/basics/intro.html) và ví dụ bổ sung 
cung cấp những tài liệu như vậy.
Phần này đưa ra một số hướng dẫn về cách khám phá API PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Mặc dù không thể giới thiệu từng hàm và lớp TensorFlow 
(và thông tin có thể nhanh chóng trở nên lỗi thời), 
[tài liệu API](https://www.tensorflow.org/api_docs) cùng với các [bài hướng dẫn](https://www.tensorflow.org/tutorials) và ví dụ bổ sung 
cung cấp những tài liệu như vậy. 
Phần này đưa ra một số hướng dẫn về cách khám phá API TensorFlow.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
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
```

## Các hàm và lớp trong một module

Để biết những hàm và lớp nào có thể được gọi trong một module,
chúng ta gọi hàm `dir`. Ví dụ, chúng ta có thể
(**truy vấn tất cả thuộc tính trong module để tạo số ngẫu nhiên**):

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

Nhìn chung, chúng ta có thể bỏ qua các hàm bắt đầu và kết thúc bằng `__` (các đối tượng đặc biệt trong Python) 
hoặc các hàm bắt đầu bằng một `_` duy nhất (thường là các hàm nội bộ). 
Dựa trên tên của các hàm hoặc thuộc tính còn lại, 
chúng ta có thể đoán rằng module này cung cấp 
nhiều phương thức để tạo số ngẫu nhiên, 
bao gồm lấy mẫu từ phân phối đều (`uniform`),
phân phối chuẩn (`normal`) và phân phối đa thức (`multinomial`).

## Các hàm và lớp cụ thể

Để biết hướng dẫn cụ thể về cách sử dụng một hàm hoặc lớp,
chúng ta có thể gọi hàm `help`. Ví dụ, hãy cùng
[**khám phá hướng dẫn sử dụng hàm `ones` của tensor**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

Từ tài liệu, chúng ta có thể thấy rằng hàm `ones` 
tạo một tensor mới với hình dạng được chỉ định 
và đặt tất cả các phần tử thành giá trị 1. 
Bất cứ khi nào có thể, bạn nên (**chạy một kiểm thử nhanh**) 
để xác nhận cách hiểu của mình:

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

Trong Jupyter notebook, chúng ta có thể sử dụng `?` để hiển thị tài liệu trong một
cửa sổ khác. Ví dụ, `list?` sẽ tạo nội dung
gần như giống hệt với `help(list)`,
và hiển thị nội dung đó trong một cửa sổ trình duyệt mới.
Ngoài ra, nếu sử dụng hai dấu chấm hỏi, chẳng hạn như `list??`,
mã Python triển khai hàm cũng sẽ được hiển thị.

Tài liệu chính thức cung cấp rất nhiều mô tả và ví dụ vượt quá phạm vi của cuốn sách này. 
Chúng tôi nhấn mạnh những trường hợp sử dụng quan trọng 
giúp bạn nhanh chóng bắt đầu giải quyết các vấn đề thực tế, 
thay vì cố gắng bao quát đầy đủ mọi nội dung. 
Chúng tôi cũng khuyến khích bạn nghiên cứu mã nguồn của các thư viện 
để xem những ví dụ về các triển khai mã production chất lượng cao. 
Qua đó, bạn sẽ trở thành một kỹ sư giỏi hơn
đồng thời cũng trở thành một nhà khoa học giỏi hơn.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17972)
:end_tab: