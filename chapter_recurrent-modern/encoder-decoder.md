```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Kiến trúc Encoder--Decoder
:label:`sec_encoder-decoder`

Nhìn chung, trong các bài toán sequence-to-sequence
như dịch máy
(:numref:`sec_machine_translation`),
đầu vào và đầu ra có độ dài khác nhau
và không được căn chỉnh.
Cách tiếp cận tiêu chuẩn để xử lý loại dữ liệu này
là thiết kế một kiến trúc *encoder--decoder* (:numref:`fig_encoder_decoder`)
gồm hai thành phần chính:
một *encoder* nhận vào một sequence có độ dài thay đổi,
và một *decoder* hoạt động như một language model có điều kiện,
nhận đầu vào đã được mã hóa
cùng với ngữ cảnh bên trái của sequence đích
và dự đoán token tiếp theo trong sequence đích.


![Kiến trúc encoder--decoder.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Hãy lấy bài toán dịch máy từ tiếng Anh sang tiếng Pháp làm ví dụ.
Cho một sequence đầu vào bằng tiếng Anh:
"They", "are", "watching", ".",
kiến trúc encoder--decoder này
trước tiên mã hóa đầu vào có độ dài thay đổi thành một trạng thái,
sau đó giải mã trạng thái đó
để tạo ra sequence đã dịch,
từng token một, làm đầu ra:
"Ils", "regardent", ".".
Vì kiến trúc encoder--decoder
là nền tảng của nhiều mô hình sequence-to-sequence khác nhau
trong các phần tiếp theo,
phần này sẽ chuyển kiến trúc đó
thành một interface sẽ được triển khai sau.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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
```

## (**Encoder**)

Trong interface của encoder,
chúng ta chỉ xác định rằng
encoder nhận các sequence có độ dài thay đổi làm đầu vào `X`.
Việc triển khai sẽ được cung cấp
bởi bất kỳ model nào kế thừa lớp `Encoder` cơ sở này.

```{.python .input}
%%tab mxnet
class Encoder(nn.Block):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Encoder(tf.keras.layers.Layer):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def call(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## [**Decoder**]

Trong interface của decoder sau đây,
chúng ta bổ sung một phương thức `init_state`
để chuyển đổi đầu ra của encoder (`enc_all_outputs`)
thành trạng thái đã được mã hóa.
Lưu ý rằng bước này
có thể yêu cầu các đầu vào bổ sung,
chẳng hạn như độ dài hợp lệ của đầu vào,
đã được giải thích
trong :numref:`sec_machine_translation`.
Để tạo một sequence có độ dài thay đổi từng token một,
mỗi lần decoder có thể ánh xạ một đầu vào
(ví dụ: token được tạo ở time step trước đó)
và trạng thái đã được mã hóa
thành một token đầu ra tại time step hiện tại.

```{.python .input}
%%tab mxnet
class Decoder(nn.Block):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Decoder(tf.keras.layers.Layer):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## [**Kết hợp Encoder và Decoder**]

Trong quá trình forward propagation,
đầu ra của encoder
được sử dụng để tạo trạng thái đã được mã hóa,
và trạng thái này sẽ tiếp tục được
decoder sử dụng làm một trong các đầu vào của nó.

```{.python .input}
%%tab mxnet, pytorch
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]
```

```{.python .input}
%%tab jax
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

Trong phần tiếp theo,
chúng ta sẽ xem cách áp dụng RNN để thiết kế
các mô hình sequence-to-sequence dựa trên
kiến trúc encoder--decoder này.


## Tóm tắt

Các kiến trúc encoder-decoder
có thể xử lý cả đầu vào và đầu ra
đều gồm các sequence có độ dài thay đổi
và vì vậy phù hợp với các bài toán sequence-to-sequence
như dịch máy.
Encoder nhận một sequence có độ dài thay đổi làm đầu vào
và chuyển đổi nó thành một trạng thái có shape cố định.
Decoder ánh xạ trạng thái đã được mã hóa có shape cố định
thành một sequence có độ dài thay đổi.


## Bài tập

1. Giả sử chúng ta sử dụng neural network để triển khai kiến trúc encoder--decoder. Encoder và decoder có nhất thiết phải là cùng một loại neural network không?
1. Ngoài dịch máy, bạn có thể nghĩ đến ứng dụng nào khác mà kiến trúc encoder--decoder có thể được áp dụng?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3864)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18021)
:end_tab: