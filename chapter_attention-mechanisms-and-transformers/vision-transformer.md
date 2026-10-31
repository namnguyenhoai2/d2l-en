```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# Transformers cho Thị giác
:label:`sec_vision-transformer`

Kiến trúc Transformer ban đầu được đề xuất
cho việc học sequence-to-sequence,
tập trung vào dịch máy.
Sau đó, Transformers đã trở thành mô hình được lựa chọn
trong nhiều tác vụ xử lý ngôn ngữ tự nhiên :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`.
Tuy nhiên, trong lĩnh vực thị giác máy tính,
kiến trúc thống trị vẫn là
CNN (:numref:`chap_modern_cnn`).
Một cách tự nhiên, các nhà nghiên cứu bắt đầu tự hỏi
liệu có thể làm tốt hơn hay không
bằng cách điều chỉnh các mô hình Transformer cho dữ liệu ảnh.
Câu hỏi này đã khơi dậy sự quan tâm to lớn
trong cộng đồng thị giác máy tính.
Gần đây, :citet:`ramachandran2019stand` đã đề xuất
một phương án thay thế convolution bằng self-attention.
Tuy nhiên, việc sử dụng các mẫu chuyên biệt trong attention
khiến việc mở rộng quy mô mô hình trên các hardware accelerator trở nên khó khăn.
Sau đó, :citet:`cordonnier2020relationship` đã chứng minh về mặt lý thuyết
rằng self-attention có thể học cách hoạt động tương tự convolution.
Về mặt thực nghiệm, các patch $2 \times 2$ được lấy từ ảnh làm đầu vào,
nhưng kích thước patch nhỏ khiến mô hình
chỉ áp dụng được cho dữ liệu ảnh có độ phân giải thấp.

Không có ràng buộc cụ thể về kích thước patch,
*vision Transformers* (ViTs)
trích xuất các patch từ ảnh
và đưa chúng vào một Transformer encoder
để thu được biểu diễn toàn cục,
cuối cùng sẽ được biến đổi để phân loại :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Đáng chú ý, Transformers có khả năng mở rộng tốt hơn CNN:
khi huấn luyện các mô hình lớn hơn trên các tập dữ liệu lớn hơn,
vision Transformers vượt trội hơn ResNets với một khoảng cách đáng kể.
Tương tự bối cảnh thiết kế kiến trúc mạng trong xử lý ngôn ngữ tự nhiên,
Transformers cũng đã trở thành yếu tố thay đổi cuộc chơi trong thị giác máy tính.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Mô hình

:numref:`fig_vit` minh họa
kiến trúc mô hình của vision Transformers.
Kiến trúc này gồm một stem
chia ảnh thành các patch,
một body dựa trên Transformer encoder nhiều tầng,
và một head biến đổi biểu diễn toàn cục
thành nhãn đầu ra.

![Kiến trúc vision Transformer. Trong ví dụ này, một ảnh được chia thành chín patch. Một token đặc biệt “&lt;cls&gt;” và chín patch ảnh đã được làm phẳng lần lượt được biến đổi thông qua patch embedding và $\mathit{n}$ khối Transformer encoder thành mười biểu diễn. Biểu diễn “&lt;cls&gt;” tiếp tục được biến đổi thành nhãn đầu ra.](../img/vit.svg)
:label:`fig_vit`

Xét một ảnh đầu vào có chiều cao $h$, chiều rộng $w$,
và $c$ kênh.
Khi chỉ định cả chiều cao và chiều rộng của patch là $p$,
ảnh được chia thành một chuỗi gồm $m = hw/p^2$ patch,
trong đó mỗi patch được làm phẳng thành một vector có độ dài $cp^2$.
Theo cách này, các patch ảnh có thể được Transformer encoder xử lý tương tự các token trong chuỗi văn bản.
Một token “&lt;cls&gt;” (class) đặc biệt và
$m$ patch ảnh đã làm phẳng được chiếu tuyến tính
thành một chuỗi gồm $m+1$ vector,
sau đó cộng với positional embedding có thể học được.
Transformer encoder nhiều tầng
biến đổi $m+1$ vector đầu vào
thành cùng số lượng biểu diễn vector đầu ra có cùng độ dài.
Nó hoạt động hoàn toàn giống Transformer encoder gốc trong :numref:`fig_transformer`,
chỉ khác ở vị trí của normalization.
Vì token “&lt;cls&gt;” attention tới tất cả các patch ảnh
thông qua self-attention (xem :numref:`fig_cnn-rnn-self-attention`),
biểu diễn của nó từ đầu ra Transformer encoder
sẽ tiếp tục được biến đổi thành nhãn đầu ra.

## Patch Embedding

Để triển khai vision Transformer, hãy bắt đầu
với patch embedding trong :numref:`fig_vit`.
Việc chia ảnh thành các patch
và chiếu tuyến tính các patch đã làm phẳng này
có thể được đơn giản hóa thành một phép convolution duy nhất,
trong đó cả kernel size và stride size đều được đặt bằng kích thước patch.

```{.python .input}
%%tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

Trong ví dụ sau, sử dụng các ảnh có chiều cao và chiều rộng `img_size` làm đầu vào,
patch embedding xuất ra `(img_size//patch_size)**2` patch
được chiếu tuyến tính thành các vector có độ dài `num_hiddens`.

```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Vision Transformer Encoder
:label:`subsec_vit-encoder`

MLP của vision Transformer encoder hơi khác
so với FFN theo vị trí của Transformer encoder gốc
(xem :numref:`subsec_positionwise-ffn`).
Thứ nhất, ở đây hàm kích hoạt sử dụng Gaussian error linear unit (GELU),
có thể được xem là phiên bản mượt hơn của ReLU :cite:`Hendrycks.Gimpel.2016`.
Thứ hai, dropout được áp dụng lên đầu ra của mỗi fully connected layer trong MLP để regularization.

```{.python .input}
%%tab pytorch
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

Việc triển khai khối vision Transformer encoder
tuân theo thiết kế pre-normalization trong :numref:`fig_vit`,
trong đó normalization được áp dụng ngay *trước* multi-head attention hoặc MLP.
Trái ngược với post-normalization ("add & norm" trong :numref:`fig_transformer`),
trong đó normalization được đặt ngay *sau* residual connection,
pre-normalization giúp việc huấn luyện Transformers hiệu quả hoặc hiệu suất hơn :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.

```{.python .input}
%%tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

Cũng như trong :numref:`subsec_transformer-encoder`,
không có khối vision Transformer encoder nào thay đổi shape đầu vào của nó.

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## Kết hợp tất cả

Forward pass của vision Transformers dưới đây rất trực tiếp.
Đầu tiên, các ảnh đầu vào được đưa vào một instance `PatchEmbedding`,
mà đầu ra của nó được nối với token embedding “&lt;cls&gt;”.
Chúng được cộng với positional embedding có thể học được trước khi dropout.
Sau đó, đầu ra được đưa vào Transformer encoder, nơi xếp chồng `num_blks` instance của lớp `ViTBlock`.
Cuối cùng, biểu diễn của token “&lt;cls&gt;” được chiếu bởi network head.

```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """Vision Transformer."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## Huấn luyện

Huấn luyện vision Transformer trên tập dữ liệu Fashion-MNIST cũng giống như cách CNNs được huấn luyện trong :numref:`chap_modern_cnn`.

```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Tóm tắt và Thảo luận

Bạn có thể đã nhận thấy rằng với các tập dữ liệu nhỏ như Fashion-MNIST,
vision Transformer mà chúng ta triển khai
không vượt trội hơn ResNet trong :numref:`sec_resnet`.
Các quan sát tương tự có thể được đưa ra ngay cả trên tập dữ liệu ImageNet (1,2 triệu ảnh).
Điều này là do Transformers *thiếu* những nguyên lý hữu ích của convolution,
chẳng hạn như tính bất biến tịnh tiến và tính cục bộ (:numref:`sec_why-conv`).
Tuy nhiên, bức tranh thay đổi khi huấn luyện các mô hình lớn hơn trên các tập dữ liệu lớn hơn (ví dụ: 300 triệu ảnh),
nơi vision Transformers vượt trội hơn ResNets với khoảng cách lớn trong phân loại ảnh, cho thấy
ưu thế nội tại của Transformers về khả năng mở rộng :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Sự ra đời của vision Transformers
đã thay đổi bối cảnh thiết kế mạng để mô hình hóa dữ liệu ảnh.
Chúng nhanh chóng được chứng minh là hiệu quả trên tập dữ liệu ImageNet
với các chiến lược huấn luyện hiệu quả dữ liệu của DeiT :cite:`touvron2021training`.
Tuy nhiên, độ phức tạp bậc hai của self-attention
(:numref:`sec_self-attention-and-positional-encoding`)
khiến kiến trúc Transformer
ít phù hợp hơn với các ảnh có độ phân giải cao.
Hướng tới một backbone network đa dụng trong thị giác máy tính,
Swin Transformers đã giải quyết độ phức tạp tính toán bậc hai
theo kích thước ảnh (:numref:`subsec_cnn-rnn-self-attention`)
và khôi phục các prior giống convolution,
mở rộng khả năng ứng dụng của Transformers cho một loạt tác vụ thị giác máy tính
ngoài phân loại ảnh với các kết quả state-of-the-art :cite:`liu2021swin`.

## Bài tập

1. Giá trị của `img_size` ảnh hưởng đến thời gian huấn luyện như thế nào?
1. Thay vì chiếu biểu diễn token “&lt;cls&gt;” thành đầu ra, bạn sẽ chiếu các biểu diễn patch đã lấy trung bình như thế nào? Hãy triển khai thay đổi này và xem nó ảnh hưởng đến độ chính xác ra sao.
1. Bạn có thể điều chỉnh các hyperparameter để cải thiện độ chính xác của vision Transformer không?

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/8943)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18032)
:end_tab: