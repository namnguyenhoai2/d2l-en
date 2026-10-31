# Phân tích cảm xúc: Sử dụng mạng nơ-ron tích chập
:label:`sec_sentiment_cnn` 


Trong :numref:`chap_cnn`,
chúng ta đã tìm hiểu các cơ chế
để xử lý
dữ liệu hình ảnh hai chiều
bằng CNN hai chiều,
được áp dụng cho
các đặc trưng cục bộ như những pixel liền kề.
Mặc dù ban đầu
được thiết kế cho thị giác máy tính,
CNN cũng được sử dụng rộng rãi
trong xử lý ngôn ngữ tự nhiên.
Nói một cách đơn giản,
hãy xem mọi chuỗi văn bản
như một hình ảnh một chiều.
Theo cách này,
CNN một chiều
có thể xử lý các đặc trưng cục bộ
như các $n$-gram trong văn bản.

Trong phần này,
chúng ta sẽ sử dụng mô hình *textCNN*
để minh họa
cách thiết kế một kiến trúc CNN
nhằm biểu diễn một văn bản đơn :cite:`Kim.2014`.
So với
:numref:`fig_nlp-map-sa-rnn`
sử dụng kiến trúc RNN cùng với việc pretrain GloVe
cho bài toán phân tích cảm xúc,
khác biệt duy nhất trong :numref:`fig_nlp-map-sa-cnn`
nằm ở
việc lựa chọn kiến trúc.


![Phần này đưa GloVe đã được pretrain vào một kiến trúc dựa trên CNN để phân tích cảm xúc.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## Tích chập một chiều

Trước khi giới thiệu mô hình,
hãy cùng xem tích chập một chiều hoạt động như thế nào.
Hãy nhớ rằng đây chỉ là một trường hợp đặc biệt
của tích chập hai chiều
dựa trên phép tương quan chéo.

![Phép tương quan chéo một chiều. Các phần được tô đậm là phần tử đầu tiên của đầu ra, cũng như các phần tử tensor đầu vào và tensor kernel được sử dụng để tính đầu ra: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

Như được trình bày trong :numref:`fig_conv1d`,
trong trường hợp một chiều,
cửa sổ tích chập
trượt từ trái sang phải
trên tensor đầu vào.
Trong quá trình trượt,
tensor con đầu vào (ví dụ, $0$ và $1$ trong :numref:`fig_conv1d`) nằm trong cửa sổ tích chập
tại một vị trí nhất định
và tensor kernel (ví dụ, $1$ và $2$ trong :numref:`fig_conv1d`) được nhân theo từng phần tử.
Tổng của các phép nhân này
cho ra một giá trị vô hướng duy nhất (ví dụ, $0\times1+1\times2=2$ trong :numref:`fig_conv1d`)
tại vị trí tương ứng của tensor đầu ra.

Chúng ta triển khai phép tương quan chéo một chiều trong hàm `corr1d` sau đây.
Với tensor đầu vào `X`
và tensor kernel `K`,
hàm trả về tensor đầu ra `Y`.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

Chúng ta có thể tạo tensor đầu vào `X` và tensor kernel `K` từ :numref:`fig_conv1d` để kiểm tra đầu ra của phép triển khai tương quan chéo một chiều ở trên.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

Đối với mọi
đầu vào một chiều có nhiều kênh,
kernel tích chập
cần có cùng số lượng kênh đầu vào.
Sau đó, với mỗi kênh,
thực hiện phép tương quan chéo trên tensor một chiều của đầu vào và tensor một chiều của kernel tích chập,
cộng các kết quả trên tất cả các kênh
để tạo ra tensor đầu ra một chiều.
:numref:`fig_conv1d_channel` minh họa phép tương quan chéo một chiều với 3 kênh đầu vào.

![Phép tương quan chéo một chiều với 3 kênh đầu vào. Các phần được tô đậm là phần tử đầu tiên của đầu ra, cũng như các phần tử tensor đầu vào và tensor kernel được sử dụng để tính đầu ra: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`


Chúng ta có thể triển khai phép tương quan chéo một chiều cho nhiều kênh đầu vào
và kiểm tra các kết quả trong :numref:`fig_conv1d_channel`.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

Lưu ý rằng
các phép tương quan chéo một chiều với nhiều kênh đầu vào
tương đương
với
các phép tương quan chéo hai chiều
với một kênh đầu vào.
Để minh họa,
dạng tương đương của
phép tương quan chéo một chiều với nhiều kênh đầu vào
trong :numref:`fig_conv1d_channel`
là
phép tương quan chéo hai chiều
với một kênh đầu vào
trong :numref:`fig_conv1d_2d`,
trong đó chiều cao của kernel tích chập
phải giống với chiều cao của tensor đầu vào.


![Phép tương quan chéo hai chiều với một kênh đầu vào. Các phần được tô đậm là phần tử đầu tiên của đầu ra, cũng như các phần tử tensor đầu vào và tensor kernel được sử dụng để tính đầu ra: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

Cả hai đầu ra trong :numref:`fig_conv1d` và :numref:`fig_conv1d_channel` chỉ có một kênh.
Tương tự như các phép tích chập hai chiều với nhiều kênh đầu ra được mô tả trong :numref:`subsec_multi-output-channels`,
chúng ta cũng có thể chỉ định nhiều kênh đầu ra
cho các phép tích chập một chiều.

## Pooling cực đại theo thời gian

Tương tự, chúng ta có thể sử dụng pooling
để trích xuất giá trị cao nhất
từ các biểu diễn chuỗi
như đặc trưng quan trọng nhất
trải dài qua các bước thời gian.
*Pooling cực đại theo thời gian*
được sử dụng trong textCNN
hoạt động giống như
global max-pooling một chiều
:cite:`Collobert.Weston.Bottou.ea.2011`.
Đối với đầu vào nhiều kênh,
trong đó mỗi kênh lưu trữ các giá trị
tại những bước thời gian khác nhau,
đầu ra của mỗi kênh
là giá trị lớn nhất
của kênh đó.
Lưu ý rằng
pooling cực đại theo thời gian
cho phép các kênh khác nhau
có số bước thời gian khác nhau.

## Mô hình textCNN

Bằng cách sử dụng tích chập một chiều
và pooling cực đại theo thời gian,
mô hình textCNN
nhận các biểu diễn token đã được pretrain riêng lẻ
làm đầu vào,
sau đó thu được và biến đổi các biểu diễn chuỗi
cho ứng dụng hạ nguồn.

Đối với một chuỗi văn bản đơn
có $n$ token được biểu diễn bằng
các vector $d$ chiều,
chiều rộng, chiều cao và số lượng kênh
của tensor đầu vào
lần lượt là $n$, $1$ và $d$.
Mô hình textCNN biến đổi đầu vào
thành đầu ra như sau:

1. Xác định nhiều kernel tích chập một chiều và thực hiện các phép tích chập riêng biệt trên các đầu vào. Các kernel tích chập có chiều rộng khác nhau có thể nắm bắt các đặc trưng cục bộ giữa số lượng token liền kề khác nhau.
1. Thực hiện pooling cực đại theo thời gian trên tất cả các kênh đầu ra, sau đó nối tất cả các đầu ra pooling vô hướng thành một vector.
1. Biến đổi vector đã nối thành các lớp đầu ra bằng fully connected layer. Có thể sử dụng Dropout để giảm overfitting.

![Kiến trúc mô hình của textCNN.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn`
minh họa kiến trúc mô hình textCNN
bằng một ví dụ cụ thể.
Đầu vào là một câu có 11 token,
trong đó
mỗi token được biểu diễn bằng một vector 6 chiều.
Vì vậy, chúng ta có một đầu vào 6 kênh với chiều rộng 11.
Xác định
hai kernel tích chập một chiều
có chiều rộng 2 và 4,
với lần lượt 4 và 5 kênh đầu ra.
Chúng tạo ra
4 kênh đầu ra có chiều rộng $11-2+1=10$
và 5 kênh đầu ra có chiều rộng $11-4+1=8$.
Mặc dù 9 kênh này có các chiều rộng khác nhau,
pooling cực đại theo thời gian
tạo ra một vector 9 chiều được nối,
cuối cùng được biến đổi
thành một vector đầu ra 2 chiều
để dự đoán cảm xúc nhị phân.



### Định nghĩa mô hình

Chúng ta triển khai mô hình textCNN trong lớp sau đây.
So với mô hình RNN hai chiều trong
:numref:`sec_sentiment_rnn`,
ngoài việc
thay thế các recurrent layer bằng các convolutional layer,
chúng ta cũng sử dụng hai embedding layer:
một lớp có trọng số có thể huấn luyện và lớp còn lại
có trọng số cố định.

```{.python .input}
#@tab mxnet
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

Hãy tạo một thực thể textCNN.
Mô hình có 3 convolutional layer với chiều rộng kernel lần lượt là 3, 4 và 5, tất cả đều có 100 kênh đầu ra.

```{.python .input}
#@tab mxnet
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)

net.apply(init_weights);
```

### Tải các vector từ đã được pretrain

Tương tự như :numref:`sec_sentiment_rnn`,
chúng ta tải các embedding GloVe 100 chiều đã được pretrain
làm các biểu diễn token được khởi tạo.
Các biểu diễn token này (embedding weight)
sẽ được huấn luyện trong `embedding`
và được cố định trong `constant_embedding`.

```{.python .input}
#@tab mxnet
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### Huấn luyện và đánh giá mô hình

Bây giờ chúng ta có thể huấn luyện mô hình textCNN để phân tích cảm xúc.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Dưới đây, chúng ta sử dụng mô hình đã huấn luyện để dự đoán cảm xúc cho hai câu đơn giản.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## Tóm tắt

* CNN một chiều có thể xử lý các đặc trưng cục bộ như $n$-gram trong văn bản.
* Các phép tương quan chéo một chiều với nhiều kênh đầu vào tương đương với các phép tương quan chéo hai chiều với một kênh đầu vào.
* Pooling cực đại theo thời gian cho phép các kênh khác nhau có số bước thời gian khác nhau.
* Mô hình textCNN biến đổi các biểu diễn token riêng lẻ thành đầu ra cho ứng dụng hạ nguồn bằng cách sử dụng các convolutional layer một chiều và các pooling layer cực đại theo thời gian.


## Bài tập

1. Tinh chỉnh các hyperparameter và so sánh hai kiến trúc dùng cho phân tích cảm xúc trong :numref:`sec_sentiment_rnn` và trong phần này, chẳng hạn như về độ chính xác phân loại và hiệu quả tính toán.
1. Bạn có thể cải thiện thêm độ chính xác phân loại của mô hình bằng cách sử dụng các phương pháp được giới thiệu trong phần bài tập của :numref:`sec_sentiment_rnn` không?
1. Thêm positional encoding vào các biểu diễn đầu vào. Điều này có cải thiện độ chính xác phân loại không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1425)
:end_tab: