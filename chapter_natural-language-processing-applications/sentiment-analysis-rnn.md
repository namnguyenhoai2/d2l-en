# Phân tích cảm xúc: Sử dụng mạng nơ-ron hồi quy
:label:`sec_sentiment_rnn` 


Tương tự như các tác vụ về độ tương đồng và phép loại suy từ,
chúng ta cũng có thể áp dụng các vector từ được huấn luyện trước
cho việc phân tích cảm xúc.
Vì tập dữ liệu đánh giá IMDb
trong :numref:`sec_sentiment`
không quá lớn,
việc sử dụng các biểu diễn văn bản
được huấn luyện trước
trên các kho ngữ liệu quy mô lớn
có thể làm giảm hiện tượng overfitting của mô hình.
Cụ thể như được minh họa
trong :numref:`fig_nlp-map-sa-rnn`,
chúng ta sẽ biểu diễn mỗi token
bằng mô hình GloVe được huấn luyện trước,
và đưa các biểu diễn token này
vào một RNN hai chiều nhiều lớp
để thu được biểu diễn của chuỗi văn bản,
biểu diễn này sẽ
được chuyển đổi thành
các đầu ra phân tích cảm xúc :cite:`Maas.Daly.Pham.ea.2011`.
Với cùng một ứng dụng downstream,
sau đây chúng ta sẽ xem xét
một lựa chọn kiến trúc khác.

![Phần này đưa GloVe được huấn luyện trước vào một kiến trúc dựa trên RNN để phân tích cảm xúc.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
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

## Biểu diễn một văn bản đơn bằng RNN

Trong các tác vụ phân loại văn bản,
chẳng hạn như phân tích cảm xúc,
một chuỗi văn bản có độ dài thay đổi
sẽ được chuyển đổi thành các nhãn có độ dài cố định.
Trong lớp `BiRNN` sau đây,
mỗi token của một chuỗi văn bản
nhận được biểu diễn GloVe
được huấn luyện trước của riêng nó
thông qua lớp embedding
(`self.embedding`),
còn toàn bộ chuỗi
được mã hóa bởi một RNN hai chiều (`self.encoder`).
Cụ thể hơn,
các trạng thái ẩn (ở lớp cuối)
của LSTM hai chiều
tại cả bước thời gian đầu tiên và cuối cùng
được nối với nhau
để làm biểu diễn của chuỗi văn bản.
Biểu diễn văn bản đơn này
sau đó được chuyển đổi thành các nhãn đầu ra
bởi một lớp fully connected (`self.decoder`)
với hai đầu ra ("positive" và "negative").

```{.python .input}
#@tab mxnet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) 
        outs = self.decoder(encoding)
        return outs
```

Hãy xây dựng một RNN hai chiều với hai lớp ẩn để biểu diễn một văn bản đơn cho việc phân tích cảm xúc.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
#@tab mxnet
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## Tải các vector từ được huấn luyện trước

Bên dưới, chúng ta tải các embedding GloVe 100 chiều được huấn luyện trước (cần nhất quán với `embed_size`) cho các token trong vocabulary.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

In hình dạng của các vector
cho tất cả token trong vocabulary.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

Chúng ta sử dụng các vector từ
được huấn luyện trước này
để biểu diễn các token trong các bài đánh giá
và sẽ không cập nhật
các vector này trong quá trình huấn luyện.

```{.python .input}
#@tab mxnet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## Huấn luyện và đánh giá mô hình

Bây giờ chúng ta có thể huấn luyện RNN hai chiều để phân tích cảm xúc.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Chúng ta định nghĩa hàm sau để dự đoán cảm xúc của một chuỗi văn bản bằng mô hình đã huấn luyện `net`.

```{.python .input}
#@tab mxnet
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Cuối cùng, hãy sử dụng mô hình đã huấn luyện để dự đoán cảm xúc cho hai câu đơn giản.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Tóm tắt

* Các vector từ được huấn luyện trước có thể biểu diễn từng token riêng lẻ trong một chuỗi văn bản.
* RNN hai chiều có thể biểu diễn một chuỗi văn bản, chẳng hạn bằng cách nối các trạng thái ẩn của chuỗi tại bước thời gian đầu tiên và cuối cùng. Biểu diễn văn bản đơn này có thể được chuyển đổi thành các nhãn bằng một lớp fully connected.



## Bài tập

1. Tăng số epoch. Bạn có thể cải thiện độ chính xác trên tập huấn luyện và tập kiểm thử không? Còn việc tinh chỉnh các hyperparameter khác thì sao?
1. Sử dụng các vector từ được huấn luyện trước lớn hơn, chẳng hạn embedding GloVe 300 chiều. Điều này có cải thiện độ chính xác phân loại không?
1. Chúng ta có thể cải thiện độ chính xác phân loại bằng cách sử dụng tokenization của spaCy không? Bạn cần cài đặt spaCy (`pip install spacy`) và cài đặt gói tiếng Anh (`python -m spacy download en`). Trong code, trước tiên hãy import spaCy (`import spacy`). Sau đó, tải gói tiếng Anh của spaCy (`spacy_en = spacy.load('en')`). Cuối cùng, định nghĩa hàm `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` và thay thế hàm `tokenizer` ban đầu. Lưu ý các dạng khác nhau của token cụm từ trong GloVe và spaCy. Ví dụ, token cụm từ "new york" có dạng "new-york" trong GloVe và có dạng "new york" sau khi tokenization bằng spaCy.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1424)
:end_tab: