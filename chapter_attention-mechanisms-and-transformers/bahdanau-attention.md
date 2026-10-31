```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Cơ chế Attention Bahdanau
:label:`sec_seq2seq_attention`

Khi tìm hiểu về dịch máy trong :numref:`sec_seq2seq`,
chúng ta đã thiết kế kiến trúc encoder--decoder cho bài toán học sequence-to-sequence
dựa trên hai RNN :cite:`Sutskever.Vinyals.Le.2014`.
Cụ thể, RNN encoder biến đổi một chuỗi có độ dài thay đổi
thành một biến ngữ cảnh có *hình dạng cố định*.
Sau đó, RNN decoder sinh chuỗi đầu ra (đích) từng token một
dựa trên các token đã sinh và biến ngữ cảnh.

Hãy nhắc lại :numref:`fig_seq2seq_details`, mà chúng ta lặp lại (:numref:`fig_s2s_attention_state`) với một số chi tiết bổ sung. Theo quy ước, trong một RNN, mọi thông tin liên quan về chuỗi nguồn được encoder chuyển đổi thành một biểu diễn trạng thái nội bộ có *số chiều cố định*. Chính trạng thái này được decoder sử dụng làm nguồn thông tin đầy đủ và duy nhất để sinh chuỗi đã dịch. Nói cách khác, cơ chế sequence-to-sequence coi trạng thái trung gian là một thống kê đủ của bất kỳ chuỗi nào có thể đã được dùng làm đầu vào.

![Mô hình sequence-to-sequence. Trạng thái do encoder sinh ra là mẩu thông tin duy nhất được chia sẻ giữa encoder và decoder.](../img/seq2seq-state.svg)
:label:`fig_s2s_attention_state`

Dù điều này khá hợp lý đối với các chuỗi ngắn, rõ ràng nó không khả thi đối với các chuỗi dài, chẳng hạn như một chương sách hoặc thậm chí chỉ một câu rất dài. Xét cho cùng, chẳng bao lâu sau sẽ không còn đủ "không gian" trong biểu diễn trung gian để lưu trữ mọi thông tin quan trọng của chuỗi nguồn. Do đó, decoder sẽ không thể dịch các câu dài và phức tạp. Một trong những người đầu tiên gặp phải vấn đề này là :citet:`Graves.2013`, người đã cố gắng thiết kế một RNN để sinh văn bản viết tay. Vì văn bản nguồn có độ dài tùy ý, họ đã thiết kế một mô hình attention khả vi
để căn chỉnh các ký tự văn bản với nét bút dài hơn nhiều,
trong đó việc căn chỉnh chỉ di chuyển theo một hướng. Điều này, đến lượt nó, dựa trên các thuật toán giải mã trong nhận dạng giọng nói, ví dụ như mô hình Markov ẩn :cite:`rabiner1993fundamentals`.

Lấy cảm hứng từ ý tưởng học cách căn chỉnh,
:citet:`Bahdanau.Cho.Bengio.2014` đã đề xuất một mô hình attention khả vi
*không có* giới hạn căn chỉnh một chiều.
Khi dự đoán một token,
nếu không phải tất cả token đầu vào đều liên quan,
mô hình chỉ căn chỉnh (hoặc chú ý)
vào các phần của chuỗi đầu vào
được xem là liên quan đến dự đoán hiện tại. Sau đó, thông tin này được dùng để cập nhật trạng thái hiện tại trước khi sinh token tiếp theo. Dù mô tả của nó khá đơn giản, *cơ chế attention Bahdanau* này được cho là đã trở thành một trong những ý tưởng có ảnh hưởng nhất trong thập kỷ qua của deep learning, mở đường cho Transformers :cite:`Vaswani.Shazeer.Parmar.ea.2017` và nhiều kiến trúc mới liên quan.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import rnn, nn
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
from jax import numpy as jnp
import jax
```

## Mô hình

Chúng ta sử dụng ký hiệu được giới thiệu bởi kiến trúc sequence-to-sequence trong :numref:`sec_seq2seq`, đặc biệt là :eqref:`eq_seq2seq_s_t`.
Ý tưởng cốt lõi là thay vì giữ trạng thái,
tức biến ngữ cảnh $\mathbf{c}$ tóm tắt câu nguồn, cố định, chúng ta cập nhật nó một cách động, như một hàm của cả văn bản gốc (các trạng thái ẩn của encoder $\mathbf{h}_{t}$) lẫn văn bản đã được sinh (các trạng thái ẩn của decoder $\mathbf{s}_{t'-1}$). Điều này tạo ra $\mathbf{c}_{t'}$, được cập nhật sau mỗi bước thời gian giải mã $t'$. Giả sử chuỗi đầu vào có độ dài $T$. Trong trường hợp này, biến ngữ cảnh là đầu ra của attention pooling:

$$\mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}.$$

Chúng ta sử dụng $\mathbf{s}_{t' - 1}$ làm query, và
$\mathbf{h}_{t}$ làm cả key lẫn value. Lưu ý rằng $\mathbf{c}_{t'}$ sau đó được dùng để sinh trạng thái $\mathbf{s}_{t'}$ và sinh một token mới: xem :eqref:`eq_seq2seq_s_t`. Cụ thể, trọng số attention $\alpha$ được tính như trong :eqref:`eq_attn-scoring-alpha`
bằng cách sử dụng hàm chấm điểm additive attention
được định nghĩa bởi :eqref:`eq_additive-attn`.
Kiến trúc RNN encoder--decoder
có sử dụng attention này được minh họa trong :numref:`fig_s2s_attention_details`. Lưu ý rằng về sau mô hình này đã được sửa đổi để đưa các token đã sinh trong decoder vào làm ngữ cảnh bổ sung (tức là tổng attention không dừng ở $T$ mà tiếp tục đến $t'-1$). Chẳng hạn, xem :citet:`chan2015listen` để biết mô tả về chiến lược này khi áp dụng cho nhận dạng giọng nói.

![Các lớp trong mô hình RNN encoder--decoder với cơ chế attention Bahdanau.](../img/seq2seq-details-attention.svg)
:label:`fig_s2s_attention_details`

## Định nghĩa Decoder với Attention

Để triển khai RNN encoder--decoder với attention,
chúng ta chỉ cần định nghĩa lại decoder (việc bỏ qua các ký hiệu đã sinh khỏi hàm attention giúp đơn giản hóa thiết kế). Hãy bắt đầu với [**giao diện cơ sở cho decoder có attention**] bằng cách định nghĩa lớp có tên khá dễ đoán là `AttentionDecoder`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class AttentionDecoder(d2l.Decoder):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

Chúng ta cần [**triển khai RNN decoder**]
trong lớp `Seq2SeqAttentionDecoder`.
Trạng thái của decoder được khởi tạo bằng
(i) các trạng thái ẩn của lớp cuối cùng của encoder tại mọi bước thời gian, được dùng làm key và value cho attention;
(ii) trạng thái ẩn của encoder ở mọi lớp tại bước thời gian cuối cùng, dùng để khởi tạo trạng thái ẩn của decoder;
và (iii) độ dài hợp lệ của encoder, để loại trừ các token padding trong attention pooling.
Tại mỗi bước thời gian giải mã, trạng thái ẩn của lớp cuối cùng của decoder, thu được ở bước thời gian trước đó, được dùng làm query của cơ chế attention.
Cả đầu ra của cơ chế attention và embedding đầu vào được nối lại để làm đầu vào cho RNN decoder.

```{.python .input}
%%tab mxnet
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = np.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            hidden_state = hidden_state[0]
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (batch_size, num_steps, num_hiddens).
        # Length of list hidden_state is num_layers, where the shape of its
        # element is (batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (tf.transpose(outputs, (1, 0, 2)), hidden_state,
                enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Shape of output enc_outputs: # (batch_size, num_steps, num_hiddens)
        # Length of list hidden_state is num_layers, where the shape of its
        # element is (batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X)  # Input X has shape: (batch_size, num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Concatenate on the feature dimension
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (batch_size, num_steps, vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class Seq2SeqAttentionDecoder(nn.Module):
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.attention = d2l.AdditiveAttention(self.num_hiddens, self.dropout)
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.dense = nn.Dense(self.vocab_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout=self.dropout)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        # Attention Weights are returned as part of state; init with None
        return (outputs.transpose(1, 0, 2), hidden_state, enc_valid_lens)

    @nn.compact
    def __call__(self, X, state, training=False):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        # Ignore Attention value in state
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).transpose(1, 0, 2)
        outputs, attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = jnp.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context, attention_w = self.attention(query, enc_outputs,
                                                  enc_outputs, enc_valid_lens,
                                                  training=training)
            # Concatenate on the feature dimension
            x = jnp.concatenate((context, jnp.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.transpose(1, 0, 2), hidden_state,
                                         training=training)
            outputs.append(out)
            attention_weights.append(attention_w)

        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'dec_attention_weights', attention_weights)

        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(jnp.concatenate(outputs, axis=0))
        return outputs.transpose(1, 0, 2), [enc_outputs, hidden_state,
                                            enc_valid_lens]
```

Sau đây, chúng ta [**kiểm tra decoder đã triển khai**] với attention
bằng một minibatch gồm bốn chuỗi, mỗi chuỗi dài bảy bước thời gian.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
if tab.selected('mxnet'):
    X = d2l.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('pytorch'):
    X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('tensorflow'):
    X = tf.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X, training=False), None)
    output, state = decoder(X, state, training=False)
if tab.selected('jax'):
    X = jnp.zeros((batch_size, num_steps), dtype=jnp.int32)
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(),
                                                        X, training=False)[0],
                               None)
    (output, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                  state, training=False)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

## [**Huấn luyện**]

Giờ đây khi đã xác định decoder mới, chúng ta có thể thực hiện tương tự như :numref:`sec_seq2seq_training`:
xác định các hyperparameter, khởi tạo
một encoder thông thường và một decoder có attention,
rồi huấn luyện mô hình này cho dịch máy.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = d2l.Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = d2l.Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

Sau khi mô hình được huấn luyện,
chúng ta dùng nó để [**dịch một vài câu tiếng Anh**]
sang tiếng Pháp và tính điểm BLEU của chúng.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(
        trainer.state.params, data.build(engs, fras), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

Hãy [**trực quan hóa các trọng số attention**]
khi dịch câu tiếng Anh cuối cùng.
Chúng ta thấy rằng mỗi query gán các trọng số không đồng đều
trên các cặp key--value.
Điều này cho thấy rằng tại mỗi bước giải mã,
các phần khác nhau của chuỗi đầu vào
được tổng hợp có chọn lọc trong attention pooling.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
if tab.selected('jax'):
    _, (dec_attention_weights, _) = model.predict_step(
        trainer.state.params, data.build([engs[-1]], [fras[-1]]),
        data.num_steps, True)
attention_weights = d2l.concat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = d2l.reshape(attention_weights, (1, 1, -1, data.num_steps))
```

```{.python .input}
%%tab mxnet
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab tensorflow
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab jax
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

## Tóm tắt

Khi dự đoán một token, nếu không phải tất cả token đầu vào đều liên quan, RNN encoder--decoder với cơ chế attention Bahdanau sẽ tổng hợp có chọn lọc các phần khác nhau của chuỗi đầu vào. Điều này đạt được bằng cách coi trạng thái (biến ngữ cảnh) là đầu ra của additive attention pooling.
Trong RNN encoder--decoder, cơ chế attention Bahdanau coi trạng thái ẩn của decoder ở bước thời gian trước làm query, và các trạng thái ẩn của encoder ở mọi bước thời gian làm cả key lẫn value.


## Bài tập

1. Thay GRU bằng LSTM trong thí nghiệm.
1. Sửa đổi thí nghiệm để thay hàm chấm điểm additive attention bằng scaled dot-product. Điều này ảnh hưởng thế nào đến hiệu quả huấn luyện?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1065)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3868)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18028)
:end_tab: