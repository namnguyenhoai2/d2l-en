# Biểu diễn bộ mã hóa hai chiều từ các Transformer (BERT)
:label:`sec_bert`

Chúng ta đã giới thiệu một số mô hình embedding từ cho việc hiểu ngôn ngữ tự nhiên.
Sau khi pretrain, đầu ra có thể được xem như một ma trận,
trong đó mỗi hàng là một vector biểu diễn một từ trong một vocabulary được xác định trước.
Trên thực tế, tất cả các mô hình embedding từ này đều *không phụ thuộc ngữ cảnh*.
Hãy bắt đầu bằng cách minh họa đặc tính này.


## Từ không phụ thuộc ngữ cảnh đến nhạy ngữ cảnh

Hãy nhớ lại các thí nghiệm trong :numref:`sec_word2vec_pretraining` và :numref:`sec_synonyms`.
Chẳng hạn, cả word2vec và GloVe đều gán cùng một vector đã pretrain cho cùng một từ, bất kể ngữ cảnh của từ đó (nếu có).
Về mặt hình thức, biểu diễn không phụ thuộc ngữ cảnh của bất kỳ token $x$ nào
là một hàm $f(x)$ chỉ nhận $x$ làm đầu vào.
Do ngôn ngữ tự nhiên có nhiều từ đa nghĩa và ngữ nghĩa phức tạp,
các biểu diễn không phụ thuộc ngữ cảnh có những hạn chế rõ ràng.
Ví dụ, từ "crane" trong các ngữ cảnh
"a crane is flying" và "a crane driver came" có ý nghĩa hoàn toàn khác nhau;
do đó, cùng một từ có thể được gán các biểu diễn khác nhau tùy thuộc vào ngữ cảnh.

Điều này thúc đẩy sự phát triển của các biểu diễn từ *nhạy ngữ cảnh*,
trong đó biểu diễn của từ phụ thuộc vào ngữ cảnh của chúng.
Vì vậy, biểu diễn nhạy ngữ cảnh của token $x$ là một hàm $f(x, c(x))$
phụ thuộc vào cả $x$ và ngữ cảnh $c(x)$ của nó.
Các biểu diễn nhạy ngữ cảnh phổ biến
bao gồm TagLM (language-model-augmented sequence tagger) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe (Context Vectors) :cite:`McCann.Bradbury.Xiong.ea.2017`,
và ELMo (Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`.

Ví dụ, bằng cách nhận toàn bộ sequence làm đầu vào,
ELMo là một hàm gán một biểu diễn cho mỗi từ trong sequence đầu vào.
Cụ thể, ELMo kết hợp tất cả biểu diễn của các layer trung gian từ bidirectional LSTM đã pretrain làm biểu diễn đầu ra.
Sau đó, biểu diễn ELMo sẽ được thêm vào model supervised hiện có của một downstream task
dưới dạng các feature bổ sung, chẳng hạn bằng cách nối biểu diễn ELMo với biểu diễn ban đầu (ví dụ: GloVe) của các token trong model hiện có.
Một mặt,
tất cả trọng số trong model bidirectional LSTM đã pretrain đều được đóng băng sau khi thêm các biểu diễn ELMo.
Mặt khác,
model supervised hiện có được tùy chỉnh riêng cho một task nhất định.
Bằng cách tận dụng các model tốt nhất khác nhau cho từng task vào thời điểm đó,
việc thêm ELMo đã cải thiện state of the art trên sáu task xử lý ngôn ngữ tự nhiên:
phân tích cảm xúc, suy luận ngôn ngữ tự nhiên,
gán nhãn vai nghĩa, phân giải đồng tham chiếu,
nhận dạng thực thể có tên và trả lời câu hỏi.


## Từ phụ thuộc task đến không phụ thuộc task

Mặc dù ELMo đã cải thiện đáng kể các giải pháp cho nhiều task xử lý ngôn ngữ tự nhiên,
mỗi giải pháp vẫn phụ thuộc vào một kiến trúc *đặc thù cho task*.
Tuy nhiên, trên thực tế, việc thiết kế một kiến trúc cụ thể cho từng task xử lý ngôn ngữ tự nhiên không hề đơn giản.
Model GPT (Generative Pre-Training) là một nỗ lực nhằm thiết kế
một model tổng quát *không phụ thuộc task* cho các biểu diễn nhạy ngữ cảnh :cite:`Radford.Narasimhan.Salimans.ea.2018`.
Được xây dựng trên Transformer decoder,
GPT pretrain một language model sẽ được dùng để biểu diễn các text sequence.
Khi áp dụng GPT cho một downstream task,
đầu ra của language model sẽ được đưa vào một linear output layer được thêm vào
để dự đoán label của task.
Trái ngược rõ rệt với ELMo, vốn đóng băng các parameter của model đã pretrain,
GPT fine-tune *tất cả* parameter trong Transformer decoder đã pretrain
trong quá trình supervised learning của downstream task.
GPT được đánh giá trên mười hai task về suy luận ngôn ngữ tự nhiên,
trả lời câu hỏi, độ tương đồng câu và phân loại,
đồng thời cải thiện state of the art ở chín task trong số đó với những thay đổi tối thiểu
đối với kiến trúc model.

Tuy nhiên, do bản chất autoregressive của language model,
GPT chỉ nhìn về phía trước (từ trái sang phải).
Trong các ngữ cảnh "i went to the bank to deposit cash" và "i went to the bank to sit down",
vì "bank" nhạy với ngữ cảnh ở bên trái,
GPT sẽ trả về cùng một biểu diễn cho "bank",
mặc dù chúng có ý nghĩa khác nhau.


## BERT: Kết hợp ưu điểm của cả hai

Như chúng ta đã thấy,
ELMo mã hóa ngữ cảnh theo hai chiều nhưng sử dụng các kiến trúc đặc thù cho task;
trong khi GPT không phụ thuộc task nhưng mã hóa ngữ cảnh từ trái sang phải.
Kết hợp ưu điểm của cả hai,
BERT (Bidirectional Encoder Representations from Transformers)
mã hóa ngữ cảnh theo hai chiều và yêu cầu thay đổi tối thiểu về kiến trúc
cho nhiều task xử lý ngôn ngữ tự nhiên :cite:`Devlin.Chang.Lee.ea.2018`.
Sử dụng một Transformer encoder đã pretrain,
BERT có thể biểu diễn bất kỳ token nào dựa trên ngữ cảnh hai chiều của token đó.
Trong quá trình supervised learning của các downstream task,
BERT tương tự GPT ở hai khía cạnh.
Thứ nhất, các biểu diễn BERT sẽ được đưa vào một output layer được thêm vào,
với những thay đổi tối thiểu đối với kiến trúc model tùy theo bản chất của task,
chẳng hạn như dự đoán cho từng token so với dự đoán cho toàn bộ sequence.
Thứ hai,
tất cả parameter của Transformer encoder đã pretrain đều được fine-tune,
trong khi output layer bổ sung sẽ được train từ đầu.
:numref:`fig_elmo-gpt-bert` mô tả sự khác biệt giữa ELMo, GPT và BERT.

![So sánh ELMo, GPT và BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`


BERT tiếp tục cải thiện state of the art trên mười một task xử lý ngôn ngữ tự nhiên
thuộc các nhóm lớn: (i) phân loại một văn bản (ví dụ: phân tích cảm xúc), (ii) phân loại cặp văn bản (ví dụ: suy luận ngôn ngữ tự nhiên),
(iii) trả lời câu hỏi, (iv) gắn thẻ văn bản (ví dụ: nhận dạng thực thể có tên).
Tất cả đều được đề xuất vào năm 2018,
từ ELMo nhạy ngữ cảnh đến GPT và BERT không phụ thuộc task,
việc pretrain các biểu diễn sâu cho ngôn ngữ tự nhiên, tuy đơn giản về mặt khái niệm nhưng mạnh mẽ về mặt thực nghiệm,
đã cách mạng hóa các giải pháp cho nhiều task xử lý ngôn ngữ tự nhiên.

Trong phần còn lại của chương này,
chúng ta sẽ đi sâu vào quá trình pretrain BERT.
Khi các ứng dụng xử lý ngôn ngữ tự nhiên được giải thích trong :numref:`chap_nlp_app`,
chúng ta sẽ minh họa cách fine-tune BERT cho các ứng dụng downstream.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Biểu diễn đầu vào**]
:label:`subsec_bert_input_rep`

Trong xử lý ngôn ngữ tự nhiên,
một số task (ví dụ: phân tích cảm xúc) nhận một văn bản đơn làm đầu vào,
trong khi một số task khác (ví dụ: suy luận ngôn ngữ tự nhiên)
nhận một cặp text sequence làm đầu vào.
BERT input sequence biểu diễn rõ ràng cả văn bản đơn lẫn cặp văn bản.
Trong trường hợp đầu tiên,
BERT input sequence là phép nối của
classification token đặc biệt “&lt;cls&gt;”,
các token của một text sequence
và separation token đặc biệt “&lt;sep&gt;”.
Trong trường hợp thứ hai,
BERT input sequence là phép nối của
“&lt;cls&gt;”, các token của text sequence thứ nhất,
“&lt;sep&gt;”, các token của text sequence thứ hai và “&lt;sep&gt;”.
Chúng ta sẽ nhất quán phân biệt thuật ngữ "BERT input sequence"
với các loại "sequence" khác.
Ví dụ, một *BERT input sequence* có thể bao gồm một *text sequence* hoặc hai *text sequence*.

Để phân biệt các cặp văn bản,
các segment embedding được học $\mathbf{e}_A$ và $\mathbf{e}_B$
được thêm lần lượt vào token embedding của sequence thứ nhất và sequence thứ hai.
Đối với đầu vào là văn bản đơn, chỉ $\mathbf{e}_A$ được sử dụng.

`get_tokens_and_segments` sau đây nhận một hoặc hai câu
làm đầu vào, sau đó trả về các token của BERT input sequence
và các segment ID tương ứng.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT chọn Transformer encoder làm kiến trúc hai chiều của mình.
Tương tự Transformer encoder,
positional embedding được thêm vào mọi vị trí của BERT input sequence.
Tuy nhiên, khác với Transformer encoder ban đầu,
BERT sử dụng positional embedding *có thể học*.
Tóm lại, :numref:`fig_bert-input` cho thấy
embedding của BERT input sequence là tổng
của token embedding, segment embedding và positional embedding.

![Embedding của BERT input sequence là tổng
của token embedding, segment embedding và positional embedding.](../img/bert-input.svg)
:label:`fig_bert-input`

[**class `BERTEncoder`**] sau đây tương tự class `TransformerEncoder`
được triển khai trong :numref:`sec_transformer`.
Khác với `TransformerEncoder`, `BERTEncoder` sử dụng
segment embedding và positional embedding có thể học.

```{.python .input}
#@tab mxnet
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

Giả sử kích thước vocabulary là 10000.
Để minh họa [**suy luận forward của `BERTEncoder`**],
hãy tạo một instance của nó và khởi tạo các parameter của instance.

```{.python .input}
#@tab mxnet
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_blks, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
ffn_num_input, num_blks, dropout = 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
```

Chúng ta định nghĩa `tokens` là 2 BERT input sequence có độ dài 8,
trong đó mỗi token là một index trong vocabulary.
Suy luận forward của `BERTEncoder` với đầu vào `tokens`
trả về kết quả đã encode, trong đó mỗi token được biểu diễn bằng một vector
có độ dài được xác định trước bởi hyperparameter `num_hiddens`.
Hyperparameter này thường được gọi là *hidden size*
(số hidden unit) của Transformer encoder.

```{.python .input}
#@tab mxnet
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## Các task pretrain
:label:`subsec_bert_pretraining_tasks`

Suy luận forward của `BERTEncoder` cung cấp biểu diễn BERT
của mỗi token trong văn bản đầu vào và các
token đặc biệt được chèn “&lt;cls&gt;” và “&lt;seq&gt;”.
Tiếp theo, chúng ta sẽ sử dụng các biểu diễn này để tính hàm loss
cho việc pretrain BERT.
Quá trình pretrain bao gồm hai task sau:
masked language modeling và next sentence prediction.

### [**Masked Language Modeling**]
:label:`subsec_mlm`

Như được minh họa trong :numref:`sec_language-model`,
một language model dự đoán một token bằng cách sử dụng ngữ cảnh ở bên trái.
Để mã hóa ngữ cảnh theo hai chiều nhằm biểu diễn từng token,
BERT ngẫu nhiên che các token và sử dụng các token từ ngữ cảnh hai chiều để
dự đoán những token bị che theo cách tự giám sát.
Nhiệm vụ này được gọi là một *masked language model*.

Trong tác vụ pretraining này,
15% số token sẽ được chọn ngẫu nhiên làm các token bị che để dự đoán.
Để dự đoán một token bị che mà không gian lận bằng cách sử dụng nhãn,
một cách tiếp cận đơn giản là luôn thay thế nó bằng một token đặc biệt “&lt;mask&gt;” trong chuỗi đầu vào BERT.
Tuy nhiên, token đặc biệt nhân tạo “&lt;mask&gt;” sẽ không bao giờ xuất hiện
trong quá trình fine-tuning.
Để tránh sự không tương thích giữa pretraining và fine-tuning,
nếu một token bị che để dự đoán (ví dụ: "great" được chọn để che và dự đoán trong "this movie is great"),
thì trong đầu vào, nó sẽ được thay thế bằng:

* một token đặc biệt “&lt;mask&gt;” trong 80% số lần (ví dụ: "this movie is great" trở thành "this movie is &lt;mask&gt;");
* một token ngẫu nhiên trong 10% số lần (ví dụ: "this movie is great" trở thành "this movie is drink");
* token nhãn không thay đổi trong 10% số lần (ví dụ: "this movie is great" vẫn là "this movie is great").

Lưu ý rằng trong 10% của 15% số lần, một token ngẫu nhiên được chèn vào.
Nhiễu thỉnh thoảng này khuyến khích BERT giảm thiên lệch đối với token bị che (đặc biệt khi token nhãn không thay đổi) trong quá trình mã hóa ngữ cảnh hai chiều.

Chúng ta triển khai lớp `MaskLM` sau đây để dự đoán các token bị che
trong tác vụ masked language model của quá trình pretraining BERT.
Việc dự đoán sử dụng một MLP có một hidden layer (`self.mlp`).
Trong forward inference, nó nhận hai đầu vào:
kết quả đã mã hóa của `BERTEncoder` và các vị trí token cần dự đoán.
Đầu ra là các kết quả dự đoán tại những vị trí này.

```{.python .input}
#@tab mxnet
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

Để minh họa [**forward inference của `MaskLM`**],
chúng ta tạo instance `mlm` của nó và khởi tạo instance này.
Hãy nhớ rằng `encoded_X` từ forward inference của `BERTEncoder`
biểu diễn 2 chuỗi đầu vào BERT.
Chúng ta định nghĩa `mlm_positions` là 3 chỉ số cần dự đoán trong một trong hai chuỗi đầu vào BERT của `encoded_X`.
Forward inference của `mlm` trả về các kết quả dự đoán `mlm_Y_hat`
tại tất cả các vị trí bị che `mlm_positions` của `encoded_X`.
Đối với mỗi dự đoán, kích thước của kết quả bằng với kích thước vocabulary.

```{.python .input}
#@tab mxnet
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

Với các nhãn ground truth `mlm_Y` của những token được dự đoán `mlm_Y_hat` dưới các mask,
chúng ta có thể tính cross-entropy loss của tác vụ masked language model trong quá trình pretraining BERT.

```{.python .input}
#@tab mxnet
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### [**Next Sentence Prediction**]
:label:`subsec_nsp`

Mặc dù masked language modeling có khả năng mã hóa ngữ cảnh hai chiều
để biểu diễn các từ, nó không mô hình hóa tường minh mối quan hệ logic
giữa các cặp văn bản.
Để giúp hiểu mối quan hệ giữa hai chuỗi văn bản,
BERT xem xét một tác vụ phân loại nhị phân, *next sentence prediction*, trong quá trình pretraining.
Khi tạo các cặp câu cho pretraining,
một nửa số lần chúng thực sự là các câu liên tiếp với nhãn "True";
trong nửa số lần còn lại, câu thứ hai được lấy ngẫu nhiên từ corpus với nhãn "False".

Lớp `NextSentencePred` sau đây sử dụng một MLP có một hidden layer
để dự đoán liệu câu thứ hai có phải là câu tiếp theo của câu thứ nhất
trong chuỗi đầu vào BERT hay không.
Do cơ chế self-attention trong Transformer encoder,
biểu diễn của token đặc biệt “&lt;cls&gt;” trong BERT
mã hóa cả hai câu từ đầu vào.
Vì vậy, output layer (`self.output`) của MLP classifier nhận `X` làm đầu vào,
trong đó `X` là đầu ra của hidden layer trong MLP có đầu vào là token “&lt;cls&gt;” đã được mã hóa.

```{.python .input}
#@tab mxnet
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

Chúng ta có thể thấy rằng [**forward inference của một instance `NextSentencePred`**]
trả về các dự đoán nhị phân cho mỗi chuỗi đầu vào BERT.

```{.python .input}
#@tab mxnet
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default will not flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

Cross-entropy loss của 2 phép phân loại nhị phân cũng có thể được tính.

```{.python .input}
#@tab mxnet
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

Đáng chú ý là tất cả các nhãn trong cả hai tác vụ pretraining nói trên
đều có thể dễ dàng thu được từ corpus pretraining mà không cần nỗ lực gán nhãn thủ công.
BERT nguyên bản được pretrain trên sự kết hợp của BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015`
và Wikipedia tiếng Anh.
Hai corpus văn bản này rất lớn:
chúng lần lượt có 800 triệu từ và 2,5 tỷ từ.


## [**Kết hợp tất cả lại**]

Khi pretraining BERT, hàm loss cuối cùng là một tổ hợp tuyến tính của
cả hàm loss của masked language modeling và next sentence prediction.
Bây giờ chúng ta có thể định nghĩa lớp `BERTModel` bằng cách khởi tạo ba lớp
`BERTEncoder`, `MaskLM`, và `NextSentencePred`.
Forward inference trả về các biểu diễn BERT đã mã hóa `encoded_X`,
các dự đoán của masked language modeling `mlm_Y_hat`,
và các dự đoán next sentence `nsp_Y_hat`.

```{.python .input}
#@tab mxnet
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, 
                 num_heads, num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len=max_len)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## Tóm tắt

* Các mô hình word embedding như word2vec và GloVe không phụ thuộc ngữ cảnh. Chúng gán cùng một vector đã pretrain cho cùng một từ bất kể ngữ cảnh của từ đó (nếu có). Chúng khó xử lý tốt hiện tượng đa nghĩa hoặc ngữ nghĩa phức tạp trong ngôn ngữ tự nhiên.
* Đối với các biểu diễn từ phụ thuộc ngữ cảnh như ELMo và GPT, biểu diễn của các từ phụ thuộc vào ngữ cảnh của chúng.
* ELMo mã hóa ngữ cảnh theo hai chiều nhưng sử dụng các kiến trúc dành riêng cho từng tác vụ (tuy nhiên, trên thực tế, việc xây dựng một kiến trúc cụ thể cho mọi tác vụ xử lý ngôn ngữ tự nhiên là không đơn giản); trong khi GPT không phụ thuộc tác vụ nhưng mã hóa ngữ cảnh từ trái sang phải.
* BERT kết hợp ưu điểm của cả hai: nó mã hóa ngữ cảnh theo hai chiều và yêu cầu thay đổi kiến trúc ở mức tối thiểu cho nhiều tác vụ xử lý ngôn ngữ tự nhiên.
* Các embedding của chuỗi đầu vào BERT là tổng của token embedding, segment embedding và positional embedding.
* Pretraining BERT gồm hai tác vụ: masked language modeling và next sentence prediction. Tác vụ đầu có khả năng mã hóa ngữ cảnh hai chiều để biểu diễn các từ, trong khi tác vụ sau mô hình hóa tường minh mối quan hệ logic giữa các cặp văn bản.


## Bài tập

1. Trong điều kiện mọi yếu tố khác như nhau, một masked language model sẽ cần nhiều hay ít bước pretraining hơn để hội tụ so với một language model từ trái sang phải? Tại sao?
1. Trong triển khai ban đầu của BERT, positionwise feed-forward network trong `BERTEncoder` (thông qua `d2l.TransformerEncoderBlock`) và fully connected layer trong `MaskLM` đều sử dụng Gaussian error linear unit (GELU) :cite:`Hendrycks.Gimpel.2016` làm hàm activation. Hãy tìm hiểu sự khác biệt giữa GELU và ReLU.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1490)
:end_tab: