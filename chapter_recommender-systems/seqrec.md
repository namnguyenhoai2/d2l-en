# Hệ thống đề xuất có nhận biết trình tự

Trong các phần trước, chúng ta trừu tượng hóa bài toán đề xuất thành bài toán hoàn thiện ma trận mà không xét đến hành vi ngắn hạn của người dùng. Trong phần này, chúng ta sẽ giới thiệu một mô hình đề xuất có tính đến các nhật ký tương tác của người dùng được sắp xếp theo trình tự. Đây là một hệ thống đề xuất có nhận biết trình tự :cite:`Quadrana.Cremonesi.Jannach.2018`, trong đó đầu vào là danh sách các hành động trước đây của người dùng được sắp xếp theo thứ tự và thường có kèm dấu thời gian. Nhiều nghiên cứu gần đây đã chứng minh tính hữu ích của việc đưa thông tin này vào mô hình hóa các mẫu hành vi theo thời gian của người dùng và phát hiện sự dịch chuyển trong mối quan tâm của họ.

Mô hình chúng ta sẽ giới thiệu, Caser :cite:`Tang.Wang.2018`, viết tắt của convolutional sequence embedding recommendation model, sử dụng mạng nơ-ron tích chập để nắm bắt ảnh hưởng của các mẫu động trong những hoạt động gần đây của người dùng. Thành phần chính của Caser gồm một mạng tích chập ngang và một mạng tích chập dọc, lần lượt nhằm phát hiện các mẫu trình tự ở cấp độ tổ hợp và cấp độ điểm. Mẫu ở cấp độ điểm biểu thị ảnh hưởng của một mục đơn lẻ trong trình tự lịch sử đến mục tiêu, trong khi mẫu ở cấp độ tổ hợp biểu thị ảnh hưởng của một số hành động trước đó đến mục tiêu tiếp theo. Ví dụ, việc mua cả sữa và bơ cùng lúc dẫn đến xác suất mua bột mì cao hơn so với chỉ mua một trong hai món. Ngoài ra, mối quan tâm chung của người dùng, hay sở thích dài hạn, cũng được mô hình hóa trong các lớp kết nối đầy đủ cuối cùng, tạo ra mô hình hóa toàn diện hơn về mối quan tâm của người dùng. Chi tiết mô hình được mô tả như sau.

## Kiến trúc mô hình

Trong hệ thống đề xuất có nhận biết trình tự, mỗi người dùng được gắn với một trình tự gồm một số mục từ tập mục. Gọi $S^u = (S_1^u, ... S_{|S_u|}^u)$ là trình tự được sắp xếp theo thứ tự. Mục tiêu của Caser là đề xuất mục bằng cách xem xét cả sở thích chung và ý định ngắn hạn của người dùng. Giả sử chúng ta xem xét $L$ mục trước đó, một ma trận embedding biểu diễn các tương tác trước thời điểm $t$ có thể được xây dựng như sau:

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

trong đó $\mathbf{Q} \in \mathbb{R}^{n \times k}$ biểu diễn các embedding của mục và $\mathbf{q}_i$ biểu thị hàng thứ $i^\textrm{th}$. $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$ có thể được sử dụng để suy ra mối quan tâm nhất thời của người dùng $u$ tại thời điểm $t$. Chúng ta có thể xem ma trận đầu vào $\mathbf{E}^{(u, t)}$ như một hình ảnh, làm đầu vào cho hai thành phần tích chập tiếp theo.

Lớp tích chập ngang có $d$ bộ lọc ngang $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$, còn lớp tích chập dọc có $d'$ bộ lọc dọc $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$. Sau một loạt phép tích chập và pooling, chúng ta thu được hai đầu ra:

$$
\mathbf{o} = \textrm{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \textrm{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

trong đó $\mathbf{o} \in \mathbb{R}^d$ là đầu ra của mạng tích chập ngang và $\mathbf{o}' \in \mathbb{R}^{kd'}$ là đầu ra của mạng tích chập dọc. Để đơn giản, chúng ta bỏ qua chi tiết của các phép tích chập và pooling. Hai đầu ra này được nối lại và đưa vào một lớp mạng nơ-ron kết nối đầy đủ để thu được các biểu diễn ở cấp độ cao hơn.

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

trong đó $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$ là ma trận trọng số và $\mathbf{b} \in \mathbb{R}^k$ là bias. Vector được học $\mathbf{z} \in \mathbb{R}^k$ là biểu diễn cho ý định ngắn hạn của người dùng.

Cuối cùng, hàm dự đoán kết hợp sở thích ngắn hạn và sở thích chung của người dùng, được định nghĩa như sau:

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

trong đó $\mathbf{V} \in \mathbb{R}^{n \times 2k}$ là một ma trận embedding khác của mục. $\mathbf{b}' \in \mathbb{R}^n$ là bias riêng của mục. $\mathbf{P} \in \mathbb{R}^{m \times k}$ là ma trận embedding của người dùng, biểu diễn sở thích chung của người dùng. $\mathbf{p}_u \in \mathbb{R}^{ k}$ là hàng thứ $u^\textrm{th}$ của $P$ và $\mathbf{v}_i \in \mathbb{R}^{2k}$ là hàng thứ $i^\textrm{th}$ của $\mathbf{V}$.

Mô hình có thể được học bằng BPR hoặc Hinge loss. Kiến trúc của Caser được minh họa bên dưới:

![Minh họa mô hình Caser](../img/rec-caser.svg)

Trước tiên, chúng ta import các thư viện cần thiết.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## Triển khai mô hình
Đoạn mã sau triển khai mô hình Caser. Mô hình gồm một lớp tích chập dọc, một lớp tích chập ngang và một lớp kết nối đầy đủ.

```{.python .input  n=4}
#@tab mxnet
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## Tập dữ liệu tuần tự với negative sampling
Để xử lý dữ liệu tương tác tuần tự, chúng ta cần triển khai lại lớp `Dataset`. Đoạn mã sau tạo một lớp dataset mới có tên `SeqDataset`. Trong mỗi mẫu, lớp này xuất ra danh tính người dùng, $L$ mục mà người dùng đã tương tác trước đó dưới dạng một trình tự, và mục tiếp theo mà người dùng tương tác làm mục tiêu. Hình sau minh họa quy trình tải dữ liệu cho một người dùng. Giả sử người dùng này thích 9 bộ phim, chúng ta sắp xếp chín bộ phim theo thứ tự thời gian. Bộ phim mới nhất được giữ lại làm mục kiểm tra. Với tám bộ phim còn lại, chúng ta có thể tạo ba mẫu huấn luyện, trong đó mỗi mẫu chứa một trình tự gồm năm bộ phim ($L=5$) và mục tiếp theo của trình tự đó làm mục tiêu. Các mẫu âm cũng được đưa vào dataset tùy chỉnh.

![Minh họa quy trình tạo dữ liệu](../img/rec-seq-data.svg)

```{.python .input  n=5}
#@tab mxnet
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## Tải dataset MovieLens 100K

Sau đó, chúng ta đọc và chia dataset MovieLens 100K ở chế độ có nhận biết trình tự, đồng thời tải dữ liệu huấn luyện bằng sequential dataloader đã triển khai ở trên.

```{.python .input  n=6}
#@tab mxnet
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

Cấu trúc dữ liệu huấn luyện được minh họa ở trên. Phần tử đầu tiên là danh tính người dùng, danh sách tiếp theo biểu thị năm mục gần nhất mà người dùng này thích, còn phần tử cuối cùng là mục mà người dùng thích sau năm mục đó.

## Huấn luyện mô hình
Bây giờ, hãy huấn luyện mô hình. Chúng ta sử dụng cùng thiết lập như NeuMF, bao gồm learning rate, optimizer và $k$, trong phần trước để kết quả có thể so sánh được.

```{.python .input  n=7}
#@tab mxnet
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# Running takes > 1h (pending fix from MXNet)
# d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter, num_users, num_items, num_epochs, devices, d2l.evaluate_ranking, candidates, eval_step=1)
```

## Tóm tắt
* Suy luận sở thích ngắn hạn và dài hạn của người dùng có thể giúp dự đoán hiệu quả hơn mục tiếp theo mà họ sẽ yêu thích.
* Mạng nơ-ron tích chập có thể được sử dụng để nắm bắt sở thích ngắn hạn của người dùng từ các tương tác tuần tự.

## Bài tập

* Thực hiện ablation study bằng cách loại bỏ một trong hai mạng tích chập ngang và dọc. Thành phần nào quan trọng hơn?
* Thay đổi hyperparameter $L$. Liệu việc sử dụng lịch sử tương tác dài hơn có mang lại độ chính xác cao hơn không?
* Ngoài bài toán đề xuất có nhận biết trình tự được giới thiệu ở trên, còn có một loại bài toán đề xuất có nhận biết trình tự khác gọi là session-based recommendation :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`. Bạn có thể giải thích sự khác biệt giữa hai bài toán này không?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/404)
:end_tab: