# Collaborative Filtering thần kinh cho xếp hạng cá nhân hóa

Phần này vượt ra ngoài phản hồi tường minh, giới thiệu framework collaborative filtering thần kinh (NCF) cho bài toán recommendation với phản hồi ngầm. Phản hồi ngầm xuất hiện phổ biến trong các hệ thống recommender. Những hành động như nhấp chuột, mua hàng và xem là các dạng phản hồi ngầm thường gặp, dễ thu thập và thể hiện rõ sở thích của người dùng. Mô hình chúng ta sẽ giới thiệu, có tên NeuMF :cite:`He.Liao.Zhang.ea.2017`, viết tắt của neural matrix factorization, nhằm giải quyết bài toán xếp hạng cá nhân hóa với phản hồi ngầm. Mô hình này tận dụng tính linh hoạt và tính phi tuyến của mạng neural để thay thế các tích vô hướng trong matrix factorization, nhằm tăng khả năng biểu diễn của mô hình. Cụ thể, mô hình được cấu trúc từ hai subnetworks gồm generalized matrix factorization (GMF) và MLP, đồng thời mô hình hóa các tương tác theo hai nhánh thay vì chỉ sử dụng các tích vô hướng đơn giản. Đầu ra của hai mạng này được nối lại để tính các điểm dự đoán cuối cùng. Không giống bài toán dự đoán rating trong AutoRec, mô hình này tạo ra một danh sách recommendation được xếp hạng cho mỗi người dùng dựa trên phản hồi ngầm. Chúng ta sẽ sử dụng hàm loss xếp hạng cá nhân hóa được giới thiệu ở phần trước để huấn luyện mô hình này.

## Mô hình NeuMF

Như đã đề cập, NeuMF kết hợp hai subnetworks. GMF là một phiên bản mạng neural tổng quát của matrix factorization, trong đó đầu vào là tích theo từng phần tử của các latent factor của người dùng và item. Mô hình này gồm hai neural layers:

$$
\mathbf{x} = \mathbf{p}_u \odot \mathbf{q}_i \\
\hat{y}_{ui} = \alpha(\mathbf{h}^\top \mathbf{x}),
$$

trong đó $\odot$ biểu thị tích Hadamard của các vector. $\mathbf{P} \in \mathbb{R}^{m \times k}$ và $\mathbf{Q} \in \mathbb{R}^{n \times k}$ lần lượt tương ứng với các ma trận latent của người dùng và item. $\mathbf{p}_u \in \mathbb{R}^{ k}$ là hàng thứ $u^\textrm{th}$ của $P$, còn $\mathbf{q}_i \in \mathbb{R}^{ k}$ là hàng thứ $i^\textrm{th}$ của $Q$. $\alpha$ và $h$ lần lượt biểu thị hàm activation và trọng số của output layer. $\hat{y}_{ui}$ là điểm dự đoán mà người dùng $u$ có thể gán cho item $i$.

Một thành phần khác của mô hình này là MLP. Để tăng tính linh hoạt của mô hình, subnetwork MLP không dùng chung embedding của người dùng và item với GMF. MLP sử dụng phép nối embedding của người dùng và item làm đầu vào. Với các kết nối phức tạp và các phép biến đổi phi tuyến, MLP có khả năng ước lượng những tương tác tinh vi giữa người dùng và item. Cụ thể hơn, subnetwork MLP được định nghĩa như sau:

$$
\begin{aligned}
z^{(1)} &= \phi_1(\mathbf{U}_u, \mathbf{V}_i) = \left[ \mathbf{U}_u, \mathbf{V}_i \right] \\
\phi^{(2)}(z^{(1)})  &= \alpha^1(\mathbf{W}^{(2)} z^{(1)} + b^{(2)}) \\
&... \\
\phi^{(L)}(z^{(L-1)}) &= \alpha^L(\mathbf{W}^{(L)} z^{(L-1)} + b^{(L)})) \\
\hat{y}_{ui} &= \alpha(\mathbf{h}^\top\phi^L(z^{(L-1)}))
\end{aligned}
$$

trong đó $\mathbf{W}^*, \mathbf{b}^*$ và $\alpha^*$ lần lượt biểu thị ma trận trọng số, vector bias và hàm activation. $\phi^*$ biểu thị hàm của layer tương ứng. $\mathbf{z}^*$ biểu thị đầu ra của layer tương ứng.

Để kết hợp kết quả của GMF và MLP, thay vì cộng đơn giản, NeuMF nối hai layer áp chót của hai subnetworks để tạo thành một feature vector có thể được truyền đến các layer tiếp theo. Sau đó, các đầu ra được chiếu bằng ma trận $\mathbf{h}$ và một hàm activation sigmoid. Prediction layer được biểu diễn như sau:
$$
\hat{y}_{ui} = \sigma(\mathbf{h}^\top[\mathbf{x}, \phi^L(z^{(L-1)})]).
$$

Hình dưới đây minh họa kiến trúc mô hình NeuMF.

![Minh họa mô hình NeuMF](../img/rec-neumf.svg)

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## Triển khai mô hình
Đoạn code sau triển khai mô hình NeuMF. Mô hình gồm một generalized matrix factorization model và một MLP với các vector embedding khác nhau cho người dùng và item. Cấu trúc của MLP được điều khiển bằng tham số `nums_hiddens`. ReLU được sử dụng làm hàm activation mặc định.

```{.python .input  n=2}
#@tab mxnet
class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens,
                 **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Dense(num_hiddens, activation='relu',
                                  use_bias=True))
        self.prediction_layer = nn.Dense(1, activation='sigmoid', use_bias=False)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(np.concatenate([p_mlp, q_mlp], axis=1))
        con_res = np.concatenate([gmf, mlp], axis=1)
        return self.prediction_layer(con_res)
```

## Dataset tùy chỉnh với Negative Sampling

Đối với pairwise ranking loss, một bước quan trọng là negative sampling. Với mỗi người dùng, các item mà người dùng chưa từng tương tác là các item ứng viên (các entry chưa được quan sát). Hàm sau nhận identity của người dùng và các item ứng viên làm đầu vào, rồi lấy mẫu ngẫu nhiên các item âm cho từng người dùng từ tập ứng viên của người dùng đó. Trong giai đoạn training, mô hình đảm bảo rằng các item mà người dùng thích được xếp hạng cao hơn những item mà họ không thích hoặc chưa từng tương tác.

```{.python .input  n=3}
#@tab mxnet
class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
```

## Bộ đánh giá

Trong phần này, chúng ta áp dụng chiến lược chia theo thời gian để xây dựng tập training và tập test. Hai thước đo đánh giá gồm hit rate tại ngưỡng cắt $\ell$ cho trước ($\textrm{Hit}@\ell$) và diện tích dưới đường cong ROC (AUC) được sử dụng để đánh giá hiệu quả của mô hình. Hit rate tại vị trí $\ell$ cho mỗi người dùng cho biết item được recommendation có nằm trong danh sách xếp hạng top $\ell$ hay không. Định nghĩa chính thức như sau:

$$
\textrm{Hit}@\ell = \frac{1}{m} \sum_{u \in \mathcal{U}} \textbf{1}(rank_{u, g_u} <= \ell),
$$

trong đó $\textbf{1}$ biểu thị một hàm chỉ báo, có giá trị bằng một nếu item ground truth được xếp hạng trong danh sách top $\ell$, và bằng không trong trường hợp ngược lại. $rank_{u, g_u}$ biểu thị thứ hạng của item ground truth $g_u$ của người dùng $u$ trong danh sách recommendation (thứ hạng lý tưởng là 1). $m$ là số người dùng. $\mathcal{U}$ là tập người dùng.

Định nghĩa của AUC như sau:

$$
\textrm{AUC} = \frac{1}{m} \sum_{u \in \mathcal{U}} \frac{1}{|\mathcal{I} \backslash S_u|} \sum_{j \in I \backslash S_u} \textbf{1}(rank_{u, g_u} < rank_{u, j}),
$$

trong đó $\mathcal{I}$ là tập item. $S_u$ là các item ứng viên của người dùng $u$. Lưu ý rằng nhiều protocol đánh giá khác như precision, recall và normalized discounted cumulative gain (NDCG) cũng có thể được sử dụng.

Hàm sau tính số hit và AUC cho từng người dùng.

```{.python .input  n=4}
#@tab mxnet
#@save
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc
```

Sau đó, Hit rate và AUC tổng thể được tính như sau.

```{.python .input  n=5}
#@tab mxnet
#@save
def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([np.array(user_ids)])
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([np.array(item_ids)])
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x), shuffle=False, last_batch="keep",
            batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, devices, even_split=False)
                 for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```

## Huấn luyện và đánh giá mô hình

Hàm training được định nghĩa bên dưới. Chúng ta huấn luyện mô hình theo pairwise manner.

```{.python .input  n=6}
#@tab mxnet
#@save
def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            with autograd.record():
                p_pos = [net(*t) for t in zip(*input_data[:-1])]
                p_neg = [net(*t) for t in zip(*input_data[:-2],
                                              input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            [l.backward(retain_graph=False) for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean()/len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        with autograd.predict_mode():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Bây giờ, chúng ta có thể tải dataset MovieLens 100k và huấn luyện mô hình. Vì dataset MovieLens chỉ có các rating, nên với một phần độ chính xác bị giảm, chúng ta nhị phân hóa các rating này thành số 0 và 1. Nếu người dùng đã rating một item, chúng ta xem phản hồi ngầm là một; nếu không thì xem là không. Hành động rating một item có thể được xem là một hình thức cung cấp phản hồi ngầm. Ở đây, chúng ta chia dataset theo chế độ `seq-aware`, trong đó các item mà người dùng tương tác gần đây nhất được giữ lại cho tập test.

```{.python .input  n=11}
#@tab mxnet
batch_size = 1024
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_iter = gluon.data.DataLoader(
    PRDataset(users_train, items_train, candidates, num_items ), batch_size,
    True, last_batch="rollover", num_workers=d2l.get_dataloader_workers())
```

Sau đó, chúng ta tạo và khởi tạo mô hình. Chúng ta sử dụng một MLP ba layer với kích thước hidden cố định là 10.

```{.python .input  n=8}
#@tab mxnet
devices = d2l.try_all_gpus()
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
```

Đoạn code sau huấn luyện mô hình.

```{.python .input  n=12}
#@tab mxnet
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_ranking(net, train_iter, test_iter, loss, trainer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)
```

## Tóm tắt

* Việc thêm tính phi tuyến vào mô hình matrix factorization có lợi cho việc cải thiện năng lực và hiệu quả của mô hình.
* NeuMF là sự kết hợp giữa matrix factorization và một MLP. MLP nhận phép nối embedding của người dùng và item làm đầu vào.

## Bài tập

* Thay đổi kích thước của các latent factor. Kích thước của latent factor ảnh hưởng như thế nào đến hiệu năng của mô hình?
* Thay đổi kiến trúc (ví dụ: số lượng layer, số neuron của mỗi layer) của MLP để kiểm tra ảnh hưởng của chúng đến hiệu năng.
* Thử các optimizer, learning rate và weight decay rate khác nhau.
* Thử sử dụng hinge loss được định nghĩa ở phần trước để tối ưu mô hình này.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/403)
:end_tab: