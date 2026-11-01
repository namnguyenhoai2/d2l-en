# Phân rã ma trận

Phân rã ma trận :cite:`Koren.Bell.Volinsky.2009` là một thuật toán lâu đời và được công nhận rộng rãi trong các tài liệu nghiên cứu về hệ thống gợi ý. Phiên bản đầu tiên của mô hình phân rã ma trận được Simon Funk đề xuất trong một [bài đăng trên blog](https://sifter.org/%7Esimon/journal/20061211.html), trong đó ông mô tả ý tưởng phân rã ma trận tương tác. Sau đó, thuật toán này trở nên phổ biến rộng rãi nhờ cuộc thi Netflix được tổ chức vào năm 2006. Vào thời điểm đó, Netflix, một công ty cung cấp dịch vụ truyền phát nội dung đa phương tiện và cho thuê video, đã công bố một cuộc thi nhằm cải thiện hiệu suất hệ thống gợi ý của mình. Đội đạt thành tích tốt nhất và cải thiện 10 phần trăm so với hệ thống cơ sở của Netflix, tức Cinematch, sẽ giành được giải thưởng một triệu USD. Vì vậy, cuộc thi này đã thu hút rất nhiều sự chú ý đến lĩnh vực nghiên cứu hệ thống gợi ý. Sau đó, giải thưởng lớn đã thuộc về đội BellKor's Pragmatic Chaos, một đội kết hợp giữa BellKor, Pragmatic Theory và BigChaos (hiện tại bạn chưa cần quan tâm đến các thuật toán này). Mặc dù điểm số cuối cùng là kết quả của một giải pháp ensemble (tức là sự kết hợp của nhiều thuật toán), thuật toán phân rã ma trận đóng vai trò quan trọng trong tổ hợp cuối cùng. Báo cáo kỹ thuật về giải pháp giành Giải thưởng lớn Netflix :cite:`Toscher.Jahrer.Bell.2009` cung cấp phần giới thiệu chi tiết về mô hình được sử dụng. Trong phần này, chúng ta sẽ tìm hiểu sâu các chi tiết của mô hình phân rã ma trận và cách triển khai mô hình.

## Mô hình phân rã ma trận

Phân rã ma trận là một nhóm các mô hình collaborative filtering. Cụ thể, mô hình phân rã ma trận tương tác người dùng–mặt hàng (ví dụ: ma trận đánh giá) thành tích của hai ma trận có hạng thấp hơn, qua đó nắm bắt cấu trúc hạng thấp của các tương tác người dùng–mặt hàng.

Gọi $\mathbf{R} \in \mathbb{R}^{m \times n}$ là ma trận tương tác với $m$ người dùng và $n$ mặt hàng, trong đó các giá trị của $\mathbf{R}$ biểu diễn các đánh giá tường minh. Tương tác người dùng–mặt hàng sẽ được phân rã thành một ma trận tiềm ẩn của người dùng $\mathbf{P} \in \mathbb{R}^{m \times k}$ và một ma trận tiềm ẩn của mặt hàng $\mathbf{Q} \in \mathbb{R}^{n \times k}$, trong đó $k \ll m, n$ là kích thước của nhân tố tiềm ẩn. Gọi $\mathbf{p}_u$ là hàng thứ $u^\textrm{th}$ của $\mathbf{P}$ và $\mathbf{q}_i$ là hàng thứ $i^\textrm{th}$ của $\mathbf{Q}$. Với một mặt hàng $i$ cho trước, các phần tử của $\mathbf{q}_i$ đo lường mức độ mặt hàng đó sở hữu những đặc điểm nhất định, chẳng hạn như thể loại và ngôn ngữ của một bộ phim. Với một người dùng $u$ cho trước, các phần tử của $\mathbf{p}_u$ đo lường mức độ quan tâm của người dùng đối với những đặc điểm tương ứng của các mặt hàng. Các nhân tố tiềm ẩn này có thể đo lường những khía cạnh dễ nhận biết như trong các ví dụ trên, hoặc hoàn toàn không thể diễn giải được. Các đánh giá dự đoán có thể được ước tính bằng

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

trong đó $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ là ma trận đánh giá dự đoán có cùng dạng với $\mathbf{R}$. Một vấn đề lớn của quy tắc dự đoán này là không thể mô hình hóa độ thiên lệch của người dùng/mặt hàng. Ví dụ, một số người dùng có xu hướng đưa ra đánh giá cao hơn, hoặc một số mặt hàng luôn nhận được đánh giá thấp hơn do chất lượng kém. Những độ thiên lệch này rất phổ biến trong các ứng dụng thực tế. Để nắm bắt các độ thiên lệch này, các hạng tử độ thiên lệch riêng cho người dùng và mặt hàng được đưa vào. Cụ thể, đánh giá dự đoán mà người dùng $u$ đưa ra cho mặt hàng $i$ được tính bằng

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

Sau đó, chúng ta huấn luyện mô hình phân rã ma trận bằng cách cực tiểu hóa sai số bình phương trung bình giữa điểm đánh giá dự đoán và điểm đánh giá thực tế. Hàm mục tiêu được định nghĩa như sau:

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

trong đó $\lambda$ biểu thị hệ số regularization. Hạng tử regularization $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$ được sử dụng để tránh over-fitting bằng cách phạt độ lớn của các tham số. Các cặp $(u, i)$ mà $\mathbf{R}_{ui}$ đã biết được lưu trong tập hợp
$\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \textrm{ is known}\}$. Các tham số của mô hình có thể được học bằng một thuật toán tối ưu hóa, chẳng hạn như Stochastic Gradient Descent và Adam.

Hình minh họa trực quan về mô hình phân rã ma trận được trình bày dưới đây:

![Hình minh họa mô hình phân rã ma trận](../img/rec-mf.svg)

Trong phần còn lại của phần này, chúng ta sẽ giải thích cách triển khai phân rã ma trận và huấn luyện mô hình trên tập dữ liệu MovieLens.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## Triển khai mô hình

Trước tiên, chúng ta triển khai mô hình phân rã ma trận đã mô tả ở trên. Các nhân tố tiềm ẩn của người dùng và mặt hàng có thể được tạo bằng `nn.Embedding`. `input_dim` là số lượng mặt hàng/người dùng và `output_dim` là số chiều của các nhân tố tiềm ẩn $k$. Chúng ta cũng có thể sử dụng `nn.Embedding` để tạo các độ thiên lệch của người dùng/mặt hàng bằng cách đặt `output_dim` thành một. Trong hàm `forward`, các id của người dùng và mặt hàng được sử dụng để tra cứu các embedding.

```{.python .input  n=4}
#@tab mxnet
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## Thước đo đánh giá

Tiếp theo, chúng ta triển khai thước đo RMSE (root-mean-square error), thường được sử dụng để đo sự khác biệt giữa các điểm đánh giá do mô hình dự đoán và các đánh giá thực tế đã quan sát (ground truth) :cite:`Gunawardana.Shani.2015`. RMSE được định nghĩa như sau:

$$
\textrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

trong đó $\mathcal{T}$ là tập hợp gồm các cặp người dùng và mặt hàng mà bạn muốn đánh giá. $|\mathcal{T}|$ là kích thước của tập hợp này. Chúng ta có thể sử dụng hàm RMSE do `mx.metric` cung cấp.

```{.python .input  n=3}
#@tab mxnet
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## Huấn luyện và đánh giá mô hình


Trong hàm huấn luyện, chúng ta sử dụng loss $\ell_2$ với weight decay. Cơ chế weight decay có tác dụng tương tự như regularization $\ell_2$.

```{.python .input  n=4}
#@tab mxnet
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Cuối cùng, hãy kết hợp tất cả các phần lại với nhau và huấn luyện mô hình. Ở đây, chúng ta đặt số chiều của nhân tố tiềm ẩn là 30.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

Dưới đây, chúng ta sử dụng mô hình đã huấn luyện để dự đoán đánh giá mà một người dùng (ID 20) có thể đưa ra cho một mặt hàng (ID 30).

```{.python .input  n=6}
#@tab mxnet
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Tóm tắt

* Mô hình phân rã ma trận được sử dụng rộng rãi trong các hệ thống gợi ý. Mô hình này có thể được sử dụng để dự đoán đánh giá mà người dùng có thể đưa ra cho một mặt hàng.
* Chúng ta có thể triển khai và huấn luyện phân rã ma trận cho các hệ thống gợi ý.


## Bài tập

* Thay đổi kích thước của các nhân tố tiềm ẩn. Kích thước của các nhân tố tiềm ẩn ảnh hưởng như thế nào đến hiệu suất của mô hình?
* Thử các optimizer, learning rate và weight decay rate khác nhau.
* Kiểm tra các điểm đánh giá dự đoán của những người dùng khác đối với một bộ phim cụ thể.


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/400)
:end_tab: