# Xếp hạng cá nhân hóa cho hệ thống gợi ý

Trong các phần trước, chỉ phản hồi tường minh được xem xét, và các mô hình được huấn luyện và kiểm thử trên những rating đã quan sát. Những phương pháp này có hai nhược điểm: Thứ nhất, trong các tình huống thực tế, phần lớn phản hồi không tường minh mà là phản hồi ngầm định, và việc thu thập phản hồi tường minh có thể tốn kém hơn. Thứ hai, các cặp người dùng–mục chưa được quan sát, vốn có thể dự đoán mối quan tâm của người dùng, hoàn toàn bị bỏ qua. Điều này khiến các phương pháp này không phù hợp với những trường hợp rating không bị thiếu một cách ngẫu nhiên mà bị thiếu do sở thích của người dùng. Các cặp người dùng–mục chưa được quan sát là sự pha trộn giữa phản hồi tiêu cực thực sự (người dùng không quan tâm đến các mục đó) và các giá trị bị thiếu (người dùng có thể tương tác với các mục đó trong tương lai). Trong matrix factorization và AutoRec, chúng ta đơn giản bỏ qua các cặp chưa được quan sát. Rõ ràng, các mô hình này không có khả năng phân biệt giữa các cặp đã quan sát và chưa được quan sát, đồng thời thường không phù hợp với các tác vụ xếp hạng cá nhân hóa.

Vì mục tiêu này, một nhóm các mô hình gợi ý hướng đến việc tạo ra các danh sách gợi ý đã được xếp hạng từ phản hồi ngầm định ngày càng trở nên phổ biến. Nhìn chung, các mô hình xếp hạng cá nhân hóa có thể được tối ưu bằng các phương pháp pointwise, pairwise hoặc listwise. Các phương pháp pointwise xem xét từng tương tác một và huấn luyện một classifier hoặc regressor để dự đoán sở thích riêng lẻ. Matrix factorization và AutoRec được tối ưu bằng các objective pointwise. Các phương pháp pairwise xem xét một cặp mục cho mỗi người dùng và cố gắng xấp xỉ thứ tự tối ưu cho cặp đó. Thông thường, các phương pháp pairwise phù hợp hơn với tác vụ xếp hạng vì việc dự đoán thứ tự tương đối tương đồng với bản chất của xếp hạng. Các phương pháp listwise xấp xỉ thứ tự của toàn bộ danh sách các mục, chẳng hạn như tối ưu trực tiếp các chỉ số xếp hạng như Normalized Discounted Cumulative Gain ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)). Tuy nhiên, các phương pháp listwise phức tạp và đòi hỏi nhiều năng lực tính toán hơn các phương pháp pointwise hoặc pairwise. Trong phần này, chúng ta sẽ giới thiệu hai objective/loss pairwise là Bayesian Personalized Ranking loss và Hinge loss, cùng các cách triển khai tương ứng.

## Bayesian Personalized Ranking Loss và cách triển khai

Bayesian personalized ranking (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` là một loss xếp hạng cá nhân hóa pairwise được suy ra từ maximum posterior estimator. Loss này đã được sử dụng rộng rãi trong nhiều mô hình gợi ý hiện có. Dữ liệu huấn luyện của BPR bao gồm cả các cặp dương và cặp âm (các giá trị bị thiếu). Loss giả định rằng người dùng ưu tiên mục dương hơn tất cả các mục chưa được quan sát khác.

Về mặt hình thức, dữ liệu huấn luyện được xây dựng từ các tuple có dạng $(u, i, j)$, biểu thị rằng người dùng $u$ ưu tiên mục $i$ hơn mục $j$. Công thức Bayesian của BPR, nhằm tối đa hóa xác suất hậu nghiệm, được cho như sau:

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

Trong đó, $\Theta$ biểu diễn các tham số của một mô hình gợi ý bất kỳ, còn $>_u$ biểu diễn thứ hạng tổng thể cá nhân hóa mong muốn của tất cả các mục đối với người dùng $u$. Chúng ta có thể xây dựng maximum posterior estimator để suy ra tiêu chí tối ưu tổng quát cho tác vụ xếp hạng cá nhân hóa.

$$
\begin{aligned}
\textrm{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$


trong đó $D \stackrel{\textrm{def}}{=} \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ là tập huấn luyện, với $I^+_u$ biểu thị các mục mà người dùng $u$ thích, $I$ biểu thị tất cả các mục, còn $I \backslash I^+_u$ biểu thị tất cả các mục khác, không bao gồm những mục mà người dùng thích. $\hat{y}_{ui}$ và $\hat{y}_{uj}$ lần lượt là các điểm số dự đoán của người dùng $u$ đối với mục $i$ và $j$. Prior $p(\Theta)$ là một phân phối chuẩn có trung bình bằng 0 và ma trận phương sai–hiệp phương sai là $\Sigma_\Theta$. Ở đây, chúng ta đặt $\Sigma_\Theta = \lambda_\Theta I$.

![Minh họa về Bayesian Personalized Ranking](../img/rec-ranking.svg)
Chúng ta sẽ triển khai base class  `mxnet.gluon.loss.Loss` và override phương thức `forward` để xây dựng Bayesian personalized ranking loss. Trước tiên, chúng ta import class Loss và module np.

```{.python .input  n=5}
#@tab mxnet
from mxnet import gluon, np, npx
npx.set_np()
```

Cách triển khai BPR loss như sau.

```{.python .input  n=2}
#@tab mxnet
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Hinge Loss và cách triển khai

Hinge loss dùng cho xếp hạng có dạng khác với [hinge loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) được cung cấp trong thư viện gluon và thường được sử dụng trong các classifier như SVM. Loss được sử dụng cho xếp hạng trong các hệ thống gợi ý có dạng sau.

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

trong đó $m$ là kích thước của safety margin. Loss này nhằm đẩy các mục âm ra xa các mục dương. Tương tự BPR, loss này nhằm tối ưu khoảng cách liên quan giữa các mẫu dương và âm thay vì các đầu ra tuyệt đối, nên rất phù hợp với các hệ thống gợi ý.

```{.python .input  n=3}
#@tab mxnet
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

Hai loss này có thể được sử dụng thay thế cho nhau trong tác vụ xếp hạng cá nhân hóa của hệ thống gợi ý.

## Tóm tắt

- Có ba loại ranking loss cho tác vụ xếp hạng cá nhân hóa trong các hệ thống gợi ý, gồm các phương pháp pointwise, pairwise và listwise.
- Hai loss pairwise là Bayesian personalized ranking loss và hinge loss có thể được sử dụng thay thế cho nhau.

## Bài tập

- Có những biến thể nào của BPR và hinge loss?
- Bạn có thể tìm thấy các mô hình gợi ý nào sử dụng BPR hoặc hinge loss?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/402)
:end_tab: