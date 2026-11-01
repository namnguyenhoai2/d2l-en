```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# Tối ưu hóa siêu tham số là gì?
:label:`sec_what_is_hpo`

Như chúng ta đã thấy trong các chương trước, mạng neural sâu có một số lượng lớn
tham số hoặc trọng số được học trong quá trình huấn luyện. Ngoài các tham số này,
mọi mạng neural còn có thêm các *siêu tham số* cần được người dùng cấu hình. Ví dụ,
để đảm bảo stochastic gradient descent hội tụ đến một cực tiểu địa phương của hàm
mất mát huấn luyện (xem :numref:`chap_optimization`), chúng ta phải điều chỉnh
learning rate và batch size. Để tránh overfitting trên các tập dữ liệu huấn luyện,
chúng ta có thể phải thiết lập các tham số regularization, chẳng hạn như weight decay
(xem :numref:`sec_weight_decay`) hoặc dropout (xem :numref:`sec_dropout`).
Chúng ta có thể xác định capacity và inductive bias của mô hình bằng cách thiết lập
số lớp và số unit hoặc filter trên mỗi lớp (tức là số lượng trọng số hiệu dụng).

Đáng tiếc là chúng ta không thể chỉ điều chỉnh các siêu tham số này bằng cách tối
thiểu hóa training loss, vì điều đó sẽ dẫn đến overfitting trên dữ liệu huấn luyện.
Ví dụ, việc đặt các tham số regularization như dropout hoặc weight decay bằng không
sẽ dẫn đến training loss nhỏ, nhưng có thể làm giảm hiệu năng generalization.

![Quy trình điển hình trong machine learning, bao gồm việc huấn luyện mô hình nhiều lần với các siêu tham số khác nhau.](../img/ml_workflow.svg)
:label:`ml_workflow`

Nếu không có một hình thức tự động hóa khác, các siêu tham số phải được thiết lập
thủ công theo cách thử và sai, trở thành một phần tốn thời gian và khó khăn trong
quy trình machine learning. Ví dụ, hãy xét việc huấn luyện một ResNet
(xem :numref:`sec_resnet`) trên CIFAR-10, quá trình này cần hơn 2 giờ
trên một instance Amazon Elastic Cloud Compute (EC2) `g4dn.xlarge`. Ngay cả
chỉ thử tuần tự mười cấu hình siêu tham số cũng sẽ mất khoảng một ngày. Tệ hơn nữa,
các siêu tham số thường không thể được chuyển trực tiếp giữa các kiến trúc và tập dữ
liệu :cite:`feurer-arxiv22,wistuba-ml18,bardenet-icml13a`, và cần được tối ưu hóa lại cho mỗi tác vụ mới.
Ngoài ra, đối với hầu hết siêu tham số, không có quy tắc kinh nghiệm rõ ràng nào,
và cần có kiến thức chuyên môn để tìm ra các giá trị hợp lý.

Các thuật toán *tối ưu hóa siêu tham số (HPO)* được thiết kế để giải quyết vấn đề
này một cách có hệ thống và tự động :cite:`feurer-automlbook18a`, bằng cách đặt vấn
đề dưới dạng một bài toán tối ưu hóa toàn cục. Mục tiêu mặc định là lỗi trên một tập
dữ liệu validation được giữ lại, nhưng về nguyên tắc có thể là bất kỳ metric nghiệp
vụ nào khác. Mục tiêu này có thể được kết hợp với hoặc bị ràng buộc bởi các mục tiêu
thứ cấp, chẳng hạn như thời gian huấn luyện, thời gian inference hoặc độ phức tạp
của mô hình.

Gần đây, tối ưu hóa siêu tham số đã được mở rộng sang *neural architecture
search (NAS)* :cite:`elsken-arxiv18a,wistuba-arxiv19`, trong đó mục tiêu là tìm ra các kiến trúc
mạng neural hoàn toàn mới. So với HPO cổ điển, NAS thậm chí còn tốn kém hơn về mặt
tính toán và đòi hỏi thêm nỗ lực để vẫn khả thi trong thực tế. Cả HPO và NAS đều có
thể được xem là các phân ngành của AutoML :cite:`hutter-book19a`, với mục tiêu
tự động hóa toàn bộ ML pipeline.

Trong phần này, chúng ta sẽ giới thiệu HPO và trình bày cách tự động tìm các siêu
tham số tốt nhất cho ví dụ logistic regression được giới thiệu trong
:numref:`sec_softmax_concise`.

##  Bài toán tối ưu hóa
:label:`sec_definition_hpo`

Chúng ta sẽ bắt đầu với một bài toán minh họa đơn giản: tìm learning rate cho mô
hình multi-class logistic regression `SoftmaxRegression` từ
:numref:`sec_softmax_concise` để tối thiểu hóa validation error trên tập dữ liệu
Fashion MNIST. Mặc dù các siêu tham số khác như batch size hoặc số epoch cũng đáng
được điều chỉnh, chúng ta chỉ tập trung vào learning rate để đơn giản hóa.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from scipy import stats
```

Trước khi có thể chạy HPO, trước hết chúng ta cần xác định hai thành phần: hàm mục
tiêu và configuration space.

### Hàm mục tiêu

Hiệu năng của một thuật toán học có thể được xem là một hàm
$f: \mathcal{X} \rightarrow \mathbb{R}$ ánh xạ từ không gian siêu tham số
$\mathbf{x} \in \mathcal{X}$ đến validation loss. Với mỗi lần đánh giá
$f(\mathbf{x})$, chúng ta phải huấn luyện và validation mô hình machine learning, việc này
có thể tốn nhiều thời gian và tài nguyên tính toán trong trường hợp mạng neural sâu
được huấn luyện trên các tập dữ liệu lớn. Với tiêu chí $f(\mathbf{x})$, mục tiêu của chúng ta là tìm
$\mathbf{x}_{\star} \in \mathrm{argmin}_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$.

Không có cách đơn giản nào để tính gradient của $f$ theo $\mathbf{x}$,
vì điều đó đòi hỏi phải lan truyền gradient qua toàn bộ quá trình huấn luyện. Mặc
dù đã có các công trình gần đây :cite:`maclaurin-icml15,franceschi-icml17a` nhằm dẫn dắt HPO bằng
các “hypergradient” xấp xỉ, chưa có phương pháp hiện có nào đạt hiệu năng cạnh tranh
với state-of-the-art, nên chúng ta sẽ không thảo luận về chúng ở đây. Hơn nữa, chi
phí tính toán để đánh giá $f$ đòi hỏi các thuật toán HPO phải tiến gần đến cực trị
toàn cục với số mẫu ít nhất có thể.

Quá trình huấn luyện mạng neural mang tính ngẫu nhiên (ví dụ: trọng số được khởi tạo
ngẫu nhiên, các mini-batch được lấy mẫu ngẫu nhiên), vì vậy các quan sát của chúng
ta sẽ có nhiễu: $y \sim f(\mathbf{x}) + \epsilon$, trong đó chúng ta thường giả định rằng
nhiễu quan sát $\epsilon \sim N(0, \sigma)$ có phân phối Gaussian.

Đối mặt với tất cả những thách thức này, chúng ta thường cố gắng nhanh chóng xác định
một tập nhỏ các cấu hình siêu tham số hoạt động tốt, thay vì đạt chính xác các cực
trị toàn cục. Tuy nhiên, do nhu cầu tính toán lớn của hầu hết các mô hình mạng neural,
ngay cả việc này cũng có thể mất vài ngày hoặc vài tuần tính toán. Trong
:numref:`sec_mf_hpo`, chúng ta sẽ tìm hiểu cách tăng tốc quá trình tối ưu
hóa bằng cách phân tán quá trình tìm kiếm hoặc sử dụng các xấp xỉ hàm mục tiêu có
chi phí đánh giá thấp hơn.

Chúng ta bắt đầu với một phương pháp tính validation error của một mô hình.

```{.python .input  n=8}
%%tab pytorch
class HPOTrainer(d2l.Trainer):  #@save
    def validation_error(self):
        self.model.eval()
        accuracy = 0
        val_batch_idx = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                x, y = self.prepare_batch(batch)
                y_hat = self.model(x)
                accuracy += self.model.accuracy(y_hat, y)
            val_batch_idx += 1
        return 1 -  accuracy / val_batch_idx
```

Chúng ta tối ưu hóa validation error theo cấu hình siêu tham số
`config`, bao gồm `learning_rate`. Với mỗi lần đánh giá, chúng ta huấn luyện
mô hình trong `max_epochs` epoch, sau đó tính và trả về validation error của mô hình:

```{.python .input  n=5}
%%tab pytorch
def hpo_objective_softmax_classification(config, max_epochs=8):
    learning_rate = config["learning_rate"]
    trainer = d2l.HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=16)
    model = d2l.SoftmaxRegression(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    return d2l.numpy(trainer.validation_error())
```

### Configuration space
:label:`sec_intro_config_spaces`

Cùng với hàm mục tiêu $f(\mathbf{x})$, chúng ta cũng cần xác định tập khả thi
$\mathbf{x} \in \mathcal{X}$ để tối ưu hóa, được gọi là
*configuration space* hoặc *search space*. Đối với ví dụ logistic regression,
chúng ta sẽ sử dụng:

```{.python .input  n=6}
config_space = {"learning_rate": stats.loguniform(1e-4, 1)}
```

Ở đây, chúng ta sử dụng đối tượng `loguniform` từ SciPy, đại diện cho một
phân phối đều từ -4 đến -1 trong không gian logarithmic. Đối tượng này cho phép
chúng ta lấy mẫu các biến ngẫu nhiên từ phân phối đó.

Mỗi siêu tham số có một kiểu dữ liệu, chẳng hạn như `float` đối với
`learning_rate`, cũng như một miền giá trị đóng và bị chặn (tức là cận dưới và
cận trên). Chúng ta thường gán một phân phối tiên nghiệm (ví dụ: uniform hoặc
log-uniform) cho mỗi siêu tham số để lấy mẫu. Một số tham số dương, chẳng hạn như
`learning_rate`, được biểu diễn tốt nhất trên thang logarithmic vì các giá trị
tối ưu có thể khác nhau nhiều bậc độ lớn, trong khi các tham số khác, chẳng hạn như
momentum, sử dụng thang tuyến tính.

Dưới đây là một ví dụ đơn giản về configuration space gồm các siêu tham số điển
hình của một multi-layer perceptron, cùng với kiểu và miền giá trị tiêu chuẩn của
chúng.

: Configuration space mẫu của multi-layer perceptron
:label:`tab_example_configspace`

| Tên                 | Kiểu        |Miền giá trị siêu tham số     | thang logarithmic |
| :----:              | :----:      |:------------------------------:|:---------:|
| learning rate       | float       |      $[10^{-6},10^{-1}]$       |    có    |
| batch size          | integer     |           $[8,256]$            |    có    |
| momentum            | float       |           $[0,0.99]$           |    không     |
| activation function | categorical | $\{\textrm{tanh}, \textrm{relu}\}$ |     -     |
| number of units     | integer     |          $[32, 1024]$          |    có    |
| number of layers    | integer     |            $[1, 6]$            |    không     |



Nhìn chung, cấu trúc của configuration space $\mathcal{X}$ có thể phức
tạp và khá khác với $\mathbb{R}^d$. Trong thực tế, một số siêu
tham số có thể phụ thuộc vào giá trị của các siêu tham số khác. Ví dụ, giả sử chúng
ta muốn điều chỉnh số lớp của một multi-layer perceptron, cũng như số unit cho mỗi
lớp. Số unit của lớp $l\textrm{-th}$ chỉ có ý nghĩa nếu mạng có ít
nhất $l+1$ lớp. Những bài toán HPO nâng cao này nằm ngoài phạm vi của chương. Bạn
đọc quan tâm có thể tham khảo :cite:`hutter-lion11a,jenatton-icml17a,baptista-icml18a`.

Configuration space đóng vai trò quan trọng trong tối ưu hóa siêu tham số, vì không
thuật toán nào có thể tìm được thứ không nằm trong configuration space. Mặt khác,
nếu các miền giá trị quá lớn, ngân sách tính toán để tìm các cấu hình hoạt động tốt
có thể trở nên không khả thi.

## Random Search
:label:`sec_rs`

*Random search* là thuật toán tối ưu hóa siêu tham số đầu tiên mà chúng ta sẽ xem
xét. Ý tưởng chính của random search là lấy mẫu độc lập từ configuration space cho
đến khi hết ngân sách được xác định trước (ví dụ: số vòng lặp tối đa), rồi trả về
cấu hình tốt nhất đã quan sát. Tất cả các lần đánh giá có thể được thực thi song
song một cách độc lập (xem :numref:`sec_rs_async`), nhưng ở đây chúng ta sử
dụng một vòng lặp tuần tự để đơn giản hóa.

```{.python .input  n=7}
errors, values = [], []
num_iterations = 5

for i in range(num_iterations):
    learning_rate = config_space["learning_rate"].rvs()
    print(f"Trial {i}: learning_rate = {learning_rate}")
    y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
    print(f"    validation_error = {y}")
    values.append(learning_rate)
    errors.append(y)
```

Khi đó, learning rate tốt nhất đơn giản là learning rate có validation error thấp nhất.

```{.python .input  n=7}
best_idx = np.argmin(errors)
print(f"optimal learning rate = {values[best_idx]}")
```

Nhờ tính đơn giản và tính tổng quát, random search là một trong những thuật toán HPO
được sử dụng thường xuyên nhất. Thuật toán này không yêu cầu triển khai phức tạp và
có thể áp dụng cho mọi configuration space, miễn là chúng ta có thể xác định một
phân phối xác suất cho từng hyperparameter.

Đáng tiếc là random search cũng có một vài hạn chế. Thứ nhất, thuật toán không
điều chỉnh phân phối lấy mẫu dựa trên các quan sát trước đó đã thu thập. Do đó,
xác suất lấy mẫu một configuration có hiệu năng kém cũng tương đương với xác suất
lấy mẫu một configuration có hiệu năng tốt hơn. Thứ hai, cùng một lượng tài nguyên
được sử dụng cho tất cả configuration, mặc dù một số configuration có thể cho thấy
hiệu năng ban đầu kém và ít có khả năng vượt qua các configuration đã thấy trước đó.

Trong các phần tiếp theo, chúng ta sẽ tìm hiểu những thuật toán tối ưu hóa
hyperparameter hiệu quả hơn về số lượng mẫu, khắc phục các hạn chế của random search
bằng cách sử dụng một model để định hướng quá trình tìm kiếm. Chúng ta cũng sẽ tìm
hiểu các thuật toán tự động dừng quá trình đánh giá những configuration có hiệu năng
kém để tăng tốc quá trình tối ưu hóa.

## Tóm tắt

Trong phần này, chúng ta đã giới thiệu tối ưu hóa hyperparameter (HPO) và cách
diễn đạt bài toán này dưới dạng một bài toán tối ưu hóa toàn cục bằng cách xác định
một configuration space và một objective function. Chúng ta cũng đã triển khai
thuật toán HPO đầu tiên, random search, và áp dụng nó cho một bài toán phân loại
softmax đơn giản.

Mặc dù random search rất đơn giản, đây là lựa chọn tốt hơn grid search, vốn chỉ
đánh giá một tập hyperparameter cố định. Random search phần nào giảm bớt lời nguyền
chiều :cite:`bellman-science66`, và có thể hiệu quả hơn grid search rất nhiều nếu
tiêu chí phụ thuộc mạnh nhất vào một tập con nhỏ các hyperparameter.

## Bài tập

1. Trong chương này, chúng ta tối ưu validation error của một model sau khi huấn luyện trên một tập training tách biệt. Để đơn giản, code của chúng ta sử dụng `Trainer.val_dataloader`, ánh xạ tới một loader bao quanh `FashionMNIST.val`.
    1. Hãy tự kiểm chứng (bằng cách xem code) rằng điều này có nghĩa là chúng ta sử dụng tập training FashionMNIST ban đầu (60000 mẫu) để huấn luyện, và *test set* ban đầu (10000 mẫu) để validation.
    2. Tại sao cách làm này có thể gây ra vấn đề? Gợi ý: Đọc lại :numref:`sec_generalization_basics`, đặc biệt là phần về *model selection*.
    3. Thay vào đó, chúng ta nên làm gì?
2. Ở trên, chúng ta đã nêu rằng thực hiện tối ưu hóa hyperparameter bằng gradient descent là rất khó. Hãy xét một bài toán nhỏ, chẳng hạn như huấn luyện một perceptron hai lớp trên tập dữ liệu FashionMNIST (:numref:`sec_mlp-implementation`) với batch size là 256. Chúng ta muốn điều chỉnh learning rate của SGD để tối thiểu hóa một validation metric sau một epoch huấn luyện.
    1. Tại sao chúng ta không thể sử dụng validation *error* cho mục đích này? Bạn sẽ sử dụng metric nào trên validation set?
    2. Hãy phác thảo (một cách gần đúng) computational graph của validation metric sau khi huấn luyện trong một epoch. Bạn có thể giả định rằng initial weights và các hyperparameter (chẳng hạn như learning rate) là các input node của graph này. Gợi ý: Đọc lại phần về computational graph trong :numref:`sec_backprop`.
    3. Hãy ước lượng sơ bộ số lượng giá trị floating point cần lưu trữ trong một forward pass trên graph này. Gợi ý: FashionMNIST có 60000 trường hợp. Giả định rằng bộ nhớ cần thiết chủ yếu do các activation sau mỗi layer chi phối, và tra cứu độ rộng của các layer trong :numref:`sec_mlp-implementation`.
    5. Ngoài lượng compute và storage khổng lồ cần thiết, tối ưu hóa hyperparameter dựa trên gradient còn gặp những vấn đề nào khác? Gợi ý: Đọc lại phần về vanishing và exploding gradient trong :numref:`sec_numerical_stability`.
    6. *Nâng cao*: Đọc :cite:`maclaurin-icml15` để tìm hiểu một cách tiếp cận tao nhã (nhưng vẫn hơi thiếu tính thực tiễn) đối với HPO dựa trên gradient.
3. Grid search là một baseline HPO khác, trong đó chúng ta xác định một grid cách đều cho mỗi hyperparameter, sau đó lặp qua tích Cartesian (tổ hợp) của chúng để đề xuất các configuration.
    1. Ở trên, chúng ta đã nêu rằng random search có thể hiệu quả hơn grid search nhiều đối với HPO trên một số lượng hyperparameter đáng kể, nếu tiêu chí phụ thuộc mạnh nhất vào một tập con nhỏ các hyperparameter. Tại sao lại như vậy? Gợi ý: Đọc :cite:`bergstra2011algorithms`.


:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12090)
:end_tab: