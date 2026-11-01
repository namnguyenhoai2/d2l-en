```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# Tối ưu hóa siêu tham số đa độ trung thực
:label:`sec_mf_hpo`

Huấn luyện mạng nơ-ron có thể tốn kém ngay cả trên các tập dữ liệu có kích thước vừa phải.
Tùy thuộc vào không gian cấu hình (:numref:`sec_intro_config_spaces`),
tối ưu hóa siêu tham số đòi hỏi từ hàng chục đến hàng trăm lần đánh giá hàm
để tìm được một cấu hình siêu tham số hoạt động tốt. Như đã thấy trong
:numref:`sec_rs_async`, chúng ta có thể tăng tốc đáng kể tổng thời gian thực
của HPO bằng cách khai thác các tài nguyên song song, nhưng điều này không làm giảm
tổng lượng tính toán cần thiết.

Trong phần này, chúng ta sẽ chỉ ra cách có thể tăng tốc việc đánh giá các cấu hình
siêu tham số. Những phương pháp như random search phân bổ cùng một lượng
tài nguyên (ví dụ: số epoch, số điểm dữ liệu huấn luyện) cho mỗi lần đánh giá
siêu tham số. :numref:`img_samples_lc` minh họa các đường cong học của một tập mạng nơ-ron
được huấn luyện với các cấu hình siêu tham số khác nhau. Chỉ sau vài epoch, chúng ta
đã có thể phân biệt trực quan giữa các cấu hình hoạt động tốt và các cấu hình
dưới tối ưu. Tuy nhiên, các đường cong học có nhiễu, và chúng ta vẫn có thể cần
đủ 100 epoch để xác định cấu hình hoạt động tốt nhất.

![Các đường cong học của các cấu hình siêu tham số ngẫu nhiên](../img/samples_lc.svg)
:label:`img_samples_lc`

Tối ưu hóa siêu tham số đa độ trung thực phân bổ nhiều tài nguyên hơn
cho các cấu hình đầy hứa hẹn và dừng sớm việc đánh giá các cấu hình hoạt động kém.
Điều này tăng tốc quá trình tối ưu hóa, vì chúng ta có thể thử số lượng cấu hình lớn hơn
với cùng tổng lượng tài nguyên.

Trang trọng hơn, chúng ta mở rộng định nghĩa trong :numref:`sec_definition_hpo`,
sao cho hàm mục tiêu $f(\mathbf{x}, r)$ của chúng ta nhận thêm một đầu vào
$r \in [r_{\mathrm{min}}, r_{max}]$, chỉ định lượng tài nguyên mà chúng ta
sẵn sàng chi cho việc đánh giá cấu hình $\mathbf{x}$. Chúng ta giả định rằng
lỗi $f(\mathbf{x}, r)$ giảm theo $r$, trong khi chi phí
tính toán $c(\mathbf{x}, r)$ tăng. Thông thường, $r$ biểu thị số
epoch để huấn luyện mạng nơ-ron, nhưng cũng có thể là kích thước tập con huấn luyện
hoặc số fold cross-validation.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
from scipy import stats
from collections import defaultdict
d2l.set_figsize()
```

## Successive Halving
:label:`sec_mf_hpo_sh`

Một trong những cách đơn giản nhất để điều chỉnh random search cho bối cảnh
đa độ trung thực là *successive halving* :cite:`jamieson-aistats16,karnin-icml13`. Ý tưởng
cơ bản là bắt đầu với $N$ cấu hình, chẳng hạn được lấy mẫu ngẫu nhiên từ
không gian cấu hình, và chỉ huấn luyện mỗi cấu hình trong $r_{\mathrm{min}}$ epoch. Sau đó,
chúng ta loại bỏ một phần các trial hoạt động kém nhất và huấn luyện các trial
còn lại lâu hơn. Lặp lại quá trình này, số trial chạy lâu hơn sẽ ít dần, cho đến khi
ít nhất một trial đạt $r_{max}$ epoch.

Trang trọng hơn, xét ngân sách tối thiểu $r_{\mathrm{min}}$ (ví dụ: 1 epoch), ngân sách
tối đa $r_{max}$, chẳng hạn `max_epochs` trong ví dụ trước, và hằng số chia đôi
$\eta\in\{2, 3, \dots\}$. Để đơn giản, giả sử rằng
$r_{max} = r_{\mathrm{min}} \eta^K$, với $K \in \mathbb{I}$. Khi đó, số cấu hình
ban đầu là $N = \eta^K$. Hãy định nghĩa tập các rung
$\mathcal{R} = \{ r_{\mathrm{min}}, r_{\mathrm{min}}\eta, r_{\mathrm{min}}\eta^2, \dots, r_{max} \}$.

Một vòng successive halving diễn ra như sau. Chúng ta bắt đầu bằng cách chạy $N$
trial đến rung đầu tiên $r_{\mathrm{min}}$. Sau khi sắp xếp các lỗi validation, chúng ta giữ lại
phần $1 / \eta$ đứng đầu (tương ứng với $\eta^{K-1}$ cấu hình) và
loại bỏ tất cả phần còn lại. Các trial còn sống sót được huấn luyện đến rung tiếp theo
($r_{\mathrm{min}}\eta$ epoch), và quá trình được lặp lại. Tại mỗi rung,
một phần $1 / \eta$ số trial sống sót và quá trình huấn luyện của chúng tiếp tục với
ngân sách lớn hơn $\eta$ lần. Với lựa chọn $N$ cụ thể này, chỉ một
trial duy nhất sẽ được huấn luyện đến ngân sách đầy đủ $r_{max}$. Khi một vòng
successive halving hoàn tất, chúng ta bắt đầu vòng tiếp theo với một tập cấu hình
ban đầu mới, lặp lại cho đến khi tổng ngân sách được sử dụng hết.

![Các đường cong học của các cấu hình siêu tham số ngẫu nhiên.](../img/sh.svg)

Chúng ta kế thừa lớp cơ sở `HPOScheduler` từ :numref:`sec_api_hpo` để
triển khai successive halving, cho phép một đối tượng `HPOSearcher` tổng quát
lấy mẫu các cấu hình (trong ví dụ dưới đây của chúng ta, đối tượng này sẽ là một `RandomSearcher`).
Ngoài ra, người dùng phải truyền vào tài nguyên tối thiểu $r_{\mathrm{min}}$, tài nguyên tối đa
$r_{max}$ và $\eta$. Bên trong scheduler, chúng ta duy trì một
hàng đợi các cấu hình vẫn cần được đánh giá cho rung hiện tại $r_i$.
Chúng ta cập nhật hàng đợi mỗi khi chuyển sang rung tiếp theo.

```{.python .input  n=2}
class SuccessiveHalvingScheduler(d2l.HPOScheduler):  #@save
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.save_hyperparameters()
        # Compute K, which is later used to determine the number of configurations
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # Define the rungs
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            # The final rung should be r_max
            self.rung_levels.append(r_max)
            self.K += 1
        # Bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        # Our processing queue
        self.queue = []
```

Ban đầu, hàng đợi của chúng ta trống, và chúng ta điền vào đó
$n = \textrm{prefact} \cdot \eta^{K}$ cấu hình, trước tiên được đánh giá tại
rung nhỏ nhất $r_{\mathrm{min}}$. Ở đây, $\textrm{prefact}$ cho phép chúng ta tái sử dụng
mã trong một ngữ cảnh khác. Trong phần này, chúng ta cố định
$\textrm{prefact} = 1$. Mỗi khi tài nguyên trở nên khả dụng và đối tượng `HPOTuner`
truy vấn hàm `suggest`, chúng ta trả về một phần tử từ hàng đợi. Khi
hoàn thành một vòng successive halving, nghĩa là chúng ta đã đánh giá tất cả
cấu hình còn sống sót ở mức tài nguyên cao nhất $r_{max}$ và hàng đợi
của chúng ta trống, chúng ta lại bắt đầu toàn bộ quá trình với một tập cấu hình
mới được lấy mẫu ngẫu nhiên.

```{.python .input  n=12}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def suggest(self):
    if len(self.queue) == 0:
        # Start a new round of successive halving
        # Number of configurations for the first rung:
        n0 = int(self.prefact * self.eta ** self.K)
        for _ in range(n0):
            config = self.searcher.sample_configuration()
            config["max_epochs"] = self.r_min  # Set r = r_min
            self.queue.append(config)
    # Return an element from the queue
    return self.queue.pop()
```

Khi thu thập một điểm dữ liệu mới, trước tiên chúng ta cập nhật mô-đun searcher.
Sau đó, chúng ta kiểm tra xem đã thu thập đủ tất cả điểm dữ liệu tại rung hiện tại hay chưa.
Nếu có, chúng ta sắp xếp tất cả cấu hình và đưa $\frac{1}{\eta}$
cấu hình đứng đầu vào hàng đợi.

```{.python .input  n=4}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def update(self, config: dict, error: float, info=None):
    ri = int(config["max_epochs"])  # Rung r_i
    # Update our searcher, e.g if we use Bayesian optimization later
    self.searcher.update(config, error, additional_info=info)
    self.all_observed_error_at_rungs[ri].append((config, error))
    if ri < self.r_max:
        # Bookkeeping
        self.observed_error_at_rungs[ri].append((config, error))
        # Determine how many configurations should be evaluated on this rung
        ki = self.K - self.rung_levels.index(ri)
        ni = int(self.prefact * self.eta ** ki)
        # If we observed all configuration on this rung r_i, we estimate the
        # top 1 / eta configuration, add them to queue and promote them for
        # the next rung r_{i+1}
        if len(self.observed_error_at_rungs[ri]) >= ni:
            kiplus1 = ki - 1
            niplus1 = int(self.prefact * self.eta ** kiplus1)
            best_performing_configurations = self.get_top_n_configurations(
                rung_level=ri, n=niplus1
            )
            riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
            # Queue may not be empty: insert new entries at the beginning
            self.queue = [
                dict(config, max_epochs=riplus1)
                for config in best_performing_configurations
            ] + self.queue
            self.observed_error_at_rungs[ri] = []  # Reset
```

Các cấu hình được sắp xếp dựa trên hiệu năng quan sát được của chúng tại rung hiện tại.

```{.python .input  n=4}
%%tab pytorch

@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def get_top_n_configurations(self, rung_level, n):
    rung = self.observed_error_at_rungs[rung_level]
    if not rung:
        return []
    sorted_rung = sorted(rung, key=lambda x: x[1])
    return [x[0] for x in sorted_rung[:n]]
```

Hãy xem successive halving hoạt động thế nào trên ví dụ mạng nơ-ron của chúng ta. Chúng ta
sẽ sử dụng $r_{\mathrm{min}} = 2$, $\eta = 2$, $r_{max} = 10$, để các mức rung là
$2, 4, 8, 10$.

```{.python .input  n=5}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2
num_gpus=1

config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Chúng ta chỉ cần thay scheduler bằng `SuccessiveHalvingScheduler` mới.

```{.python .input  n=14}
searcher = d2l.RandomSearcher(config_space, initial_config=initial_config)
scheduler = SuccessiveHalvingScheduler(
    searcher=searcher,
    eta=eta,
    r_min=min_number_of_epochs,
    r_max=max_number_of_epochs,
)
tuner = d2l.HPOTuner(
    scheduler=scheduler,
    objective=d2l.hpo_objective_lenet,
)
tuner.run(number_of_trials=30)
```

Chúng ta có thể trực quan hóa các đường cong học của tất cả cấu hình đã đánh giá.
Hầu hết cấu hình bị dừng sớm và chỉ các cấu hình hoạt động tốt hơn
mới sống sót đến $r_{max}$. Hãy so sánh điều này với random search thông thường,
vốn sẽ phân bổ $r_{max}$ cho mọi cấu hình.

```{.python .input  n=19}
for rung_index, rung in scheduler.all_observed_error_at_rungs.items():
    errors = [xi[1] for xi in rung]
    d2l.plt.scatter([rung_index] * len(errors), errors)
d2l.plt.xlim(min_number_of_epochs - 0.5, max_number_of_epochs + 0.5)
d2l.plt.xticks(
    np.arange(min_number_of_epochs, max_number_of_epochs + 1),
    np.arange(min_number_of_epochs, max_number_of_epochs + 1)
)
d2l.plt.ylabel("validation error")
d2l.plt.xlabel("epochs")
```

Cuối cùng, hãy lưu ý một chút phức tạp trong cách triển khai
`SuccessiveHalvingScheduler` của chúng ta. Giả sử một worker rảnh để chạy một job, và
`suggest` được gọi khi rung hiện tại gần như đã được lấp đầy hoàn toàn, nhưng
một worker khác vẫn đang bận đánh giá. Vì thiếu giá trị metric từ worker này,
chúng ta không thể xác định phần $1 / \eta$ đứng đầu để mở rung tiếp theo.
Mặt khác, chúng ta muốn giao một job cho worker đang rảnh để nó không bị nhàn rỗi.
Giải pháp của chúng ta là bắt đầu một vòng successive halving mới và giao
worker cho trial đầu tiên trong vòng đó. Tuy nhiên, khi một rung được hoàn thành trong
`update`, chúng ta đảm bảo chèn các cấu hình mới vào đầu hàng đợi,
để chúng được ưu tiên hơn các cấu hình từ vòng tiếp theo.

## Tóm tắt

Trong phần này, chúng ta đã giới thiệu khái niệm tối ưu hóa siêu tham số đa độ
trung thực, trong đó chúng ta giả định có quyền truy cập vào các xấp xỉ của hàm mục tiêu
có chi phí đánh giá thấp, chẳng hạn lỗi validation sau một số epoch huấn luyện nhất định
được dùng làm đại diện cho lỗi validation sau toàn bộ số epoch.
Tối ưu hóa siêu tham số đa độ trung thực giúp giảm tổng lượng tính toán của HPO
thay vì chỉ giảm thời gian thực.

Chúng ta đã triển khai và đánh giá successive halving, một thuật toán HPO đa độ
trung thực đơn giản nhưng hiệu quả.


:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12094)
:end_tab: