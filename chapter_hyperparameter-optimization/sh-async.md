```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# Successive Halving bất đồng bộ

:label:`sec_sh_async`

Như đã thấy trong :numref:`sec_rs_async`, chúng ta có thể tăng tốc HPO bằng cách
phân phối việc đánh giá các cấu hình hyperparameter trên nhiều instance hoặc
nhiều CPU / GPU trên một instance duy nhất. Tuy nhiên, so với random search,
việc chạy successive halving (SH) bất đồng bộ trong môi trường phân tán không
đơn giản. Trước khi quyết định cấu hình nào sẽ chạy tiếp theo, trước hết chúng ta
phải thu thập tất cả các quan sát ở mức rung hiện tại. Điều này yêu cầu
đồng bộ hóa các worker ở mỗi mức rung. Ví dụ, đối với mức rung thấp nhất
$r_{\mathrm{min}}$, trước hết chúng ta phải đánh giá tất cả $N = \eta^K$ cấu hình, rồi mới có thể
đưa $\frac{1}{\eta}$ cấu hình trong số đó lên mức rung tiếp theo.

Trong mọi hệ thống phân tán, việc đồng bộ hóa thường khiến các worker phải chờ.
Trước hết, chúng ta thường quan sát thấy thời gian training giữa các cấu hình
hyperparameter khác nhau đáng kể. Ví dụ, giả sử số lượng filter trên mỗi layer là
một hyperparameter, khi đó các network có ít filter hơn sẽ hoàn tất training nhanh
hơn các network có nhiều filter hơn, dẫn đến thời gian worker nhàn rỗi do các worker
chậm (straggler). Ngoài ra, số lượng slot ở một mức rung không phải lúc nào cũng
là bội số của số worker; trong trường hợp đó, một số worker thậm chí có thể
nhàn rỗi trong cả một batch.

Hình :numref:`synchronous_sh` minh họa việc lập lịch SH đồng bộ với $\eta=2$
cho bốn trial khác nhau trên hai worker. Chúng ta bắt đầu bằng việc đánh giá Trial-0 và
Trial-1 trong một epoch, rồi ngay lập tức tiếp tục với hai trial tiếp theo khi chúng
hoàn tất. Trước hết, chúng ta phải đợi Trial-2 hoàn tất; trial này mất nhiều thời gian
hơn đáng kể so với các trial khác, trước khi có thể đưa hai trial tốt nhất, tức là
Trial-0 và Trial-3, lên mức rung tiếp theo. Điều này khiến Worker-1 nhàn rỗi. Sau đó,
chúng ta tiếp tục với Rung 1. Ở đây, Trial-3 cũng mất nhiều thời gian hơn Trial-0,
dẫn đến thêm thời gian nhàn rỗi của Worker-0. Khi đạt đến Rung-2, chỉ còn trial tốt
nhất là Trial-0, và trial này chỉ chiếm một worker. Để tránh việc Worker-1 nhàn rỗi
trong thời gian đó, hầu hết các implementation của SH đã tiếp tục vòng tiếp theo
và bắt đầu đánh giá các trial mới (ví dụ Trial-4) ở rung đầu tiên.

![Successive halving đồng bộ với hai worker.](../img/sync_sh.svg)
:label:`synchronous_sh`

Successive halving bất đồng bộ (ASHA) :cite:`li-arxiv18` điều chỉnh SH cho
kịch bản parallel bất đồng bộ. Ý tưởng chính của ASHA là đưa các cấu hình lên mức
rung tiếp theo ngay khi chúng ta thu thập được ít nhất $\eta$ quan sát ở mức rung
hiện tại. Quy tắc quyết định này có thể dẫn đến các lần promotion chưa tối ưu:
các cấu hình có thể được đưa lên mức rung tiếp theo dù xét lại sau đó, chúng không
có kết quả tốt bằng phần lớn các cấu hình khác ở cùng mức rung. Mặt khác, bằng cách
này chúng ta loại bỏ hoàn toàn mọi điểm đồng bộ hóa. Trong thực tế, những lần promotion
ban đầu chưa tối ưu như vậy chỉ ảnh hưởng vừa phải đến hiệu năng, không chỉ vì thứ
hạng của các cấu hình hyperparameter thường khá nhất quán giữa các mức rung, mà còn
vì các rung tăng dần theo thời gian và ngày càng phản ánh tốt hơn phân phối các giá trị
metric ở mức đó. Nếu một worker rảnh nhưng không có cấu hình nào có thể được promotion,
chúng ta bắt đầu một cấu hình mới với $r = r_{\mathrm{min}}$, tức là mức rung đầu tiên.

:numref:`asha` minh họa việc lập lịch cho cùng các cấu hình bằng ASHA. Khi Trial-1
hoàn tất, chúng ta thu thập kết quả của hai trial (tức Trial-0 và Trial-1) và
ngay lập tức đưa trial tốt hơn trong số đó (Trial-0) lên mức rung tiếp theo. Sau khi
Trial-0 hoàn tất ở rung 1, có quá ít trial ở đó để thực hiện promotion tiếp theo.
Do đó, chúng ta tiếp tục với rung 0 và đánh giá Trial-3. Khi Trial-3 hoàn tất,
Trial-2 vẫn đang chờ. Tại thời điểm này, chúng ta đã đánh giá 3 trial ở rung 0 và
một trial ở rung 1. Vì Trial-3 có kết quả kém hơn Trial-0 ở rung 0, đồng thời
$\eta=2$, chúng ta chưa thể promotion thêm trial nào, nên Worker-1 bắt đầu Trial-4
từ đầu. Tuy nhiên, khi Trial-2 hoàn tất và có điểm thấp hơn Trial-3, Trial-3 được
đưa lên rung 1. Sau đó, chúng ta đã thu thập 2 kết quả đánh giá ở rung 1, nghĩa là
bây giờ có thể đưa Trial-0 lên rung 2. Đồng thời, Worker-1 tiếp tục đánh giá các
trial mới (tức Trial-5) ở rung 0.


![Successive halving bất đồng bộ (ASHA) với hai worker.](../img/asha.svg)
:label:`asha`

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## Hàm mục tiêu

Chúng ta sẽ sử dụng *Syne Tune* với cùng hàm mục tiêu như trong
:numref:`sec_rs_async`.

```{.python .input  n=54}
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l
    from syne_tune import Reporter

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter()
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Initialize the state of Trainer
            trainer.fit(model=model, data=data)
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

Chúng ta cũng sẽ sử dụng cùng không gian cấu hình như trước:

```{.python .input  n=55}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2

config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": max_number_of_epochs,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## Scheduler bất đồng bộ

Trước tiên, chúng ta xác định số lượng worker đánh giá các trial đồng thời. Chúng ta
cũng cần chỉ định thời gian muốn chạy random search bằng cách đặt giới hạn trên
cho tổng thời gian theo đồng hồ thực.

```{.python .input  n=56}
n_workers = 2  # Needs to be <= the number of available GPUs
max_wallclock_time = 12 * 60  # 12 minutes
```

Code để chạy ASHA là một biến thể đơn giản của phần chúng ta đã thực hiện cho
random search bất đồng bộ.

```{.python .input  n=56}
mode = "min"
metric = "validation_error"
resource_attr = "epoch"

scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
    max_resource_attr="max_epochs",
    resource_attr=resource_attr,
    grace_period=min_number_of_epochs,
    reduction_factor=eta,
)
```

Ở đây, `metric` và `resource_attr` chỉ định tên các key được sử dụng với
callback `report`, còn `max_resource_attr` cho biết input nào của hàm mục tiêu
tương ứng với $r_{\mathrm{max}}$. Ngoài ra, `grace_period` cung cấp $r_{\mathrm{min}}$, và
`reduction_factor` là $\eta$. Chúng ta có thể chạy Syne Tune như trước (quá trình này sẽ
mất khoảng 12 phút):

```{.python .input  n=57}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)

stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
tuner.run()
```

Lưu ý rằng chúng ta đang chạy một biến thể của ASHA, trong đó các trial có hiệu năng
kém sẽ bị dừng sớm. Điều này khác với implementation của chúng ta trong
:numref:`sec_mf_hpo_sh`, nơi mỗi training job được bắt đầu với một
`max_epochs` cố định. Trong trường hợp sau, một trial có hiệu năng tốt và đạt đủ
10 epoch trước hết phải training 1, rồi 2, rồi 4, rồi 8 epoch, mỗi lần đều bắt đầu
từ đầu. Kiểu lập lịch pause-and-resume này có thể được triển khai hiệu quả bằng cách
checkpoint trạng thái training sau mỗi epoch, nhưng ở đây chúng ta tránh sự phức tạp
bổ sung đó. Sau khi experiment hoàn tất, chúng ta có thể truy xuất và vẽ biểu đồ
các kết quả.

```{.python .input  n=59}
d2l.set_figsize()
e = load_experiment(tuner.name)
e.plot()
```

## Trực quan hóa quá trình tối ưu hóa

Một lần nữa, chúng ta trực quan hóa learning curve của mọi trial (mỗi màu trong biểu đồ
đại diện cho một trial). Hãy so sánh với random search bất đồng bộ trong
:numref:`sec_rs_async`. Như đã thấy với successive halving trong
:numref:`sec_mf_hpo`, phần lớn các trial bị dừng ở 1 hoặc 2 epoch
($r_{\mathrm{min}}$ hoặc $\eta * r_{\mathrm{min}}$). Tuy nhiên, các trial không dừng
tại cùng một thời điểm, vì chúng cần lượng thời gian khác nhau cho mỗi epoch. Nếu
chạy successive halving tiêu chuẩn thay vì ASHA, chúng ta sẽ cần đồng bộ hóa
các worker trước khi có thể đưa các cấu hình lên mức rung tiếp theo.

```{.python .input  n=60}
d2l.set_figsize([6, 2.5])
results = e.results
for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
d2l.plt.xlabel("wall-clock time")
d2l.plt.ylabel("objective function")
```

## Tóm tắt

So với random search, successive halving không hoàn toàn đơn giản khi chạy trong
môi trường phân tán bất đồng bộ. Để tránh các điểm đồng bộ hóa, chúng ta promotion
các cấu hình lên mức rung tiếp theo nhanh nhất có thể, ngay cả khi điều đó đồng nghĩa
với việc promotion nhầm một số cấu hình. Trong thực tế, điều này thường không gây
ảnh hưởng lớn, và lợi ích của việc lập lịch bất đồng bộ so với đồng bộ thường cao hơn
nhiều so với tổn thất do các quyết định chưa tối ưu.


:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12101)
:end_tab: