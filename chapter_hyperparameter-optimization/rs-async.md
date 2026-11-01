```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# Tìm kiếm ngẫu nhiên bất đồng bộ
:label:`sec_rs_async`

Như chúng ta đã thấy trong :numref:`sec_api_hpo` trước đó, chúng ta có thể phải chờ
hàng giờ hoặc thậm chí hàng ngày trước khi tìm kiếm ngẫu nhiên trả về một cấu hình
hyperparameter tốt, vì việc đánh giá các cấu hình hyperparameter rất tốn kém. Trong thực tế,
chúng ta thường có quyền truy cập vào một nhóm tài nguyên như nhiều GPU trên cùng một máy
hoặc nhiều máy, mỗi máy có một GPU. Điều này dẫn đến câu hỏi: *Làm thế nào để phân phối
tìm kiếm ngẫu nhiên một cách hiệu quả?*

Nhìn chung, chúng ta phân biệt giữa tối ưu hóa hyperparameter song song đồng bộ và bất đồng
bộ (xem :numref:`distributed_scheduling`). Trong thiết lập đồng bộ, chúng ta chờ tất cả các
trial đang chạy đồng thời hoàn tất trước khi bắt đầu batch tiếp theo. Hãy xét các không gian
cấu hình chứa những hyperparameter như số lượng filter hoặc số lượng layer của một mạng
neural sâu. Các cấu hình hyperparameter chứa số lượng layer hoặc filter lớn hơn đương nhiên
sẽ mất nhiều thời gian hơn để hoàn tất, và tất cả trial khác trong cùng batch sẽ phải chờ
tại các điểm đồng bộ hóa (vùng màu xám trong :numref:`distributed_scheduling`) trước khi chúng
ta có thể tiếp tục quá trình tối ưu hóa.

Trong thiết lập bất đồng bộ, chúng ta lập tức lên lịch cho một trial mới ngay khi tài nguyên
khả dụng. Điều này giúp tận dụng tối ưu tài nguyên, vì chúng ta có thể tránh mọi overhead
đồng bộ hóa. Đối với tìm kiếm ngẫu nhiên, mỗi cấu hình hyperparameter mới được chọn độc lập
với tất cả cấu hình khác, cụ thể là không tận dụng các quan sát từ bất kỳ lần đánh giá trước
đó nào. Điều này có nghĩa là chúng ta có thể dễ dàng song song hóa tìm kiếm ngẫu nhiên theo
cách bất đồng bộ. Điều này không đơn giản với các phương pháp tinh vi hơn, vốn đưa ra quyết
định dựa trên các quan sát trước đó (xem :numref:`sec_sh_async`). Mặc dù cần truy cập
nhiều tài nguyên hơn so với thiết lập tuần tự, tìm kiếm ngẫu nhiên bất đồng bộ vẫn cho tốc độ
tăng tuyến tính: một mức hiệu năng nhất định sẽ đạt được nhanh hơn $K$ lần nếu có thể chạy
song song $K$ trial.


![Phân phối quá trình tối ưu hóa hyperparameter theo cách đồng bộ hoặc bất đồng bộ. So với thiết lập tuần tự, chúng ta có thể giảm tổng thời gian thực tế trong khi vẫn giữ nguyên tổng lượng tính toán. Lập lịch đồng bộ có thể khiến các worker không hoạt động khi có các tác vụ chậm.](../img/distributed_scheduling.svg)
:label:`distributed_scheduling`

Trong notebook này, chúng ta sẽ xem xét tìm kiếm ngẫu nhiên bất đồng bộ, trong đó các trial
được thực thi trong nhiều tiến trình Python trên cùng một máy. Việc tự triển khai từ đầu chức
năng lập lịch và thực thi job phân tán rất khó. Chúng ta sẽ sử dụng *Syne Tune*
:cite:`salinas-automl22`, thư viện cung cấp cho chúng ta một interface đơn giản để thực
hiện HPO bất đồng bộ. Syne Tune được thiết kế để chạy với nhiều execution back-end khác nhau,
và bạn đọc quan tâm có thể tìm hiểu các API đơn giản của thư viện để hiểu thêm về HPO phân tán.

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## Hàm mục tiêu

Trước tiên, chúng ta phải định nghĩa một hàm mục tiêu mới để giờ đây hàm này trả về hiệu năng
cho Syne Tune thông qua callback `report`.

```{.python .input  n=34}
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

Lưu ý rằng `PythonBackend` của Syne Tune yêu cầu các dependency phải được import
bên trong phần định nghĩa hàm.

## Bộ lập lịch bất đồng bộ

Trước tiên, chúng ta xác định số lượng worker đánh giá các trial đồng thời. Chúng ta cũng
cần chỉ định thời gian muốn chạy tìm kiếm ngẫu nhiên bằng cách định nghĩa giới hạn trên cho
tổng thời gian thực tế.

```{.python .input  n=37}
n_workers = 2  # Needs to be <= the number of available GPUs

max_wallclock_time = 12 * 60  # 12 minutes
```

Tiếp theo, chúng ta chỉ rõ metric muốn tối ưu và liệu muốn tối thiểu hóa hay tối đa hóa
metric này. Cụ thể, `metric` phải tương ứng với tên đối số được truyền cho
callback `report`.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

Chúng ta sử dụng không gian cấu hình từ ví dụ trước. Trong Syne Tune, dictionary này cũng
có thể được dùng để truyền các thuộc tính hằng số cho training script. Chúng ta tận dụng
tính năng này để truyền `max_epochs`. Ngoài ra, chúng ta chỉ định cấu hình đầu
tiên cần được đánh giá trong `initial_config`.

```{.python .input  n=39}
config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": 10,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Tiếp theo, chúng ta cần chỉ định back-end để thực thi job. Ở đây, chúng ta chỉ xét việc
phân phối trên một máy cục bộ, nơi các job song song được thực thi dưới dạng các sub-process.
Tuy nhiên, đối với HPO quy mô lớn, chúng ta cũng có thể chạy việc này trên một cluster hoặc
môi trường cloud, nơi mỗi trial sử dụng toàn bộ một instance.

```{.python .input  n=40}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)
```

Bây giờ chúng ta có thể tạo bộ lập lịch cho tìm kiếm ngẫu nhiên bất đồng bộ, có hành vi
tương tự `BasicScheduler` của chúng ta trong :numref:`sec_api_hpo`.

```{.python .input  n=41}
scheduler = RandomSearch(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
)
```

Syne Tune cũng cung cấp một `Tuner`, trong đó vòng lặp experiment chính và
việc bookkeeping được tập trung, còn các tương tác giữa scheduler và back-end được trung gian
hóa.

```{.python .input  n=42}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler, 
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
```

Bây giờ hãy chạy experiment HPO phân tán. Theo tiêu chí dừng của chúng ta, experiment sẽ
chạy trong khoảng 12 phút.

```{.python .input  n=43}
tuner.run()
```

Log của tất cả cấu hình hyperparameter đã được đánh giá được lưu lại để phân tích sau này.
Vào bất kỳ thời điểm nào trong quá trình tuning job, chúng ta đều có thể dễ dàng lấy các kết
quả đạt được cho đến lúc đó và vẽ quỹ đạo incumbent.

```{.python .input  n=46}
d2l.set_figsize()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Trực quan hóa quá trình tối ưu hóa bất đồng bộ

Dưới đây, chúng ta trực quan hóa cách các learning curve của từng trial (mỗi màu trong biểu
đồ biểu thị một trial) thay đổi trong quá trình tối ưu hóa bất đồng bộ. Tại bất kỳ thời điểm
nào, có số trial đang chạy đồng thời bằng với số worker. Khi một trial hoàn tất, chúng ta
ngay lập tức bắt đầu trial tiếp theo mà không chờ các trial khác hoàn tất. Lập lịch bất đồng
bộ giảm thời gian nhàn rỗi của worker xuống mức tối thiểu.

```{.python .input  n=45}
d2l.set_figsize([6, 2.5])
results = tuning_experiment.results

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

Chúng ta có thể giảm đáng kể thời gian chờ của tìm kiếm ngẫu nhiên bằng cách phân phối các
trial trên những tài nguyên song song. Nhìn chung, chúng ta phân biệt giữa lập lịch đồng bộ
và lập lịch bất đồng bộ. Lập lịch đồng bộ có nghĩa là chúng ta lấy mẫu một batch cấu hình
hyperparameter mới sau khi batch trước đó hoàn tất. Nếu có một tác vụ chậm — tức là trial mất
nhiều thời gian hoàn tất hơn các trial khác — worker của chúng ta phải chờ tại các điểm đồng
bộ hóa. Lập lịch bất đồng bộ đánh giá một cấu hình hyperparameter mới ngay khi tài nguyên khả
dụng và do đó đảm bảo tất cả worker đều bận tại mọi thời điểm. Mặc dù tìm kiếm ngẫu nhiên dễ
được phân phối theo cách bất đồng bộ và không yêu cầu thay đổi thuật toán thực tế, các phương
pháp khác cần một số điều chỉnh bổ sung.

## Bài tập

1. Xét model `DropoutMLP` được triển khai trong :numref:`sec_dropout` và
   được sử dụng trong Bài tập 1 của :numref:`sec_api_hpo`.
    1. Triển khai một hàm mục tiêu `hpo_objective_dropoutmlp_synetune` để sử dụng với Syne Tune. Đảm bảo
       rằng hàm của bạn báo cáo lỗi validation sau mỗi epoch.
    2. Sử dụng thiết lập của Bài tập 1 trong :numref:`sec_api_hpo`, hãy so sánh
       tìm kiếm ngẫu nhiên với tối ưu hóa Bayesian. Nếu sử dụng SageMaker, bạn có thể sử dụng
       các tiện ích benchmarking của Syne Tune để chạy các experiment song song. Gợi ý: Tối
       ưu hóa Bayesian được cung cấp dưới dạng `syne_tune.optimizer.baselines.BayesianOptimization`.
    3. Đối với bài tập này, bạn cần chạy trên một instance có ít nhất 4 CPU core. Với một
       trong các phương pháp được sử dụng ở trên (tìm kiếm ngẫu nhiên, tối ưu hóa Bayesian),
       hãy chạy các experiment với `n_workers=1`, `n_workers=2`,
       `n_workers=4` và so sánh kết quả (các quỹ đạo incumbent). Ít nhất đối với tìm
       kiếm ngẫu nhiên, bạn sẽ quan sát thấy khả năng mở rộng tuyến tính theo số lượng worker.
       Gợi ý: Để có kết quả đáng tin cậy, bạn có thể cần lấy trung bình qua nhiều lần lặp cho
       mỗi trường hợp.
2. *Nâng cao*. Mục tiêu của bài tập này là triển khai một scheduler mới trong Syne Tune.
    1. Tạo một virtual environment chứa cả mã nguồn [d2lbook](https://github.com/d2l-ai/d2l-en/blob/master/INFO.md#installation-for-developers) và [syne-tune](https://syne-tune.readthedocs.io/en/latest/getting_started.html).
    2. Triển khai `LocalSearcher` từ Bài tập 2 trong :numref:`sec_api_hpo`
       như một searcher mới trong Syne Tune. Gợi ý: Đọc [hướng dẫn này](https://syne-tune.readthedocs.io/en/latest/tutorials/developer/README.html). Ngoài ra, bạn có thể làm theo [ví dụ](https://syne-tune.readthedocs.io/en/latest/examples.html#launch-hpo-experiment-with-home-made-scheduler) này.
    3. So sánh `LocalSearcher` mới của bạn với `RandomSearch` trên benchmark
       `DropoutMLP`.


:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12093)
:end_tab: