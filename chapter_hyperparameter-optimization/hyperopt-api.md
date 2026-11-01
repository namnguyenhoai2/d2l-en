```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# API Tối ưu hóa Siêu tham số
:label:`sec_api_hpo`

Trước khi đi sâu vào phương pháp, trước hết chúng ta sẽ thảo luận về một cấu trúc
mã cơ bản cho phép triển khai hiệu quả nhiều thuật toán HPO. Nhìn chung, tất cả
các thuật toán HPO được xem xét ở đây cần triển khai hai primitive ra quyết định,
*searching* và *scheduling*. Thứ nhất, chúng cần lấy mẫu các cấu hình siêu tham
số mới, thường liên quan đến một dạng tìm kiếm trong không gian cấu hình. Thứ
hai, với mỗi cấu hình, một thuật toán HPO cần lập lịch cho việc đánh giá cấu hình
đó và quyết định lượng tài nguyên cần phân bổ cho nó. Khi bắt đầu đánh giá một
cấu hình, chúng ta sẽ gọi cấu hình đó là một *trial*. Chúng ta ánh xạ các quyết
định này vào hai lớp, `HPOSearcher` và `HPOScheduler`. Ngoài ra,
chúng ta cũng cung cấp một lớp `HPOTuner` thực thi quá trình tối ưu hóa.

Khái niệm scheduler và searcher cũng được triển khai trong các thư viện HPO phổ
biến, chẳng hạn như Syne Tune :cite:`salinas-automl22`, Ray Tune
:cite:`liaw-arxiv18` hoặc Optuna :cite:`akiba-sigkdd19`.

```{.python .input  n=2}
%%tab pytorch
import time
from d2l import torch as d2l
from scipy import stats
```

## Searcher

Dưới đây, chúng ta định nghĩa một lớp cơ sở cho các searcher. Lớp này cung cấp
một cấu hình ứng viên mới thông qua hàm `sample_configuration`. Một cách đơn giản
để triển khai hàm này là lấy mẫu các cấu hình đồng đều một cách ngẫu nhiên, như
chúng ta đã làm với random search trong :numref:`sec_what_is_hpo`. Các thuật
toán phức tạp hơn, chẳng hạn như Bayesian optimization, sẽ đưa ra các quyết định
này dựa trên hiệu năng của các trial trước đó. Nhờ vậy, các thuật toán này có thể
lấy mẫu những ứng viên tiềm năng hơn theo thời gian. Chúng ta thêm hàm
`update` để cập nhật lịch sử của các trial trước đó, sau đó có thể
khai thác lịch sử này để cải thiện phân phối lấy mẫu.

```{.python .input  n=3}
%%tab pytorch
class HPOSearcher(d2l.HyperParameters):  #@save
    def sample_configuration() -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        pass
```

Đoạn mã sau đây cho thấy cách triển khai random search optimizer từ phần trước
trong API này. Như một mở rộng nhỏ, chúng ta cho phép người dùng chỉ định cấu
hình đầu tiên sẽ được đánh giá thông qua `initial_config`, còn các cấu hình
tiếp theo sẽ được lấy ngẫu nhiên.

```{.python .input  n=4}
%%tab pytorch
class RandomSearcher(HPOSearcher):  #@save
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result
```

## Scheduler

Ngoài việc lấy mẫu các cấu hình cho những trial mới, chúng ta cũng cần quyết định
khi nào và trong bao lâu sẽ chạy một trial. Trên thực tế, tất cả các quyết định
này đều được đưa ra bởi `HPOScheduler`, thành phần này ủy quyền việc lựa
chọn các cấu hình mới cho một `HPOSearcher`. Phương thức
`suggest` được gọi bất cứ khi nào có một phần tài nguyên dành cho việc
training trở nên khả dụng. Ngoài việc gọi `sample_configuration` của một searcher,
phương thức này cũng có thể quyết định các tham số như `max_epochs` (tức
là thời gian training model). Phương thức `update` được gọi bất cứ
khi nào một trial trả về một observation mới.

```{.python .input  n=5}
%%tab pytorch
class HPOScheduler(d2l.HyperParameters):  #@save
    def suggest(self) -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
```

Để triển khai random search cũng như các thuật toán HPO khác, chúng ta chỉ cần
một scheduler cơ bản, có nhiệm vụ lập lịch một cấu hình mới mỗi khi có tài nguyên
mới trở nên khả dụng.

```{.python .input  n=6}
%%tab pytorch
class BasicScheduler(HPOScheduler):  #@save
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)
```

## Tuner

Cuối cùng, chúng ta cần một component chạy scheduler/searcher và thực hiện một
số công việc ghi nhận kết quả. Đoạn mã sau triển khai việc thực thi tuần tự các
trial HPO, trong đó mỗi training job được đánh giá sau training job trước đó, và
sẽ đóng vai trò là một ví dụ cơ bản. Về sau, chúng ta sẽ sử dụng *Syne Tune* cho
các trường hợp HPO phân tán có khả năng mở rộng tốt hơn.

```{.python .input  n=7}
%%tab pytorch
class HPOTuner(d2l.HyperParameters):  #@save
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        # Bookeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")
            error = self.objective(**config)
            error = float(d2l.numpy(error.cpu()))
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")
```

## Ghi nhận Hiệu năng của các Thuật toán HPO

Với bất kỳ thuật toán HPO nào, mối quan tâm chính của chúng ta là cấu hình có
hiệu năng tốt nhất (gọi là *incumbent*) và validation error của nó sau một
khoảng thời gian wall-clock nhất định. Vì vậy, chúng ta theo dõi
`runtime` trên mỗi iteration. Giá trị này bao gồm cả thời gian chạy
một evaluation (lần gọi `objective`) và thời gian đưa ra quyết định
(lần gọi `scheduler.suggest`). Trong phần tiếp theo, chúng ta sẽ biểu diễn
`cumulative_runtime` theo `incumbent_trajectory` để trực quan hóa
*any-time performance* của thuật toán HPO, được xác định theo
`scheduler` (và `searcher`). Điều này cho phép chúng ta
định lượng không chỉ mức độ hiệu quả của cấu hình do optimizer tìm được, mà còn
cả tốc độ optimizer tìm ra cấu hình đó.

```{.python .input  n=8}
%%tab pytorch
@d2l.add_to_class(HPOTuner)  #@save
def bookkeeping(self, config: dict, error: float, runtime: float):
    self.records.append({"config": config, "error": error, "runtime": runtime})
    # Check if the last hyperparameter configuration performs better 
    # than the incumbent
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
    # Add current best observed performance to the optimization trajectory
    self.incumbent_trajectory.append(self.incumbent_error)
    # Update runtime
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

## Ví dụ: Tối ưu hóa các Siêu tham số của Mạng nơ-ron tích chập

Bây giờ, chúng ta sử dụng triển khai random search mới để tối ưu hóa
*batch size* và *learning rate* của mạng nơ-ron tích chập
`LeNet` trong :numref:`sec_lenet`. Chúng ta bắt đầu bằng
việc định nghĩa objective function, một lần nữa sẽ là validation error.

```{.python .input  n=9}
%%tab pytorch
def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):  #@save
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error
```

Chúng ta cũng cần định nghĩa không gian cấu hình. Ngoài ra, cấu hình đầu tiên
được đánh giá là thiết lập mặc định được sử dụng trong :numref:`sec_lenet`.

```{.python .input  n=10}
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Bây giờ chúng ta có thể bắt đầu random search:

```{.python .input}
searcher = RandomSearcher(config_space, initial_config=initial_config)
scheduler = BasicScheduler(searcher=searcher)
tuner = HPOTuner(scheduler=scheduler, objective=hpo_objective_lenet)
tuner.run(number_of_trials=5)
```

Dưới đây, chúng ta biểu diễn quỹ đạo tối ưu hóa của incumbent để thu được
any-time performance của random search:

```{.python .input  n=11}
board = d2l.ProgressBoard(xlabel="time", ylabel="error")
for time_stamp, error in zip(
    tuner.cumulative_runtime, tuner.incumbent_trajectory
):
    board.draw(time_stamp, error, "random search", every_n=1)
```

## So sánh các Thuật toán HPO

Cũng như với các thuật toán training hoặc kiến trúc model, việc hiểu cách so
sánh tốt nhất các thuật toán HPO khác nhau là rất quan trọng. Mỗi lần chạy HPO
phụ thuộc vào hai nguồn ngẫu nhiên chính: các tác động ngẫu nhiên của quá trình
training, chẳng hạn như khởi tạo trọng số ngẫu nhiên hoặc thứ tự mini-batch, và
tính ngẫu nhiên nội tại của chính thuật toán HPO, chẳng hạn như việc lấy mẫu
ngẫu nhiên của random search. Do đó, khi so sánh các thuật toán khác nhau, điều
quan trọng là chạy mỗi experiment nhiều lần và báo cáo các thống kê, chẳng hạn
như mean hoặc median, trên một tập hợp gồm nhiều lần lặp của thuật toán dựa trên
các seed khác nhau của random number generator.

Để minh họa điều này, chúng ta so sánh random search (xem
:numref:`sec_rs`) và Bayesian optimization
:cite:`snoek-nips12` trong việc tuning các siêu tham số của một mạng
nơ-ron feed-forward. Mỗi thuật toán được đánh giá
$50$ lần với một random seed khác nhau. Đường liền biểu thị hiệu năng trung bình
của incumbent qua $50$ lần lặp này, còn đường nét đứt biểu thị độ lệch chuẩn.
Chúng ta có thể thấy random search và Bayesian optimization có hiệu năng gần như
tương đương cho đến khoảng 1000 giây, nhưng Bayesian optimization có thể tận
dụng observation trước đó để xác định các cấu hình tốt hơn, do đó nhanh chóng
vượt qua random search sau thời điểm này.


![Ví dụ về biểu đồ any-time performance để so sánh hai thuật toán A và B.](../img/example_anytime_performance.svg)
:label:`example_anytime_performance`

## Tóm tắt

Phần này đã trình bày một interface đơn giản nhưng linh hoạt để triển khai
nhiều thuật toán HPO mà chúng ta sẽ tìm hiểu trong chương này. Có thể tìm thấy
các interface tương tự trong những framework HPO mã nguồn mở phổ biến. Chúng ta
cũng đã xem xét cách so sánh các thuật toán HPO và những cạm bẫy tiềm ẩn cần
lưu ý.

## Bài tập

1. Mục tiêu của bài tập này là triển khai objective function cho một bài toán HPO khó hơn một chút và chạy các experiment thực tế hơn. Chúng ta sẽ sử dụng MLP hai hidden layer `DropoutMLP` được triển khai trong :numref:`sec_dropout`.
    1. Viết code cho objective function, hàm này phải phụ thuộc vào tất cả các siêu tham số của model và `batch_size`. Sử dụng `max_epochs=50`. GPU không giúp ích trong trường hợp này, vì vậy `num_gpus=0`. Gợi ý: Sửa đổi `hpo_objective_lenet`.
    2. Chọn một search space hợp lý, trong đó `num_hiddens_1`, `num_hiddens_2` là các số nguyên trong $[8, 1024]$, các giá trị dropout nằm trong $[0, 0.95]$, còn `batch_size` nằm trong $[16, 384]$. Cung cấp code cho `config_space`, sử dụng các distribution hợp lý từ `scipy.stats`.
    3. Chạy random search trên ví dụ này với `number_of_trials=20` và biểu diễn kết quả. Trước tiên, hãy đảm bảo đánh giá cấu hình mặc định của :numref:`sec_dropout`, đó là `initial_config = {'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`.
2. Trong bài tập này, bạn sẽ triển khai một searcher mới (lớp con của `HPOSearcher`) đưa ra quyết định dựa trên dữ liệu trước đó. Searcher này phụ thuộc vào các tham số `probab_local`, `num_init_random`. Phương thức `sample_configuration` của nó hoạt động như sau. Trong `num_init_random` lần gọi đầu tiên, thực hiện giống như `RandomSearcher.sample_configuration`. Sau đó, với xác suất `1 - probab_local`, thực hiện giống như `RandomSearcher.sample_configuration`. Nếu không, chọn cấu hình đạt validation error nhỏ nhất cho đến thời điểm hiện tại, chọn ngẫu nhiên một trong các siêu tham số của cấu hình đó và lấy mẫu giá trị của nó một cách ngẫu nhiên như trong `RandomSearcher.sample_configuration`, nhưng giữ nguyên tất cả các giá trị khác. Trả về cấu hình này, vốn giống hệt cấu hình tốt nhất cho đến thời điểm hiện tại, ngoại trừ một siêu tham số.
    1. Viết code cho `LocalSearcher` mới này. Gợi ý: Searcher của bạn yêu cầu `config_space` làm đối số khi khởi tạo. Bạn có thể sử dụng một member có kiểu `RandomSearcher`. Bạn cũng sẽ phải triển khai phương thức `update`.
    2. Chạy lại experiment từ bài tập trước, nhưng sử dụng searcher mới thay cho `RandomSearcher`. Thử nghiệm với các giá trị khác nhau của `probab_local`, `num_init_random`. Tuy nhiên, lưu ý rằng để so sánh đúng đắn giữa các phương pháp HPO khác nhau, cần lặp lại các experiment nhiều lần và lý tưởng nhất là xem xét một số benchmark task.


:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12092)
:end_tab: