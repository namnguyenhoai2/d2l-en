# Tự động song song hóa
:label:`sec_auto_para`


Các framework deep learning (ví dụ: MXNet và PyTorch) tự động xây dựng các đồ thị tính toán ở backend. Bằng cách sử dụng một
đồ thị tính toán, hệ thống biết được tất cả các phần phụ thuộc,
và có thể chọn thực thi song song nhiều tác vụ không phụ thuộc lẫn nhau để
tăng tốc độ. Chẳng hạn, :numref:`fig_asyncgraph` trong :numref:`sec_async` khởi tạo hai biến một cách độc lập. Do đó, hệ thống có thể chọn thực thi chúng song song.


Thông thường, một operator sẽ sử dụng toàn bộ tài nguyên tính toán trên tất cả CPU hoặc trên một GPU duy nhất. Ví dụ, operator `dot` sẽ sử dụng tất cả các core (và thread) trên mọi CPU, ngay cả khi một máy có nhiều bộ xử lý CPU. Điều tương tự cũng áp dụng cho một GPU duy nhất. Vì vậy, việc song song hóa không thực sự hữu ích đối với các máy tính chỉ có một device. Với nhiều device, điều này trở nên quan trọng hơn. Mặc dù song song hóa thường phù hợp nhất giữa nhiều GPU, việc bổ sung CPU cục bộ vẫn sẽ cải thiện hiệu năng đôi chút. Ví dụ, hãy xem :citet:`Hadjis.Zhang.Mitliagkas.ea.2016`, tập trung vào việc huấn luyện các model thị giác máy tính bằng cách kết hợp GPU và CPU. Với sự tiện lợi của một framework tự động song song hóa, chúng ta có thể đạt được mục tiêu tương tự chỉ với vài dòng code Python. Trên phạm vi rộng hơn, phần thảo luận về tính toán song song tự động của chúng ta tập trung vào tính toán song song bằng cả CPU và GPU, cũng như việc song song hóa tính toán và giao tiếp.

Lưu ý rằng chúng ta cần ít nhất hai GPU để chạy các thử nghiệm trong phần này.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## Tính toán song song trên GPU

Hãy bắt đầu bằng cách xác định một workload tham chiếu để kiểm thử: hàm `run` bên dưới thực hiện 10 phép nhân ma trận-ma trận trên device do chúng ta chọn, bằng cách sử dụng dữ liệu được cấp phát vào hai biến: `x_gpu1` và `x_gpu2`.

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
Bây giờ, chúng ta áp dụng hàm này cho dữ liệu. Để bảo đảm caching không ảnh hưởng đến kết quả, chúng ta làm nóng các device bằng cách thực hiện một lượt chạy duy nhất trên mỗi device trước khi đo.
:end_tab:

:begin_tab:`pytorch`
Bây giờ, chúng ta áp dụng hàm này cho dữ liệu. Để bảo đảm caching không ảnh hưởng đến kết quả, chúng ta làm nóng các device bằng cách thực hiện một lượt chạy duy nhất trên mỗi device trước khi đo. `torch.cuda.synchronize()` chờ tất cả kernel trong mọi stream trên một device CUDA hoàn tất. Hàm này nhận một đối số `device`, là device mà chúng ta cần đồng bộ hóa. Hàm sử dụng device hiện tại, được xác định bởi `current_device()`, nếu đối số device là `None` (mặc định).
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Nếu xóa câu lệnh `waitall` giữa hai tác vụ, hệ thống sẽ được tự do tự động song song hóa việc tính toán trên cả hai device.
:end_tab:

:begin_tab:`pytorch`
Nếu xóa câu lệnh `synchronize` giữa hai tác vụ, hệ thống sẽ được tự do tự động song song hóa việc tính toán trên cả hai device.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

Trong trường hợp trên, tổng thời gian thực thi nhỏ hơn tổng thời gian của từng phần, vì framework deep learning tự động lập lịch tính toán trên cả hai device GPU mà người dùng không cần viết code phức tạp.



## Tính toán và giao tiếp song song

Trong nhiều trường hợp, chúng ta cần di chuyển dữ liệu giữa các device khác nhau, chẳng hạn giữa CPU và GPU, hoặc giữa các GPU khác nhau.
Ví dụ,
điều này xảy ra khi chúng ta muốn thực hiện tối ưu hóa phân tán, trong đó cần tổng hợp các gradient trên nhiều accelerator card. Hãy mô phỏng việc này bằng cách tính toán trên GPU rồi sao chép kết quả trở lại CPU.

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Cách này tương đối kém hiệu quả. Lưu ý rằng chúng ta có thể bắt đầu sao chép các phần của `y` sang CPU trong khi phần còn lại của danh sách vẫn đang được tính toán. Tình huống này xảy ra, chẳng hạn, khi chúng ta tính gradient trên một minibatch. Gradient của một số parameter sẽ có sẵn sớm hơn gradient của các parameter khác. Vì vậy, việc bắt đầu sử dụng băng thông của bus PCI-Express trong khi GPU vẫn đang chạy sẽ có lợi cho chúng ta. Việc xóa `waitall` giữa hai phần cho phép chúng ta mô phỏng tình huống này.
:end_tab:

:begin_tab:`pytorch`
Cách này tương đối kém hiệu quả. Lưu ý rằng chúng ta có thể bắt đầu sao chép các phần của `y` sang CPU trong khi phần còn lại của danh sách vẫn đang được tính toán. Tình huống này xảy ra, chẳng hạn, khi chúng ta tính gradient (backprop) trên một minibatch. Gradient của một số parameter sẽ có sẵn sớm hơn gradient của các parameter khác. Vì vậy, việc bắt đầu sử dụng băng thông của bus PCI-Express trong khi GPU vẫn đang chạy sẽ có lợi cho chúng ta. Trong PyTorch, một số hàm như `to()` và `copy_()` hỗ trợ đối số `non_blocking` tường minh, cho phép caller bỏ qua việc đồng bộ hóa khi không cần thiết. Thiết lập `non_blocking=True` cho phép chúng ta mô phỏng tình huống này.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Tổng thời gian cần thiết cho cả hai thao tác (như dự kiến) nhỏ hơn tổng thời gian của từng thao tác.
Lưu ý rằng tác vụ này khác với tính toán song song vì nó sử dụng một tài nguyên khác: bus giữa CPU và GPU. Trên thực tế, chúng ta có thể tính toán trên cả hai device và giao tiếp đồng thời. Như đã đề cập ở trên, tồn tại một phần phụ thuộc giữa tính toán và giao tiếp: `y[i]` phải được tính toán trước khi có thể sao chép nó sang CPU. May mắn là hệ thống có thể sao chép `y[i-1]` trong khi tính toán `y[i]` để giảm tổng thời gian chạy.

Chúng ta kết thúc bằng một minh họa về đồ thị tính toán và các phần phụ thuộc của nó đối với một MLP hai lớp khi huấn luyện trên một CPU và hai GPU, như được minh họa trong :numref:`fig_twogpu`. Việc tự lập lịch cho chương trình song song tạo ra từ đó sẽ rất khó khăn. Đây là lúc backend tính toán dựa trên đồ thị trở nên hữu ích cho việc tối ưu hóa.

![Đồ thị tính toán và các phần phụ thuộc của một MLP hai lớp trên một CPU và hai GPU.](../img/twogpu.svg)
:label:`fig_twogpu`


## Tóm tắt

* Các hệ thống hiện đại có nhiều loại device, chẳng hạn như nhiều GPU và CPU. Chúng có thể được sử dụng song song, không đồng bộ.
* Các hệ thống hiện đại cũng có nhiều loại tài nguyên giao tiếp, chẳng hạn như PCI Express, bộ lưu trữ (thường là ổ đĩa thể rắn hoặc thông qua mạng), và băng thông mạng. Chúng có thể được sử dụng song song để đạt hiệu suất tối đa.
* Backend có thể cải thiện hiệu năng thông qua tính toán và giao tiếp song song tự động.

## Bài tập

1. Tám thao tác đã được thực hiện trong hàm `run` được định nghĩa trong phần này. Không có phần phụ thuộc nào giữa chúng. Hãy thiết kế một thử nghiệm để xem framework deep learning có tự động thực thi chúng song song hay không.
1. Khi workload của một operator riêng lẻ đủ nhỏ, việc song song hóa có thể hữu ích ngay cả trên một CPU hoặc GPU duy nhất. Hãy thiết kế một thử nghiệm để kiểm chứng điều này.
1. Hãy thiết kế một thử nghiệm sử dụng tính toán song song trên CPU, GPU và giao tiếp giữa cả hai device.
1. Sử dụng một debugger như [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) của NVIDIA để kiểm tra xem code của bạn có hiệu quả hay không.
1. Thiết kế các tác vụ tính toán bao gồm những phần phụ thuộc dữ liệu phức tạp hơn, rồi chạy các thử nghiệm để xem liệu bạn có thể thu được kết quả chính xác đồng thời cải thiện hiệu năng hay không.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1681)
:end_tab: