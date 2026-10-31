# Tính toán bất đồng bộ
:label:`sec_async`

Máy tính ngày nay là những hệ thống có tính song song cao, bao gồm nhiều lõi CPU (thường là nhiều thread trên mỗi lõi), nhiều phần tử xử lý trên mỗi GPU và thường có nhiều GPU trên mỗi thiết bị. Nói ngắn gọn, chúng ta có thể xử lý nhiều việc khác nhau cùng lúc, thường là trên các thiết bị khác nhau. Đáng tiếc là Python không phải là một cách tuyệt vời để viết mã song song và bất đồng bộ, ít nhất là nếu không có thêm một số hỗ trợ. Suy cho cùng, Python là single-threaded và điều này khó có khả năng thay đổi trong tương lai. Các deep learning framework như MXNet và TensorFlow áp dụng mô hình *lập trình bất đồng bộ* để cải thiện hiệu năng,
trong khi PyTorch sử dụng scheduler của chính Python, dẫn đến một sự đánh đổi hiệu năng khác.
Đối với PyTorch, theo mặc định, các thao tác trên GPU là bất đồng bộ. Khi bạn gọi một hàm sử dụng GPU, các thao tác sẽ được đưa vào hàng đợi của thiết bị cụ thể, nhưng không nhất thiết được thực thi ngay mà có thể được thực thi sau đó. Điều này cho phép chúng ta thực hiện nhiều phép tính song song hơn, bao gồm các thao tác trên CPU hoặc các GPU khác.

Do đó, việc hiểu cách hoạt động của lập trình bất đồng bộ giúp chúng ta phát triển các chương trình hiệu quả hơn bằng cách chủ động giảm các yêu cầu tính toán và sự phụ thuộc lẫn nhau. Điều này cho phép chúng ta giảm overhead bộ nhớ và tăng mức sử dụng bộ xử lý.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## Tính bất đồng bộ thông qua Backend

:begin_tab:`mxnet`
Để khởi động, hãy xem xét bài toán minh họa sau: chúng ta muốn tạo một ma trận ngẫu nhiên và nhân ma trận đó. Hãy thực hiện việc này bằng cả NumPy và `mxnet.np` để thấy sự khác biệt.
:end_tab:

:begin_tab:`pytorch`
Để khởi động, hãy xem xét bài toán minh họa sau: chúng ta muốn tạo một ma trận ngẫu nhiên và nhân ma trận đó. Hãy thực hiện việc này bằng cả NumPy và tensor PyTorch để thấy sự khác biệt.
Lưu ý rằng `tensor` của PyTorch được định nghĩa trên GPU.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
Kết quả benchmark thông qua MXNet nhanh hơn nhiều bậc độ lớn. Vì cả hai đều được thực thi trên cùng một bộ xử lý, hẳn phải có điều gì khác đang diễn ra.
Việc buộc MXNet hoàn tất mọi phép tính của backend trước khi trả về cho thấy điều gì đã xảy ra trước đó: phép tính được backend thực thi trong khi frontend trả quyền điều khiển về cho Python.
:end_tab:

:begin_tab:`pytorch`
Kết quả benchmark thông qua PyTorch nhanh hơn nhiều bậc độ lớn.
Phép nhân vô hướng NumPy được thực thi trên bộ xử lý CPU, trong khi
phép nhân ma trận PyTorch được thực thi trên GPU và do đó phép nhân sau
được kỳ vọng sẽ nhanh hơn nhiều. Tuy nhiên, chênh lệch thời gian rất lớn cho thấy
hẳn phải có điều gì khác đang diễn ra.
Theo mặc định, các thao tác trên GPU trong PyTorch là bất đồng bộ.
Việc buộc PyTorch hoàn tất mọi phép tính trước khi trả về cho thấy
điều gì đã xảy ra trước đó: phép tính được backend thực thi
trong khi frontend trả quyền điều khiển về cho Python.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
Nói chung, MXNet có một frontend để tương tác trực tiếp với người dùng, chẳng hạn thông qua Python, cũng như một backend được hệ thống sử dụng để thực hiện phép tính.
Như được minh họa trong :numref:`fig_frontends`, người dùng có thể viết các chương trình MXNet bằng nhiều ngôn ngữ frontend khác nhau, chẳng hạn như Python, R, Scala và C++. Bất kể sử dụng ngôn ngữ lập trình frontend nào, việc thực thi các chương trình MXNet chủ yếu diễn ra trong backend gồm các triển khai C++. Các thao tác do ngôn ngữ frontend phát hành được chuyển đến backend để thực thi.
Backend quản lý các thread riêng của mình, liên tục thu thập và thực thi các tác vụ đang nằm trong hàng đợi. Lưu ý rằng để hoạt động được, backend phải có khả năng theo dõi các quan hệ phụ thuộc giữa nhiều bước khác nhau trong computational graph. Do đó, không thể song song hóa các thao tác phụ thuộc lẫn nhau.
:end_tab:

:begin_tab:`pytorch`
Nói chung, PyTorch có một frontend để tương tác trực tiếp với người dùng, chẳng hạn thông qua Python, cũng như một backend được hệ thống sử dụng để thực hiện phép tính.
Như được minh họa trong :numref:`fig_frontends`, người dùng có thể viết các chương trình PyTorch bằng nhiều ngôn ngữ frontend khác nhau, chẳng hạn như Python và C++. Bất kể sử dụng ngôn ngữ lập trình frontend nào, việc thực thi các chương trình PyTorch chủ yếu diễn ra trong backend gồm các triển khai C++. Các thao tác do ngôn ngữ frontend phát hành được chuyển đến backend để thực thi.
Backend quản lý các thread riêng của mình, liên tục thu thập và thực thi các tác vụ đang nằm trong hàng đợi.
Lưu ý rằng để hoạt động được, backend phải có khả năng theo dõi các
quan hệ phụ thuộc giữa nhiều bước khác nhau trong computational graph.
Do đó, không thể song song hóa các thao tác phụ thuộc lẫn nhau.
:end_tab:

![Frontend của ngôn ngữ lập trình và backend của deep learning framework.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

Hãy xem một ví dụ minh họa khác để hiểu rõ hơn một chút về dependency graph.

```{.python .input}
#@tab mxnet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![Backend theo dõi các quan hệ phụ thuộc giữa nhiều bước khác nhau trong computational graph.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`



Đoạn mã ở trên cũng được minh họa trong :numref:`fig_asyncgraph`.
Mỗi khi thread frontend Python thực thi một trong ba câu lệnh đầu tiên, nó chỉ cần trả tác vụ về hàng đợi backend. Khi kết quả của câu lệnh cuối cùng cần được *in ra*, thread frontend Python sẽ chờ thread backend C++ hoàn tất việc tính toán kết quả của biến `z`. Một lợi ích của thiết kế này là thread frontend Python không cần thực hiện các phép tính thực tế. Vì vậy, hiệu năng tổng thể của chương trình hầu như không bị ảnh hưởng, bất kể hiệu năng của Python. :numref:`fig_threading` minh họa cách frontend và backend tương tác.

![Tương tác giữa frontend và backend.](../img/threading.svg)
:label:`fig_threading`




## Rào cản và trình chặn

:begin_tab:`mxnet`
Có một số thao tác sẽ buộc Python phải chờ hoàn tất:

* Rõ ràng nhất là `npx.waitall()` sẽ chờ cho đến khi mọi phép tính hoàn tất, bất kể các chỉ thị tính toán được phát hành vào thời điểm nào. Trên thực tế, không nên sử dụng toán tử này trừ khi thực sự cần thiết, vì nó có thể dẫn đến hiệu năng kém.
* Nếu chỉ muốn chờ cho đến khi một biến cụ thể khả dụng, chúng ta có thể gọi `z.wait_to_read()`. Trong trường hợp này, MXNet sẽ chặn việc trả quyền điều khiển về Python cho đến khi biến `z` được tính xong. Các phép tính khác hoàn toàn có thể tiếp tục sau đó.

Hãy xem điều này hoạt động như thế nào trên thực tế.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
Cả hai thao tác mất khoảng cùng một khoảng thời gian để hoàn tất. Ngoài các thao tác chặn rõ ràng, chúng tôi khuyến nghị bạn lưu ý đến những trình chặn *ngầm định*. Việc in một biến rõ ràng đòi hỏi biến đó phải khả dụng và do đó là một trình chặn. Cuối cùng, việc chuyển đổi sang NumPy thông qua `z.asnumpy()` và chuyển đổi sang scalar thông qua `z.item()` đều là các thao tác chặn, vì NumPy không có khái niệm về tính bất đồng bộ. Nó cần quyền truy cập vào các giá trị, giống như hàm `print`.

Việc thường xuyên sao chép một lượng nhỏ dữ liệu từ phạm vi của MXNet sang NumPy rồi sao chép ngược lại có thể làm mất hiệu năng của một đoạn mã vốn hiệu quả, vì mỗi thao tác như vậy yêu cầu computational graph đánh giá tất cả các kết quả trung gian cần thiết để lấy được giá trị liên quan *trước khi* có thể thực hiện bất kỳ việc gì khác.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## Cải thiện phép tính

:begin_tab:`mxnet`
Trên một hệ thống có nhiều thread (ngay cả laptop thông thường cũng có từ 4 thread trở lên, còn trên các server nhiều socket, con số này có thể vượt quá 256), overhead của việc scheduling các thao tác có thể trở nên đáng kể. Vì vậy, việc thực hiện phép tính và scheduling một cách bất đồng bộ và song song là rất đáng mong muốn. Để minh họa lợi ích của cách làm này, hãy xem điều gì xảy ra nếu chúng ta tăng một biến lên 1 nhiều lần, lần lượt theo tuần tự hoặc bất đồng bộ. Chúng ta mô phỏng việc thực thi đồng bộ bằng cách chèn một rào cản `wait_to_read` giữa mỗi phép cộng.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Có thể tóm tắt tương tác được đơn giản hóa một chút giữa thread frontend Python và thread backend C++ như sau:
1. Frontend yêu cầu backend đưa tác vụ tính toán `y = x + 1` vào hàng đợi.
1. Sau đó, backend nhận các tác vụ tính toán từ hàng đợi và thực hiện các phép tính thực tế.
1. Sau đó, backend trả kết quả tính toán về frontend.
Giả sử thời lượng của ba giai đoạn này lần lượt là $t_1$, $t_2$ và $t_3$. Nếu không sử dụng lập trình bất đồng bộ, tổng thời gian cần để thực hiện 10000 phép tính xấp xỉ bằng $10000 (t_1+ t_2 + t_3)$. Nếu sử dụng lập trình bất đồng bộ, tổng thời gian cần để thực hiện 10000 phép tính có thể giảm xuống còn $t_1 + 10000 t_2 + t_3$ (giả sử $10000 t_2 > 9999t_1$), vì frontend không phải chờ backend trả kết quả tính toán sau mỗi vòng lặp.
:end_tab:


## Tóm tắt


* Các deep learning framework có thể tách frontend Python khỏi backend thực thi. Điều này cho phép đưa các lệnh vào backend một cách bất đồng bộ với tốc độ cao và tận dụng tính song song đi kèm.
* Tính bất đồng bộ giúp frontend có khả năng phản hồi khá tốt. Tuy nhiên, hãy cẩn thận không làm đầy hàng đợi tác vụ quá mức, vì điều đó có thể dẫn đến việc tiêu thụ quá nhiều bộ nhớ. Nên đồng bộ hóa sau mỗi minibatch để giữ frontend và backend gần như đồng bộ.
* Các nhà cung cấp chip cung cấp những công cụ phân tích hiệu năng tinh vi, giúp thu được góc nhìn chi tiết hơn nhiều về hiệu quả của deep learning.

:begin_tab:`mxnet`
* Lưu ý rằng việc chuyển đổi từ cơ chế quản lý bộ nhớ của MXNet sang Python sẽ buộc backend phải chờ cho đến khi biến cụ thể đó sẵn sàng. Các hàm như `print`, `asnumpy` và `item` đều có tác động này. Điều này có thể hữu ích, nhưng việc sử dụng đồng bộ hóa bất cẩn có thể làm giảm hiệu năng nghiêm trọng.
:end_tab:


## Bài tập

:begin_tab:`mxnet`
1. Ở trên, chúng ta đã đề cập rằng việc sử dụng phép tính bất đồng bộ có thể giảm tổng thời gian cần thiết để thực hiện 10000 phép tính xuống còn $t_1 + 10000 t_2 + t_3$. Tại sao ở đây chúng ta phải giả định rằng $10000 t_2 > 9999 t_1$?
1. Đo sự khác biệt giữa `waitall` và `wait_to_read`. Gợi ý: thực hiện một số lệnh rồi đồng bộ hóa để nhận kết quả trung gian.
:end_tab:

:begin_tab:`pytorch`
1. Trên CPU, hãy benchmark các phép nhân ma trận tương tự trong phần này. Bạn còn có thể quan sát tính bất đồng bộ thông qua backend không?
:end_tab:

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/2564)
:end_tab: