# Sử dụng Amazon SageMaker
:label:`sec_sagemaker`

Các ứng dụng deep learning
có thể yêu cầu nhiều tài nguyên tính toán đến mức
dễ dàng vượt quá
khả năng cung cấp của máy cục bộ.
Các dịch vụ điện toán đám mây
cho phép bạn
chạy mã yêu cầu nhiều GPU trong cuốn sách này
dễ dàng hơn
bằng cách sử dụng các máy tính mạnh hơn.
Phần này sẽ giới thiệu
cách sử dụng Amazon SageMaker
để chạy mã của cuốn sách này.

## Đăng ký

Trước tiên, chúng ta cần đăng ký tài khoản tại https://aws.amazon.com/.
Để tăng cường bảo mật,
bạn nên sử dụng xác thực hai yếu tố.
Bạn cũng nên
thiết lập các cảnh báo chi tiết về việc lập hóa đơn và chi tiêu để
tránh các khoản phí bất ngờ,
ví dụ như
khi quên dừng các instance đang chạy.
Sau khi đăng nhập vào tài khoản AWS,
hãy truy cập [console](http://console.aws.amazon.com/) của bạn và tìm kiếm "Amazon SageMaker" (xem :numref:`fig_sagemaker`),
sau đó nhấp vào mục này để mở bảng điều khiển SageMaker.

![Tìm kiếm và mở bảng điều khiển SageMaker.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## Tạo một SageMaker Instance

Tiếp theo, hãy tạo một notebook instance như mô tả trong :numref:`fig_sagemaker-create`.

![Tạo một SageMaker instance.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker cung cấp nhiều [loại instance](https://aws.amazon.com/sagemaker/pricing/instance-types/) với năng lực tính toán và mức giá khác nhau.
Khi tạo một notebook instance,
chúng ta có thể chỉ định tên và loại của instance.
Trong :numref:`fig_sagemaker-create-2`, chúng ta chọn `ml.p3.2xlarge`: với một GPU Tesla V100 và CPU 8 lõi, instance này đủ mạnh cho hầu hết nội dung trong cuốn sách.

![Chọn loại instance.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
Toàn bộ cuốn sách ở định dạng ipynb để chạy với SageMaker có tại https://github.com/d2l-ai/d2l-en-sagemaker. Chúng ta có thể chỉ định URL của repository GitHub này (:numref:`fig_sagemaker-create-3`) để cho phép SageMaker clone repository khi tạo instance.
:end_tab:

:begin_tab:`pytorch`
Toàn bộ cuốn sách ở định dạng ipynb để chạy với SageMaker có tại https://github.com/d2l-ai/d2l-pytorch-sagemaker. Chúng ta có thể chỉ định URL của repository GitHub này (:numref:`fig_sagemaker-create-3`) để cho phép SageMaker clone repository khi tạo instance.
:end_tab:

:begin_tab:`tensorflow`
Toàn bộ cuốn sách ở định dạng ipynb để chạy với SageMaker có tại https://github.com/d2l-ai/d2l-tensorflow-sagemaker. Chúng ta có thể chỉ định URL của repository GitHub này (:numref:`fig_sagemaker-create-3`) để cho phép SageMaker clone repository khi tạo instance.
:end_tab:

![Chỉ định repository GitHub.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## Chạy và dừng một Instance

Việc tạo instance
có thể mất vài phút.
Khi instance sẵn sàng,
hãy nhấp vào liên kết "Open Jupyter" bên cạnh instance đó (:numref:`fig_sagemaker-open`) để bạn có thể
chỉnh sửa và chạy tất cả notebook Jupyter
của cuốn sách này trên instance đó
(tương tự các bước trong :numref:`sec_jupyter`).

![Mở Jupyter trên SageMaker instance đã tạo.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`


Sau khi hoàn tất công việc,
đừng quên dừng instance để tránh
tiếp tục bị tính phí (:numref:`fig_sagemaker-stop`).

![Dừng một SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## Cập nhật Notebook

:begin_tab:`mxnet`
Các notebook của cuốn sách mã nguồn mở này sẽ được cập nhật thường xuyên trong repository [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker)
trên GitHub.
Để cập nhật lên phiên bản mới nhất,
bạn có thể mở một terminal trên SageMaker instance (:numref:`fig_sagemaker-terminal`).
:end_tab:

:begin_tab:`pytorch`
Các notebook của cuốn sách mã nguồn mở này sẽ được cập nhật thường xuyên trong repository [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker)
trên GitHub.
Để cập nhật lên phiên bản mới nhất,
bạn có thể mở một terminal trên SageMaker instance (:numref:`fig_sagemaker-terminal`).
:end_tab:


:begin_tab:`tensorflow`
Các notebook của cuốn sách mã nguồn mở này sẽ được cập nhật thường xuyên trong repository [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker)
trên GitHub.
Để cập nhật lên phiên bản mới nhất,
bạn có thể mở một terminal trên SageMaker instance (:numref:`fig_sagemaker-terminal`).
:end_tab:


![Mở một terminal trên SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

Bạn có thể muốn commit các thay đổi cục bộ trước khi pull các bản cập nhật từ repository từ xa.
Nếu không,
chỉ cần loại bỏ tất cả thay đổi cục bộ của bạn
bằng các lệnh sau trong terminal:

:begin_tab:`mxnet`

```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`pytorch`

```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`tensorflow`

```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```


:end_tab:

## Tóm tắt

* Chúng ta có thể tạo một notebook instance bằng Amazon SageMaker để chạy mã yêu cầu nhiều GPU trong cuốn sách này.
* Chúng ta có thể cập nhật notebook thông qua terminal trên Amazon SageMaker instance.


## Bài tập


1. Chỉnh sửa và chạy bất kỳ phần nào yêu cầu GPU bằng Amazon SageMaker.
1. Mở một terminal để truy cập thư mục cục bộ chứa tất cả notebook của cuốn sách này.


[Thảo luận](https://discuss.d2l.ai/t/422)