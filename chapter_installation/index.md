# Cài đặt
:label:`chap_installation`

Để bắt đầu,
chúng ta sẽ cần một môi trường để chạy Python,
Jupyter Notebook, các thư viện liên quan
và mã cần thiết để chạy chính cuốn sách.

## Cài đặt Miniconda

Lựa chọn đơn giản nhất là cài đặt
[Miniconda](https://conda.io/en/latest/miniconda.html).
Lưu ý rằng cần sử dụng phiên bản Python 3.x.
Bạn có thể bỏ qua các bước sau
nếu máy của bạn đã cài đặt conda.

Truy cập trang web Miniconda và xác định
phiên bản phù hợp với hệ thống của bạn
dựa trên phiên bản Python 3.x và kiến trúc máy.
Giả sử phiên bản Python của bạn là 3.9
(phiên bản chúng tôi đã kiểm thử).
Nếu bạn sử dụng macOS,
bạn sẽ tải xuống tập lệnh bash
có tên chứa chuỗi "MacOSX",
đi đến vị trí tải xuống
và thực hiện cài đặt như sau
(lấy máy Mac Intel làm ví dụ):

```bash
# Tên tệp có thể thay đổi
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Người dùng Linux
sẽ tải xuống tệp
có tên chứa chuỗi "Linux"
và thực hiện lệnh sau tại vị trí tải xuống:

```bash
# Tên tệp có thể thay đổi
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Người dùng Windows sẽ tải xuống và cài đặt Miniconda theo [hướng dẫn trực tuyến](https://conda.io/en/latest/miniconda.html).
Trên Windows, bạn có thể tìm kiếm `cmd` để mở Command Prompt (trình thông dịch dòng lệnh) dùng để chạy các lệnh.

Tiếp theo, khởi tạo shell để chúng ta có thể chạy `conda` trực tiếp.

```bash
~/miniconda3/bin/conda init
```


Sau đó đóng và mở lại shell hiện tại.
Bạn sẽ có thể tạo
một environment mới như sau:

```bash
conda create --name d2l python=3.9 -y
```


Bây giờ chúng ta có thể kích hoạt environment `d2l`:

```bash
conda activate d2l
```


## Cài đặt Deep Learning Framework và Package `d2l`

Trước khi cài đặt bất kỳ deep learning framework nào,
trước tiên hãy kiểm tra xem
máy của bạn có GPU phù hợp hay không
(các GPU dùng để xuất hình ảnh
trên một laptop thông thường không liên quan đến mục đích của chúng ta).
Ví dụ,
nếu máy tính của bạn có GPU NVIDIA và đã cài đặt [CUDA](https://developer.nvidia.com/cuda-downloads),
thì bạn đã sẵn sàng.
Nếu máy của bạn không có GPU,
hiện tại bạn chưa cần lo lắng.
CPU của bạn cung cấp đủ năng lực tính toán
để giúp bạn hoàn thành vài chương đầu tiên.
Chỉ cần nhớ rằng bạn sẽ muốn sử dụng GPU
trước khi chạy các model lớn hơn.


:begin_tab:`mxnet`

Để cài đặt phiên bản MXNet hỗ trợ GPU,
chúng ta cần biết phiên bản CUDA đã được cài đặt.
Bạn có thể kiểm tra bằng cách chạy `nvcc --version`
hoặc `cat /usr/local/cuda/version.txt`.
Giả sử bạn đã cài đặt CUDA 11.2,
hãy thực hiện lệnh sau:

```bash
# Dành cho người dùng macOS và Linux
pip install mxnet-cu112==1.9.1

# Dành cho người dùng Windows
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


Bạn có thể thay đổi các chữ số cuối theo phiên bản CUDA của mình, ví dụ: `cu101` cho
CUDA 10.1 và `cu90` cho CUDA 9.0.


Nếu máy của bạn không có GPU NVIDIA
hoặc CUDA,
bạn có thể cài đặt phiên bản CPU
như sau:

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

Bạn có thể cài đặt PyTorch (các phiên bản được chỉ định đã được kiểm thử tại thời điểm viết tài liệu) với hỗ trợ CPU hoặc GPU như sau:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
Bạn có thể cài đặt TensorFlow với hỗ trợ CPU hoặc GPU như sau:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
Bạn có thể cài đặt JAX và Flax với hỗ trợ CPU hoặc GPU như sau:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


Nếu máy của bạn không có GPU NVIDIA
hoặc CUDA,
bạn có thể cài đặt phiên bản CPU
như sau:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


Bước tiếp theo là cài đặt
package `d2l` do chúng tôi phát triển
nhằm đóng gói
các hàm và class thường được sử dụng
trong toàn bộ cuốn sách:

```bash
pip install d2l==1.0.3
```


## Tải xuống và chạy mã

Tiếp theo, bạn sẽ muốn tải xuống các notebook
để có thể chạy từng khối mã trong sách.
Chỉ cần nhấp vào tab "Notebooks" ở đầu
bất kỳ trang HTML nào trên [trang web D2L.ai](https://d2l.ai/)
để tải xuống mã, sau đó giải nén.
Ngoài ra, bạn có thể lấy các notebook
từ command line như sau:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

Nếu `unzip` chưa được cài đặt, trước tiên hãy chạy `sudo apt-get install unzip`.
Bây giờ chúng ta có thể khởi động server Jupyter Notebook bằng cách chạy:

```bash
jupyter notebook
```


Tại thời điểm này, bạn có thể mở http://localhost:8888
(trang này có thể đã tự động mở) trong trình duyệt web.
Sau đó, chúng ta có thể chạy mã cho từng phần trong sách.
Mỗi khi mở một cửa sổ command line mới,
bạn sẽ cần thực thi `conda activate d2l`
để kích hoạt runtime environment
trước khi chạy các notebook D2L,
hoặc cập nhật các package
(Deep Learning Framework
hoặc package `d2l`).
Để thoát khỏi environment,
hãy chạy `conda deactivate`.


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17964)
:end_tab: