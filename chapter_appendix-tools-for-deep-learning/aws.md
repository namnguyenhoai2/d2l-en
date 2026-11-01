# Sử dụng các phiên bản EC2 của AWS
:label:`sec_aws`

Trong phần này, chúng ta sẽ hướng dẫn bạn cách cài đặt tất cả thư viện trên một máy Linux thô. Hãy nhớ rằng trong :numref:`sec_sagemaker` chúng ta đã thảo luận về cách sử dụng Amazon SageMaker, trong khi tự xây dựng một phiên bản sẽ tiết kiệm chi phí hơn trên AWS. Hướng dẫn này gồm ba bước:

1. Yêu cầu một phiên bản Linux có GPU từ AWS EC2.
1. Cài đặt CUDA (hoặc sử dụng Amazon Machine Image đã cài đặt sẵn CUDA).
1. Cài đặt deep learning framework và các thư viện khác để chạy mã của cuốn sách.

Quy trình này cũng áp dụng cho các phiên bản khác (và các cloud khác), dù cần một vài thay đổi nhỏ. Trước khi tiếp tục, bạn cần tạo một tài khoản AWS; xem :numref:`sec_sagemaker` để biết thêm chi tiết.


## Tạo và chạy một phiên bản EC2

Sau khi đăng nhập vào tài khoản AWS, nhấp vào "EC2" (:numref:`fig_aws`) để đi đến bảng điều khiển EC2.

![Mở bảng điều khiển EC2.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2` hiển thị bảng điều khiển EC2.

![Bảng điều khiển EC2.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Thiết lập vị trí

Chọn một trung tâm dữ liệu ở gần để giảm độ trễ, chẳng hạn như "Oregon" (được đánh dấu bằng khung màu đỏ ở góc trên bên phải của :numref:`fig_ec2`). Nếu bạn ở Trung Quốc,
bạn có thể chọn một khu vực Châu Á - Thái Bình Dương ở gần, chẳng hạn như Seoul hoặc Tokyo. Lưu ý
rằng một số trung tâm dữ liệu có thể không có các phiên bản GPU.


### Tăng hạn mức

Trước khi chọn một phiên bản, hãy kiểm tra xem có giới hạn
số lượng hay không bằng cách nhấp vào nhãn "Limits" trên thanh bên trái như minh họa trong
:numref:`fig_ec2`. 
:numref:`fig_limits` minh họa một ví dụ về giới hạn như vậy. Tài khoản hiện không thể mở các phiên bản "p2.xlarge" tùy theo khu vực. Nếu
bạn cần mở một hoặc nhiều phiên bản, hãy nhấp vào liên kết "Request limit increase" để
yêu cầu hạn ngạch phiên bản cao hơn.
Thông thường, mất một ngày làm việc để
xử lý yêu cầu.

![Giới hạn số lượng phiên bản.](../img/limits.png)
:width:`700px`
:label:`fig_limits`


### Khởi chạy một phiên bản

Tiếp theo, nhấp vào nút "Launch Instance" được đánh dấu bằng khung màu đỏ trong :numref:`fig_ec2` để khởi chạy phiên bản của bạn.

Trước tiên, chúng ta chọn một Amazon Machine Image (AMI) phù hợp. Chọn một phiên bản Ubuntu (:numref:`fig_ubuntu`).


![Chọn một AMI.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 cung cấp nhiều cấu hình phiên bản khác nhau để lựa chọn. Điều này đôi khi có thể khiến người mới bắt đầu cảm thấy choáng ngợp. :numref:`tab_ec2` liệt kê các máy phù hợp khác nhau.

:Các loại phiên bản EC2 khác nhau
:label:`tab_ec2`

| Tên | GPU         | Ghi chú                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | lỗi thời                       |
| p2   | Kepler K80  | cũ nhưng thường rẻ khi dùng spot |
| g3   | Maxwell M60 | sự cân bằng tốt                |
| p3   | Volta V100  | hiệu năng cao cho FP16         |
| p4   | Ampere A100 | hiệu năng cao cho quá trình training quy mô lớn |
| g4   | Turing T4   | được tối ưu cho inference FP16/INT8 |


Tất cả các máy chủ này đều có nhiều biến thể, cho biết số lượng GPU được sử dụng. Ví dụ, p2.xlarge có 1 GPU, còn p2.16xlarge có 16 GPU và nhiều bộ nhớ hơn. Để biết thêm chi tiết, hãy xem [tài liệu AWS EC2](https://aws.amazon.com/ec2/instance-types/) hoặc [trang tóm tắt](https://www.ec2instances.info). Với mục đích minh họa, p2.xlarge là đủ (được đánh dấu bằng khung màu đỏ trong :numref:`fig_p2x`).

![Chọn một phiên bản.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

Lưu ý rằng bạn nên sử dụng một phiên bản hỗ trợ GPU với các driver phù hợp và một deep learning framework hỗ trợ GPU. Nếu không, bạn sẽ không nhận được lợi ích nào từ việc sử dụng GPU.

Tiếp theo, chúng ta chọn key pair được dùng để truy cập
phiên bản. Nếu chưa có key pair, hãy nhấp vào "Create new key pair" trong :numref:`fig_keypair` để tạo một key pair. Sau đó,
bạn có thể chọn key pair
đã tạo trước đó.
Hãy chắc chắn rằng bạn tải key pair xuống và lưu trữ ở nơi an toàn nếu
bạn tạo một key pair mới. Đây là cách duy nhất để SSH vào máy chủ.

![Chọn một key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Trong ví dụ này, chúng ta giữ nguyên cấu hình mặc định cho "Network settings" (nhấp vào nút "Edit" để cấu hình các mục như subnet và security group). Chúng ta chỉ tăng kích thước ổ cứng mặc định lên 64 GB (:numref:`fig_disk`). Lưu ý rằng riêng CUDA đã chiếm 4 GB.

![Thay đổi kích thước ổ cứng.](../img/disk.png)
:width:`700px`
:label:`fig_disk`


Nhấp vào "Launch Instance" để khởi chạy
phiên bản đã tạo. Nhấp vào
instance ID hiển thị trong :numref:`fig_launching` để xem trạng thái của phiên bản này.

![Nhấp vào instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Kết nối với phiên bản

Như minh họa trong :numref:`fig_connect`, sau khi trạng thái phiên bản chuyển sang màu xanh lục, hãy nhấp chuột phải vào phiên bản và chọn `Connect` để xem phương thức truy cập phiên bản.

![Xem phương thức truy cập phiên bản.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

Nếu đây là key mới, key đó không được phép hiển thị công khai để SSH hoạt động. Đi đến thư mục nơi bạn lưu `D2L_key.pem` và 
thực thi lệnh sau
để key không còn hiển thị công khai:

```bash
chmod 400 D2L_key.pem
```


![Xem phương thức truy cập và khởi động phiên bản.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


Bây giờ, sao chép lệnh SSH trong khung màu đỏ phía dưới của :numref:`fig_chmod` và dán vào dòng lệnh:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```


Khi dòng lệnh hỏi "Are you sure you want to continue connecting (yes/no)", hãy nhập "yes" và nhấn Enter để đăng nhập vào phiên bản.

Bây giờ, máy chủ của bạn đã sẵn sàng.


## Cài đặt CUDA

Trước khi cài đặt CUDA, hãy nhớ cập nhật phiên bản bằng các driver mới nhất.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


Ở đây, chúng ta tải xuống CUDA 12.1. Truy cập [kho lưu trữ chính thức](https://developer.nvidia.com/cuda-toolkit-archive) của NVIDIA để tìm liên kết tải xuống như minh họa trong :numref:`fig_cuda`.

![Tìm địa chỉ tải xuống CUDA 12.1.](../img/cuda121.png)
:width:`500px`
:label:`fig_cuda`

Sao chép hướng dẫn và dán vào terminal để cài đặt CUDA 12.1.

```bash
# Liên kết và tên tệp có thể thay đổi
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```


Sau khi cài đặt chương trình, chạy lệnh sau để xem các GPU:

```bash
nvidia-smi
```


Cuối cùng, thêm CUDA vào library path để các thư viện khác có thể tìm thấy nó, chẳng hạn như bằng cách thêm các dòng sau vào cuối `~/.bashrc`.

```bash
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64
```


## Cài đặt các thư viện để chạy mã

Để chạy mã của cuốn sách này,
chỉ cần làm theo các bước trong :ref:`chap_installation`
dành cho người dùng Linux trên phiên bản EC2
và sử dụng các mẹo sau
khi làm việc trên một máy chủ Linux từ xa:

* Để tải xuống bash script trên trang cài đặt Miniconda, hãy nhấp chuột phải vào liên kết tải xuống và chọn "Copy Link Address", sau đó thực thi `wget [copied link address]`.
* Sau khi chạy `~/miniconda3/bin/conda init`, bạn có thể thực thi `source ~/.bashrc` thay vì đóng rồi mở lại shell hiện tại.


## Chạy Jupyter Notebook từ xa

Để chạy Jupyter Notebook từ xa, bạn cần sử dụng SSH port forwarding. Suy cho cùng, máy chủ trên cloud không có màn hình hoặc bàn phím. Để thực hiện việc này, hãy đăng nhập vào máy chủ từ máy tính để bàn (hoặc laptop) của bạn như sau:

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```


Tiếp theo, đi đến vị trí
của mã cuốn sách đã tải xuống
trên phiên bản EC2,
sau đó chạy:

```
conda activate d2l
jupyter notebook
```


:numref:`fig_jupyter` hiển thị kết quả có thể nhận được sau khi bạn chạy Jupyter Notebook. Dòng cuối cùng là URL của port 8888.

![Kết quả sau khi chạy Jupyter Notebook. Dòng cuối cùng là URL của port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Vì bạn đã sử dụng port forwarding đến port 8889,
hãy sao chép dòng cuối cùng trong khung màu đỏ của :numref:`fig_jupyter`,
thay "8888" bằng "8889" trong URL,
và mở URL đó trong trình duyệt cục bộ.


## Đóng các phiên bản không sử dụng

Vì các dịch vụ cloud được tính phí theo thời gian sử dụng, bạn nên đóng những phiên bản không được sử dụng. Lưu ý rằng có các lựa chọn thay thế:

* "Stopping" một phiên bản nghĩa là bạn có thể khởi động lại phiên bản đó. Điều này tương tự như tắt nguồn máy chủ thông thường của bạn. Tuy nhiên, các phiên bản đã dừng vẫn bị tính một khoản phí nhỏ cho không gian ổ đĩa được giữ lại.
* "Terminating" một phiên bản sẽ xóa mọi dữ liệu liên quan đến phiên bản đó. Việc này bao gồm cả ổ đĩa, vì vậy bạn không thể khởi động lại phiên bản. Chỉ thực hiện việc này nếu bạn chắc chắn rằng sẽ không cần đến phiên bản đó trong tương lai.

Nếu muốn sử dụng instance làm template cho nhiều instance khác,
hãy nhấp chuột phải vào ví dụ trong :numref:`fig_connect` và chọn "Image" $\rightarrow$
"Create" để tạo image của instance. Sau khi hoàn tất, chọn
"Instance State" $\rightarrow$ "Terminate" để terminate instance. Lần tới
khi muốn sử dụng instance này, bạn có thể làm theo các bước trong phần này 
để tạo một instance dựa trên
image đã lưu. Điểm khác biệt duy nhất là trong mục "1. Choose AMI" được hiển thị trong
:numref:`fig_ubuntu`, bạn phải sử dụng tùy chọn "My AMIs" ở bên trái để chọn image đã lưu.
Instance được tạo sẽ giữ lại thông tin được lưu trên ổ đĩa của image.
Ví dụ, bạn sẽ không phải cài đặt lại CUDA và các runtime environment khác.


## Tóm tắt

* Chúng ta có thể khởi chạy và dừng các instance theo nhu cầu mà không cần mua và tự xây dựng máy tính riêng.
* Chúng ta cần cài đặt CUDA trước khi sử dụng deep learning framework hỗ trợ GPU.
* Chúng ta có thể sử dụng port forwarding để chạy Jupyter Notebook trên một máy chủ từ xa.


## Bài tập

1. Cloud mang lại sự tiện lợi, nhưng không hề rẻ. Hãy tìm hiểu cách khởi chạy [spot instances](https://aws.amazon.com/ec2/spot/) để xem cách giảm chi phí.
1. Thử nghiệm với các GPU server khác nhau. Tốc độ của chúng như thế nào?
1. Thử nghiệm với các multi-GPU server. Bạn có thể mở rộng quy mô tốt đến mức nào?


[Thảo luận](https://discuss.d2l.ai/t/423)