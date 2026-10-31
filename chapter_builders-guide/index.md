# Hướng dẫn dành cho người xây dựng
:label:`chap_computation`

Bên cạnh các tập dữ liệu khổng lồ và phần cứng mạnh mẽ,
các công cụ phần mềm tuyệt vời đã đóng một vai trò không thể thiếu
trong sự phát triển nhanh chóng của deep learning.
Bắt đầu với thư viện Theano mang tính đột phá được phát hành vào năm 2007,
các công cụ open-source linh hoạt đã cho phép các nhà nghiên cứu
nhanh chóng tạo prototype cho mô hình, tránh công việc lặp lại
khi tái sử dụng các thành phần chuẩn
mà vẫn duy trì khả năng thực hiện các sửa đổi ở mức thấp.
Theo thời gian, các thư viện deep learning đã phát triển
để cung cấp các abstraction ngày càng thô hơn.
Cũng như các nhà thiết kế chất bán dẫn đã chuyển từ việc đặc tả transistor
sang mạch logic rồi viết code,
các nhà nghiên cứu neural network đã chuyển từ việc suy nghĩ về
hành vi của từng artificial neuron
sang hình dung network dưới dạng toàn bộ layer,
và hiện nay thường thiết kế architecture với các *block* thô hơn nhiều trong đầu.


Cho đến nay, chúng ta đã giới thiệu một số khái niệm machine learning cơ bản,
dần tiến tới các mô hình deep learning hoạt động đầy đủ.
Trong chương trước,
chúng ta đã triển khai từng thành phần của một MLP từ đầu
và thậm chí còn chỉ ra cách tận dụng các API cấp cao
để triển khai cùng các mô hình đó một cách dễ dàng.
Để giúp bạn đi xa đến vậy trong thời gian ngắn như vậy, chúng ta đã *gọi đến* các thư viện,
nhưng bỏ qua các chi tiết nâng cao hơn về *cách chúng hoạt động*.
Trong chương này, chúng ta sẽ vén bức màn,
đào sâu hơn vào các thành phần then chốt của tính toán deep learning,
cụ thể là xây dựng mô hình, truy cập và khởi tạo tham số,
thiết kế layer và block tùy chỉnh, đọc và ghi mô hình vào đĩa,
cũng như tận dụng GPU để đạt được mức tăng tốc đáng kể.
Những kiến thức này sẽ đưa bạn từ *end user* thành *power user*,
cung cấp các công cụ cần thiết để hưởng lợi từ
một thư viện deep learning trưởng thành mà vẫn giữ được sự linh hoạt
để triển khai các mô hình phức tạp hơn, bao gồm cả những mô hình do chính bạn phát minh!
Mặc dù chương này không giới thiệu mô hình hoặc tập dữ liệu mới nào,
các chương về mô hình nâng cao tiếp theo phụ thuộc rất nhiều vào những kỹ thuật này.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```
