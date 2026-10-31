# Mạng Nơ-ron Tuyến tính cho Phân loại
:label:`chap_classification`

Giờ đây, khi bạn đã tìm hiểu toàn bộ các cơ chế vận hành,
bạn đã sẵn sàng áp dụng những kỹ năng đã học vào nhiều loại bài toán rộng hơn.
Ngay cả khi chúng ta chuyển sang phân loại,
phần lớn quy trình vẫn giữ nguyên:
tải dữ liệu, đưa dữ liệu qua mô hình,
tạo đầu ra, tính loss,
lấy gradient theo trọng số,
và cập nhật mô hình.
Tuy nhiên, dạng chính xác của các target,
cách tham số hóa của output layer,
và lựa chọn loss function sẽ được điều chỉnh
để phù hợp với bối cảnh *phân loại*.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```
