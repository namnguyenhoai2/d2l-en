# Mạng nơ-ron tuyến tính cho bài toán hồi quy
:label:`chap_regression`

Trước khi quan tâm đến việc xây dựng các mạng nơ-ron sâu,
sẽ rất hữu ích nếu triển khai một số mạng nông,
trong đó đầu vào kết nối trực tiếp với đầu ra.
Điều này quan trọng vì một vài lý do.
Thứ nhất, thay vì bị phân tâm bởi các kiến trúc phức tạp,
chúng ta có thể tập trung vào những kiến thức cơ bản về huấn luyện mạng nơ-ron,
bao gồm tham số hóa lớp đầu ra, xử lý dữ liệu,
xác định hàm mất mát và huấn luyện mô hình.
Thứ hai, lớp mạng nông này tình cờ bao gồm tập hợp các mô hình tuyến tính,
vốn bao hàm nhiều phương pháp dự đoán thống kê kinh điển,
bao gồm hồi quy tuyến tính và hồi quy softmax.
Việc hiểu các công cụ kinh điển này có ý nghĩa then chốt
vì chúng được sử dụng rộng rãi trong nhiều bối cảnh
và chúng ta thường cần dùng chúng làm baseline
khi biện minh cho việc sử dụng các kiến trúc phức tạp hơn.
Chương này sẽ tập trung chuyên sâu vào hồi quy tuyến tính,
còn chương tiếp theo sẽ mở rộng phạm vi mô hình hóa của chúng ta
bằng cách phát triển các mạng nơ-ron tuyến tính cho bài toán phân loại.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```
