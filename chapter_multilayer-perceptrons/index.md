# Perceptron đa tầng
:label:`chap_perceptrons`

Trong chương này, chúng ta sẽ giới thiệu mạng *sâu* thực sự đầu tiên của bạn.
Các mạng sâu đơn giản nhất được gọi là *perceptron đa tầng*,
và chúng bao gồm nhiều tầng nơ-ron,
mỗi tầng được kết nối đầy đủ với các nơ-ron ở tầng bên dưới
(từ đó chúng nhận đầu vào)
và các nơ-ron ở tầng bên trên (mà chúng lần lượt tác động đến).
Mặc dù vi phân tự động
đơn giản hóa đáng kể việc triển khai các thuật toán deep learning,
chúng ta sẽ đi sâu vào cách các gradient này
được tính toán trong các mạng sâu.
Sau đó, chúng ta sẽ
sẵn sàng để
thảo luận về các vấn đề liên quan đến tính ổn định số và khởi tạo tham số,
những yếu tố then chốt để huấn luyện thành công các mạng sâu.
Khi huấn luyện các mô hình có năng lực cao như vậy, chúng ta có nguy cơ overfitting. Vì thế, chúng ta sẽ
xem lại regularization và khả năng khái quát hóa
đối với các mạng sâu.
Xuyên suốt chương này, chúng tôi hướng tới
việc giúp bạn nắm vững không chỉ các khái niệm mà còn cả thực hành sử dụng mạng sâu.
Ở cuối chương, chúng ta áp dụng những gì đã giới thiệu cho đến nay vào một trường hợp thực tế: dự đoán giá nhà.
Các vấn đề liên quan đến hiệu năng tính toán, khả năng mở rộng và hiệu quả
của các mô hình sẽ được để lại cho các chương tiếp theo.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```
