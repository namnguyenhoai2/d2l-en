# Học tăng cường
:label:`chap_reinforcement_learning`


**Pratik Chaudhari** (*University of Pennsylvania and Amazon*), **Rasool Fakoor** (*Amazon*) và **Kavosh Asadi** (*Amazon*)

Học tăng cường (Reinforcement Learning, RL) là một tập hợp các kỹ thuật cho phép chúng ta xây dựng những hệ thống machine learning đưa ra quyết định một cách tuần tự. Ví dụ, một gói hàng chứa quần áo mới mà bạn mua từ một nhà bán lẻ trực tuyến sẽ đến trước cửa nhà bạn sau một chuỗi quyết định, chẳng hạn như nhà bán lẻ tìm quần áo trong kho gần nhà bạn nhất, cho quần áo vào hộp, vận chuyển hộp bằng đường bộ hoặc đường hàng không, rồi giao đến nhà bạn trong thành phố. Có nhiều biến số ảnh hưởng đến quá trình giao gói hàng, chẳng hạn như quần áo có sẵn trong kho hay không, mất bao lâu để vận chuyển hộp, hộp có đến thành phố của bạn trước khi xe giao hàng hằng ngày rời đi hay không, v.v. Ý tưởng then chốt là ở mỗi giai đoạn, những biến số mà chúng ta thường không kiểm soát được này sẽ ảnh hưởng đến toàn bộ chuỗi sự kiện trong tương lai; ví dụ, nếu việc đóng gói hộp trong kho bị trì hoãn, nhà bán lẻ có thể cần gửi gói hàng bằng đường hàng không thay vì đường bộ để bảo đảm giao hàng đúng hạn. Các phương pháp Học tăng cường cho phép chúng ta thực hiện hành động phù hợp ở mỗi giai đoạn của một bài toán ra quyết định tuần tự nhằm cuối cùng tối đa hóa một tiện ích nào đó, chẳng hạn như giao gói hàng đến bạn đúng hạn.

Những bài toán ra quyết định tuần tự như vậy xuất hiện ở nhiều nơi khác, chẳng hạn như khi chơi [Go](https://en.wikipedia.org/wiki/Go_(game), nước đi hiện tại của bạn quyết định các nước đi tiếp theo, còn nước đi của đối thủ là những biến số mà bạn không thể kiểm soát... một chuỗi các nước đi cuối cùng quyết định bạn có thắng hay không; những bộ phim mà Netflix đề xuất cho bạn lúc này quyết định bạn xem gì, việc bạn có thích bộ phim đó hay không là điều Netflix không biết, và cuối cùng một chuỗi các đề xuất phim quyết định mức độ hài lòng của bạn với Netflix. Học tăng cường đang được sử dụng ngày nay để phát triển các giải pháp hiệu quả cho những bài toán này :cite:`mnih2013playing,Silver.Huang.Maddison.ea.2016`. Điểm khác biệt then chốt giữa học tăng cường và deep learning tiêu chuẩn là trong deep learning tiêu chuẩn, dự đoán của một model đã được huấn luyện trên một dữ liệu kiểm thử không ảnh hưởng đến các dự đoán trên dữ liệu kiểm thử trong tương lai; còn trong học tăng cường, các quyết định ở những thời điểm tương lai (trong RL, các quyết định còn được gọi là action) bị ảnh hưởng bởi những quyết định đã được đưa ra trong quá khứ.

Trong chương này, chúng ta sẽ phát triển các nền tảng của học tăng cường và có kinh nghiệm thực hành trong việc triển khai một số phương pháp học tăng cường phổ biến. Trước tiên, chúng ta sẽ phát triển một khái niệm gọi là Markov Decision Process (MDP), cho phép chúng ta tư duy về những bài toán ra quyết định tuần tự như vậy. Một thuật toán có tên Value Iteration sẽ là hiểu biết đầu tiên của chúng ta về cách giải các bài toán học tăng cường, với giả định rằng chúng ta biết các biến số không chịu sự kiểm soát trong một MDP (trong RL, các biến này được gọi là environment) thường hoạt động như thế nào. Sử dụng phiên bản tổng quát hơn của Value Iteration, một thuật toán có tên Q-Learning, chúng ta sẽ có thể thực hiện các action phù hợp ngay cả khi không nhất thiết biết đầy đủ về environment. Sau đó, chúng ta sẽ nghiên cứu cách sử dụng các mạng deep network cho các bài toán học tăng cường bằng cách bắt chước các action của một expert. Cuối cùng, chúng ta sẽ phát triển một phương pháp học tăng cường sử dụng deep network để thực hiện action trong những environment chưa biết. Những kỹ thuật này tạo nền tảng cho các thuật toán RL nâng cao hơn đang được sử dụng ngày nay trong nhiều ứng dụng thực tế; một số ứng dụng trong đó sẽ được đề cập trong chương này.

![Cấu trúc của học tăng cường](../img/RL_main.png)
:width:`400px`
:label:`fig_rl_big`

```toc
:maxdepth: 2

mdp
value-iter
qlearning
```
