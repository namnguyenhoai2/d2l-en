# Phụ lục: Toán học cho Deep Learning
:label:`chap_appendix_math`

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*) và các tác giả của cuốn sách này


Một trong những điểm tuyệt vời của deep learning hiện đại là phần lớn nội dung của nó có thể được hiểu và sử dụng mà không cần nắm vững toàn bộ nền tảng toán học bên dưới. Đây là dấu hiệu cho thấy lĩnh vực này đang trưởng thành. Cũng như hầu hết các nhà phát triển phần mềm hiện nay không còn cần bận tâm đến lý thuyết về các hàm tính được, những người làm deep learning cũng không nhất thiết phải lo lắng về các nền tảng lý thuyết của việc học theo phương pháp cực đại khả năng.

Tuy nhiên, chúng ta vẫn chưa hoàn toàn đạt đến mức đó.

Trong thực tế, đôi khi bạn sẽ cần hiểu cách các lựa chọn về kiến trúc ảnh hưởng đến dòng gradient, hoặc những giả định ngầm mà bạn đưa ra khi huấn luyện với một hàm mất mát nhất định. Bạn có thể cần biết entropy thực sự đo lường điều gì, và nó có thể giúp bạn hiểu chính xác ý nghĩa của bits-per-character trong mô hình của mình như thế nào. Tất cả những điều này đều đòi hỏi sự hiểu biết sâu hơn về toán học.

Phụ lục này nhằm cung cấp cho bạn nền tảng toán học cần thiết để hiểu lý thuyết cốt lõi của deep learning hiện đại, nhưng không có tính bao quát đầy đủ. Trước tiên, chúng ta sẽ tìm hiểu sâu hơn về đại số tuyến tính. Chúng ta xây dựng cách hiểu hình học về tất cả các đối tượng và phép toán đại số tuyến tính thường gặp, qua đó có thể hình dung tác động của nhiều phép biến đổi khác nhau lên dữ liệu. Một nội dung then chốt là xây dựng những kiến thức cơ bản về phân rã riêng (eigen-decomposition).

Tiếp theo, chúng ta phát triển lý thuyết giải tích vi phân đến mức có thể hiểu đầy đủ vì sao gradient là hướng giảm dốc nhất, và vì sao back-propagation lại có dạng như vậy. Sau đó, giải tích tích phân được thảo luận ở mức độ cần thiết để hỗ trợ chủ đề tiếp theo của chúng ta: lý thuyết xác suất.

Các bài toán gặp trong thực tế thường không mang tính chắc chắn, vì vậy chúng ta cần một ngôn ngữ để diễn đạt về những điều không chắc chắn. Chúng ta sẽ ôn lại lý thuyết về biến ngẫu nhiên và những phân phối thường gặp nhất, để có thể thảo luận về các mô hình theo cách tiếp cận xác suất. Đây là nền tảng cho bộ phân loại naive Bayes, một kỹ thuật phân loại xác suất.

Một lĩnh vực có mối liên hệ chặt chẽ với lý thuyết xác suất là thống kê. Mặc dù thống kê là một lĩnh vực quá rộng để có thể trình bày đầy đủ trong một phần ngắn, chúng ta sẽ giới thiệu những khái niệm nền tảng mà mọi người làm machine learning nên biết, đặc biệt là: đánh giá và so sánh các bộ ước lượng, tiến hành kiểm định giả thuyết và xây dựng khoảng tin cậy.

Cuối cùng, chúng ta chuyển sang chủ đề lý thuyết thông tin, lĩnh vực nghiên cứu toán học về việc lưu trữ và truyền tải thông tin. Đây là ngôn ngữ cốt lõi giúp chúng ta thảo luận một cách định lượng về lượng thông tin mà một mô hình nắm giữ trong một miền diễn ngôn.

Kết hợp lại, các nội dung này tạo thành những khái niệm toán học cốt lõi cần thiết để bắt đầu tiến tới sự hiểu biết sâu sắc về deep learning.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```
