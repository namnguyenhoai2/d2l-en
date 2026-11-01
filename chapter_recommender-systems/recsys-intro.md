# Tổng quan về Hệ thống Gợi ý



Trong thập kỷ qua, Internet đã phát triển thành một nền tảng cho các dịch vụ trực tuyến quy mô lớn, làm thay đổi sâu sắc cách chúng ta giao tiếp, đọc tin tức, mua sản phẩm và xem phim. Đồng thời, số lượng *item* chưa từng có (chúng tôi dùng thuật ngữ *item* để chỉ phim, tin tức, sách và sản phẩm) được cung cấp trực tuyến đòi hỏi một hệ thống có thể giúp chúng ta khám phá những item mà mình yêu thích. Do đó, hệ thống gợi ý là những công cụ lọc thông tin mạnh mẽ, có thể hỗ trợ các dịch vụ cá nhân hóa và cung cấp trải nghiệm phù hợp cho từng người dùng. Tóm lại, hệ thống gợi ý đóng vai trò then chốt trong việc tận dụng nguồn dữ liệu dồi dào hiện có để giúp việc lựa chọn trở nên dễ dàng hơn. Ngày nay, hệ thống gợi ý là nền tảng cốt lõi của nhiều nhà cung cấp dịch vụ trực tuyến như Amazon, Netflix và YouTube. Hãy nhớ lại ví dụ về những cuốn sách Deep learning được Amazon gợi ý trong :numref:`subsec_recommender_systems`. Lợi ích của việc sử dụng hệ thống gợi ý có hai mặt: Một mặt, hệ thống có thể giảm đáng kể công sức của người dùng trong việc tìm kiếm item và giảm bớt vấn đề quá tải thông tin. Mặt khác, hệ thống có thể tạo thêm giá trị kinh doanh cho các nhà cung cấp dịch vụ trực tuyến và là một nguồn doanh thu quan trọng. Chương này sẽ giới thiệu các khái niệm nền tảng, những mô hình kinh điển và các tiến bộ gần đây với deep learning trong lĩnh vực hệ thống gợi ý, cùng với các ví dụ đã được triển khai.

![Minh họa về Quy trình Gợi ý](../img/rec-intro.svg)


## Collaborative Filtering

Chúng ta bắt đầu hành trình với khái niệm quan trọng trong hệ thống gợi ý — collaborative filtering (CF), lần đầu được đặt tên bởi hệ thống Tapestry :cite:`Goldberg.Nichols.Oki.ea.1992`, với ý nghĩa “mọi người cộng tác để giúp nhau thực hiện quá trình lọc nhằm xử lý lượng lớn email và tin nhắn được đăng lên các newsgroup”. Thuật ngữ này dần được mở rộng với nhiều ý nghĩa hơn. Theo nghĩa rộng, đây là quá trình lọc thông tin hoặc mẫu bằng các kỹ thuật có sự cộng tác giữa nhiều người dùng, agent và nguồn dữ liệu. CF có nhiều dạng và đã có rất nhiều phương pháp CF được đề xuất kể từ khi xuất hiện.

Nhìn chung, các kỹ thuật CF có thể được phân loại thành: CF dựa trên bộ nhớ, CF dựa trên mô hình và các phương pháp hybrid của chúng :cite:`Su.Khoshgoftaar.2009`. Các kỹ thuật CF dựa trên bộ nhớ tiêu biểu là CF dựa trên nearest neighbor, chẳng hạn như CF dựa trên user và CF dựa trên item :cite:`Sarwar.Karypis.Konstan.ea.2001`. Các mô hình latent factor như matrix factorization là những ví dụ về CF dựa trên mô hình. CF dựa trên bộ nhớ có những hạn chế khi xử lý dữ liệu thưa và quy mô lớn, vì nó tính các giá trị tương đồng dựa trên những item chung. Các phương pháp dựa trên mô hình ngày càng phổ biến hơn nhờ khả năng xử lý tính thưa và khả năng mở rộng tốt hơn. Nhiều phương pháp CF dựa trên mô hình có thể được mở rộng bằng neural network, tạo ra các mô hình linh hoạt và có khả năng mở rộng hơn, với tốc độ tính toán được tăng cường nhờ deep learning :cite:`Zhang.Yao.Sun.ea.2019`. Nhìn chung, CF chỉ sử dụng dữ liệu tương tác user-item để đưa ra dự đoán và gợi ý. Ngoài CF, các hệ thống gợi ý dựa trên nội dung và dựa trên ngữ cảnh cũng hữu ích trong việc kết hợp mô tả nội dung của item/user và các tín hiệu ngữ cảnh như timestamp và location. Rõ ràng, chúng ta có thể cần điều chỉnh loại hoặc cấu trúc mô hình khi có các dữ liệu đầu vào khác nhau.



## Phản hồi Tường minh và Phản hồi Ngầm định

Để học sở thích của người dùng, hệ thống cần thu thập phản hồi từ họ. Phản hồi có thể là tường minh hoặc ngầm định :cite:`Hu.Koren.Volinsky.2008`. Ví dụ, [IMDb](https://www.imdb.com/) thu thập đánh giá theo thang điểm từ một đến mười sao cho các bộ phim. YouTube cung cấp các nút thích và không thích để người dùng thể hiện sở thích của mình. Rõ ràng, việc thu thập phản hồi tường minh đòi hỏi người dùng phải chủ động thể hiện mối quan tâm của họ. Tuy nhiên, phản hồi tường minh không phải lúc nào cũng sẵn có, vì nhiều người dùng có thể không muốn đánh giá sản phẩm. So với phản hồi tường minh, phản hồi ngầm định thường sẵn có hơn, vì nó chủ yếu liên quan đến việc mô hình hóa hành vi ngầm định như thao tác nhấp của người dùng. Vì vậy, nhiều hệ thống gợi ý tập trung vào phản hồi ngầm định, vốn gián tiếp phản ánh ý kiến của người dùng thông qua việc quan sát hành vi của họ. Phản hồi ngầm định có nhiều dạng, bao gồm lịch sử mua hàng, lịch sử duyệt, các nội dung đã xem và thậm chí cả chuyển động chuột. Ví dụ, một người dùng đã mua nhiều cuốn sách của cùng một tác giả có thể rất yêu thích tác giả đó. Lưu ý rằng phản hồi ngầm định vốn chứa nhiều nhiễu. Chúng ta chỉ có thể *đoán* sở thích và động cơ thực sự của họ. Việc một người dùng xem một bộ phim không nhất thiết cho thấy họ có đánh giá tích cực về bộ phim đó.



## Các Nhiệm vụ Gợi ý

Trong vài thập kỷ qua, nhiều nhiệm vụ gợi ý đã được nghiên cứu. Dựa trên lĩnh vực ứng dụng, có các nhiệm vụ gợi ý phim, gợi ý tin tức, gợi ý điểm ưa thích :cite:`Ye.Yin.Lee.ea.2011` và nhiều nhiệm vụ khác. Cũng có thể phân biệt các nhiệm vụ dựa trên loại phản hồi và dữ liệu đầu vào; ví dụ, nhiệm vụ dự đoán đánh giá nhằm dự đoán các đánh giá tường minh. Gợi ý Top-$n$ (xếp hạng item) xếp hạng tất cả các item cho từng người dùng dựa trên phản hồi ngầm định. Nếu cũng có thông tin timestamp, chúng ta có thể xây dựng hệ thống gợi ý nhận biết chuỗi :cite:`Quadrana.Cremonesi.Jannach.2018`. Một nhiệm vụ phổ biến khác là dự đoán click-through rate, cũng dựa trên phản hồi ngầm định nhưng có thể sử dụng nhiều đặc trưng phân loại khác nhau. Việc gợi ý cho người dùng mới và gợi ý item mới cho những người dùng hiện tại được gọi là gợi ý cold-start :cite:`Schein.Popescul.Ungar.ea.2002`.



## Tóm tắt

* Hệ thống gợi ý rất quan trọng đối với người dùng cá nhân và các ngành công nghiệp. Collaborative filtering là một khái niệm then chốt trong việc gợi ý.
* Có hai loại phản hồi: phản hồi ngầm định và phản hồi tường minh. Nhiều nhiệm vụ gợi ý đã được nghiên cứu trong thập kỷ qua.

## Bài tập

1. Bạn có thể giải thích hệ thống gợi ý ảnh hưởng đến cuộc sống hằng ngày của mình như thế nào không?
2. Bạn nghĩ những nhiệm vụ gợi ý thú vị nào có thể được nghiên cứu?

[Thảo luận](https://discuss.d2l.ai/t/398)