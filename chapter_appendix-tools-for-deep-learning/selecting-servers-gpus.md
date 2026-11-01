# Lựa chọn Server và GPU
:label:`sec_buy_gpu`

Việc huấn luyện deep learning nhìn chung đòi hỏi lượng tính toán rất lớn. Hiện nay, GPU là bộ tăng tốc phần cứng có hiệu quả chi phí cao nhất cho deep learning. Cụ thể, so với CPU, GPU rẻ hơn và có hiệu năng cao hơn, thường cao hơn trên một bậc độ lớn. Hơn nữa, một server có thể hỗ trợ nhiều GPU, tối đa 8 GPU đối với các server cao cấp. Con số phổ biến hơn là tối đa 4 GPU cho một workstation kỹ thuật, vì yêu cầu về nhiệt, làm mát và điện năng tăng nhanh vượt quá khả năng đáp ứng của một tòa nhà văn phòng. Đối với các triển khai lớn hơn, cloud computing (ví dụ: các instance Amazon [P3](https://aws.amazon.com/ec2/instance-types/p3/) và [G4](https://aws.amazon.com/blogs/aws/in-the-works-ec2-instances-g4-with-nvidia-t4-gpus/)) là giải pháp thực tế hơn nhiều.


## Lựa chọn Server

Thông thường không cần mua CPU cao cấp có nhiều thread, vì phần lớn phép tính được thực hiện trên GPU. Tuy nhiên, do global interpreter lock (GIL) trong Python, hiệu năng single-thread của CPU có thể quan trọng trong những trường hợp sử dụng 4--8 GPU. Khi các yếu tố khác tương đương, điều này cho thấy CPU có ít core hơn nhưng tần số clock cao hơn có thể là lựa chọn kinh tế hơn. Ví dụ, khi lựa chọn giữa CPU 6-core 4 GHz và CPU 8-core 3.5 GHz, CPU đầu tiên được ưu tiên hơn nhiều, dù tốc độ tổng hợp của nó thấp hơn.

Một yếu tố quan trọng cần cân nhắc là GPU tiêu thụ nhiều điện năng và do đó tỏa ra nhiều nhiệt. Điều này đòi hỏi hệ thống làm mát rất tốt và chassis đủ lớn để lắp GPU. Nếu có thể, hãy tuân theo các hướng dẫn dưới đây:

1. **Bộ nguồn**. GPU tiêu thụ lượng điện năng đáng kể. Hãy dự trù tối đa 350W cho mỗi thiết bị (kiểm tra *peak demand* của card đồ họa thay vì nhu cầu thông thường, vì code hiệu quả có thể sử dụng rất nhiều năng lượng). Nếu bộ nguồn không đáp ứng được nhu cầu, hệ thống sẽ trở nên không ổn định.
1. **Kích thước Chassis**. GPU có kích thước lớn và các đầu nối nguồn phụ thường cần thêm không gian. Ngoài ra, chassis lớn cũng dễ làm mát hơn.
1. **Làm mát GPU**. Nếu có nhiều GPU, bạn có thể cân nhắc đầu tư vào hệ thống làm mát bằng nước. Đồng thời, hãy ưu tiên *reference designs* ngay cả khi chúng có ít quạt hơn, vì chúng đủ mỏng để cho phép không khí đi vào giữa các thiết bị. Nếu mua GPU có nhiều quạt, GPU có thể quá dày để nhận đủ không khí khi lắp nhiều GPU, khiến hệ thống bị thermal throttling.
1. **Khe PCIe**. Việc truyền dữ liệu đến và đi từ GPU (cũng như trao đổi dữ liệu giữa các GPU) cần băng thông rất lớn. Chúng tôi khuyến nghị các khe PCIe 3.0 có 16 lane. Nếu lắp nhiều GPU, hãy đọc kỹ mô tả motherboard để đảm bảo băng thông 16$\times$ vẫn khả dụng khi nhiều GPU được sử dụng cùng lúc, đồng thời các khe bổ sung sử dụng PCIe 3.0 thay vì PCIe 2.0. Một số motherboard giảm băng thông xuống 8$\times$ hoặc thậm chí 4$\times$ khi lắp nhiều GPU. Điều này một phần là do số lượng lane PCIe mà CPU cung cấp.

Tóm lại, dưới đây là một số khuyến nghị khi xây dựng deep learning server:

* **Người mới bắt đầu**. Hãy mua một GPU cấp thấp có mức tiêu thụ điện năng thấp (các GPU gaming giá rẻ phù hợp cho deep learning thường sử dụng 150--200W). Nếu may mắn, máy tính hiện tại của bạn đã hỗ trợ GPU đó.
* **1 GPU**. CPU cấp thấp 4 core là đủ và hầu hết motherboard đều đáp ứng được. Hãy hướng tới ít nhất 32 GB DRAM và đầu tư vào SSD để truy cập dữ liệu cục bộ. Bộ nguồn 600W là đủ. Hãy mua GPU có nhiều quạt.
* **2 GPU**. CPU cấp thấp 4-6 core là đủ. Hãy hướng tới 64 GB DRAM và đầu tư vào SSD. Bạn sẽ cần khoảng 1000W cho hai GPU cao cấp. Về mainboard, hãy đảm bảo chúng có *hai* khe PCIe 3.0 x16. Nếu có thể, hãy chọn mainboard có hai khoảng trống (khoảng cách 60mm) giữa các khe PCIe 3.0 x16 để tăng lượng không khí. Trong trường hợp này, hãy mua hai GPU có nhiều quạt.
* **4 GPU**. Hãy đảm bảo mua CPU có tốc độ single-thread tương đối nhanh (tức là tần số clock cao). Có lẽ bạn sẽ cần CPU có nhiều lane PCIe hơn, chẳng hạn như AMD Threadripper. Bạn có thể sẽ cần mainboard tương đối đắt tiền để có 4 khe PCIe 3.0 x16, vì chúng có thể cần PLX để multiplex các lane PCIe. Hãy mua GPU có reference design, thân hẹp và cho phép không khí đi vào giữa các GPU. Bạn cần bộ nguồn 1600--2000W và ổ cắm điện trong văn phòng có thể không đáp ứng được mức này. Server này có thể sẽ *ồn và nóng*. Bạn không nên đặt nó dưới bàn. Khuyến nghị sử dụng 128 GB DRAM. Hãy trang bị SSD (1--2 TB NVMe) để lưu trữ cục bộ và một loạt hard disk trong cấu hình RAID để lưu dữ liệu.
* **8 GPU**. Bạn cần mua chassis server chuyên dụng cho multi-GPU với nhiều bộ nguồn dự phòng (ví dụ: 2+1 cho 1600W mỗi bộ nguồn). Cấu hình này sẽ yêu cầu CPU server dual socket, 256 GB ECC DRAM, card mạng nhanh (khuyến nghị 10 GBE), đồng thời bạn cần kiểm tra xem server có hỗ trợ *physical form factor* của GPU hay không. Luồng không khí và cách bố trí dây dẫn khác biệt đáng kể giữa GPU dành cho người dùng phổ thông và GPU dành cho server (ví dụ: RTX 2080 so với Tesla V100). Điều này có nghĩa là bạn có thể không lắp được GPU dành cho người dùng phổ thông vào server do không đủ khoảng hở cho cáp nguồn hoặc thiếu wiring harness phù hợp (như một trong các đồng tác giả đã phải đau đớn phát hiện).


## Lựa chọn GPU

Hiện nay, AMD và NVIDIA là hai nhà sản xuất GPU chuyên dụng chính. NVIDIA là hãng đầu tiên tham gia lĩnh vực deep learning và cung cấp khả năng hỗ trợ tốt hơn cho các framework deep learning thông qua CUDA. Vì vậy, phần lớn người mua chọn GPU NVIDIA.

NVIDIA cung cấp hai loại GPU, hướng đến người dùng cá nhân (ví dụ: dòng GTX và RTX) và người dùng doanh nghiệp (thông qua dòng Tesla). Hai loại GPU này có năng lực tính toán tương đương. Tuy nhiên, GPU dành cho doanh nghiệp thường sử dụng hệ thống làm mát cưỡng bức (thụ động), có nhiều memory hơn và có memory ECC (error correcting). Những GPU này phù hợp hơn với data center và thường có giá cao gấp mười lần GPU dành cho người dùng phổ thông.

Nếu là một công ty lớn có hơn 100 server, bạn nên cân nhắc dòng NVIDIA Tesla hoặc sử dụng GPU server trên cloud. Đối với một phòng thí nghiệm hoặc công ty vừa và nhỏ có hơn 10 server, dòng NVIDIA RTX có lẽ là lựa chọn hiệu quả chi phí nhất. Bạn có thể mua server được cấu hình sẵn với chassis Supermicro hoặc Asus, có thể chứa hiệu quả 4--8 GPU.

Các nhà cung cấp GPU thường phát hành một thế hệ mới sau mỗi một đến hai năm, chẳng hạn như dòng GTX 1000 (Pascal) phát hành năm 2017 và dòng RTX 2000 (Turing) phát hành năm 2019. Mỗi dòng cung cấp một số model khác nhau với các mức hiệu năng khác nhau. Hiệu năng GPU chủ yếu là sự kết hợp của ba tham số sau:

1. **Năng lực tính toán**. Nhìn chung, chúng ta quan tâm đến năng lực tính toán số thực 32-bit. Việc huấn luyện với số thực 16-bit (FP16) cũng đang trở nên phổ biến. Nếu chỉ quan tâm đến prediction, bạn cũng có thể sử dụng số nguyên 8-bit. Thế hệ GPU Turing mới nhất cung cấp khả năng tăng tốc 4-bit. Đáng tiếc là tại thời điểm viết tài liệu này, các thuật toán huấn luyện network có độ chính xác thấp vẫn chưa phổ biến.
1. **Dung lượng Memory**. Khi model lớn hơn hoặc batch được sử dụng trong quá trình huấn luyện tăng lên, bạn sẽ cần nhiều GPU memory hơn. Hãy kiểm tra loại memory HBM2 (High Bandwidth Memory) và GDDR6 (Graphics DDR). HBM2 nhanh hơn nhưng đắt hơn nhiều.
1. **Băng thông Memory**. Bạn chỉ có thể tận dụng tối đa năng lực tính toán khi có đủ băng thông memory. Nếu sử dụng GDDR6, hãy tìm các memory bus rộng.

Đối với hầu hết người dùng, chỉ cần xem xét năng lực tính toán là đủ. Lưu ý rằng nhiều GPU cung cấp các loại tăng tốc khác nhau. Ví dụ, TensorCores của NVIDIA tăng tốc một tập hợp con các operator lên 5$\times$. Hãy đảm bảo các library của bạn hỗ trợ tính năng này. GPU memory không nên nhỏ hơn 4 GB (8 GB sẽ tốt hơn nhiều). Cố gắng tránh sử dụng GPU đồng thời cho việc hiển thị GUI (thay vào đó, hãy sử dụng graphics tích hợp). Nếu không thể tránh, hãy bổ sung thêm 2 GB RAM để đảm bảo an toàn.

:numref:`fig_flopsvsprice` so sánh năng lực tính toán số thực 32-bit và giá của các model thuộc nhiều dòng GTX 900, GTX 1000 và RTX 2000. Mức giá được đề xuất là mức giá tìm thấy trên Wikipedia tại thời điểm viết tài liệu này.

![So sánh năng lực tính toán số thực và mức giá. ](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`


Chúng ta có thể thấy một số điểm sau:

1. Trong mỗi dòng, giá và hiệu năng gần như tỷ lệ thuận. Các model Titan có mức giá cao hơn đáng kể do có dung lượng GPU memory lớn hơn. Tuy nhiên, các model mới hơn có hiệu quả chi phí tốt hơn, như có thể thấy khi so sánh 980 Ti và 1080 Ti. Giá của dòng RTX 2000 dường như không được cải thiện nhiều. Tuy nhiên, nguyên nhân là vì chúng cung cấp hiệu năng low precision vượt trội hơn nhiều (FP16, INT8 và INT4).
2. Tỷ lệ hiệu năng trên chi phí của dòng GTX 1000 cao hơn khoảng hai lần so với dòng 900.
3. Đối với dòng RTX 2000, hiệu năng (tính bằng GFLOPs) là một hàm *affine* của giá.

![Năng lực tính toán số thực và mức tiêu thụ năng lượng. ](../img/wattvsprice.svg)
:label:`fig_wattvsprice`


:numref:`fig_wattvsprice` cho thấy mức tiêu thụ năng lượng tăng gần như tuyến tính theo lượng tính toán. Thứ hai, các thế hệ sau có hiệu quả cao hơn. Điều này dường như mâu thuẫn với biểu đồ tương ứng của dòng RTX 2000. Tuy nhiên, đây là hệ quả của TensorCores, vốn tiêu thụ lượng năng lượng lớn một cách không tương xứng.


## Tóm tắt

* Khi xây dựng server, hãy chú ý đến nguồn điện, lane bus PCIe, tốc độ single-thread của CPU và hệ thống làm mát.
* Nếu có thể, bạn nên mua GPU thuộc thế hệ mới nhất.
* Hãy sử dụng cloud cho các triển khai lớn.
* Server mật độ cao có thể không tương thích với mọi GPU. Hãy kiểm tra thông số cơ khí và làm mát trước khi mua.
* Sử dụng FP16 hoặc độ chính xác thấp hơn để đạt hiệu quả cao.


[Thảo luận](https://discuss.d2l.ai/t/425)