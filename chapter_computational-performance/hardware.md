# Phần cứng
:label:`sec_hardware`

Việc xây dựng các hệ thống có hiệu năng cao đòi hỏi hiểu rõ về các thuật toán và mô hình để nắm bắt các khía cạnh thống kê của bài toán. Đồng thời, việc có ít nhất một chút kiến thức về phần cứng bên dưới cũng là điều không thể thiếu. Phần này không thể thay thế cho một khóa học bài bản về phần cứng và thiết kế hệ thống. Thay vào đó, phần này có thể là điểm khởi đầu để hiểu tại sao một số thuật toán lại hiệu quả hơn những thuật toán khác và làm thế nào để đạt được throughput tốt. Một thiết kế tốt có thể dễ dàng tạo ra khác biệt một bậc độ lớn và ngược lại, điều này có thể quyết định việc có thể train một network (ví dụ: trong một tuần) hay hoàn toàn không thể (trong 3 tháng, do đó bỏ lỡ deadline).
Chúng ta sẽ bắt đầu bằng việc xem xét máy tính. Sau đó, chúng ta sẽ phóng to để xem xét kỹ hơn về CPU và GPU. Cuối cùng, chúng ta sẽ thu nhỏ lại để xem xét cách nhiều máy tính được kết nối trong một server center hoặc trên cloud.

![Các con số về latency mà mọi lập trình viên nên biết.](../img/latencynumbers.png)
:label:`fig_latencynumbers`

Những độc giả thiếu kiên nhẫn có thể chỉ cần xem :numref:`fig_latencynumbers`. Nội dung này được lấy từ [bài viết tương tác](https://people.eecs.berkeley.edu/%7Ercs/research/interactive_latency.html) của Colin Scott, bài viết cung cấp một cái nhìn tổng quan hữu ích về những tiến bộ trong thập kỷ qua. Các con số ban đầu là từ [bài nói chuyện tại Stanford của Jeff Dean năm 2010](https://static.googleusercontent.com/media/research.google.com/en//people/jeff/Stanford-DL-Nov-2010.pdf).
Phần thảo luận dưới đây giải thích một số cơ sở lý luận cho các con số này và cách chúng có thể định hướng chúng ta trong việc thiết kế thuật toán. Phần thảo luận dưới đây có mức độ khái quát rất cao và chỉ mang tính lướt qua. Rõ ràng, phần này *không thể thay thế* cho một khóa học bài bản, mà chỉ nhằm cung cấp đủ thông tin để một người lập mô hình thống kê có thể đưa ra các quyết định thiết kế phù hợp. Để có cái nhìn chuyên sâu về kiến trúc máy tính, chúng tôi giới thiệu :cite:`Hennessy.Patterson.2011` hoặc một khóa học gần đây về chủ đề này, chẳng hạn như khóa học của [Arste Asanovic](http://inst.eecs.berkeley.edu/%7Ecs152/sp19/).

## Máy tính

Hầu hết các nhà nghiên cứu và người thực hành deep learning đều có quyền truy cập vào một máy tính với lượng memory và năng lực tính toán đáng kể, một dạng accelerator như GPU hoặc nhiều accelerator như vậy. Một máy tính bao gồm các thành phần chính sau:

* Một processor (còn gọi là CPU) có khả năng thực thi các chương trình mà chúng ta cung cấp (ngoài việc chạy một operating system và nhiều tác vụ khác), thường gồm 8 core trở lên.
* Memory (RAM) để lưu trữ và truy xuất các kết quả tính toán, chẳng hạn như các weight vector và activation, cùng dữ liệu training.
* Một kết nối mạng Ethernet (đôi khi có nhiều kết nối) với tốc độ từ 1 GB/s đến 100 GB/s. Trên các server cao cấp, có thể tìm thấy những interconnect tiên tiến hơn.
* Một expansion bus tốc độ cao (PCIe) để kết nối hệ thống với một hoặc nhiều GPU. Server có tối đa 8 accelerator, thường được kết nối theo một topology tiên tiến, trong khi các hệ thống desktop có 1 hoặc 2 accelerator, tùy thuộc vào ngân sách của người dùng và công suất của power supply.
* Durable storage, chẳng hạn như hard disk drive từ tính hoặc solid state drive, trong nhiều trường hợp được kết nối bằng bus PCIe. Thành phần này cung cấp khả năng truyền dữ liệu training vào hệ thống và lưu trữ các checkpoint trung gian khi cần.

![Kết nối giữa các thành phần của một máy tính.](../img/mobo-symbol.svg)
:label:`fig_mobo-symbol`

Như :numref:`fig_mobo-symbol` cho thấy, hầu hết các thành phần (network, GPU và storage) được kết nối với CPU thông qua bus PCIe. Bus này gồm nhiều lane được gắn trực tiếp vào CPU. Ví dụ, AMD Threadripper 3 có 64 lane PCIe 4.0, mỗi lane có khả năng truyền dữ liệu 16 Gbit/s theo cả hai hướng. Memory được gắn trực tiếp vào CPU với tổng bandwidth lên đến 100 GB/s.

Khi chạy code trên máy tính, chúng ta cần chuyển dữ liệu đến các processor (CPU hoặc GPU), thực hiện tính toán, sau đó chuyển kết quả ra khỏi processor trở lại RAM và durable storage. Vì vậy, để đạt được hiệu năng tốt, chúng ta cần bảo đảm quy trình này diễn ra trơn tru mà không có hệ thống nào trở thành bottleneck chính. Ví dụ, nếu không thể load image đủ nhanh thì processor sẽ không có việc để làm. Tương tự, nếu không thể chuyển matrix đủ nhanh đến CPU (hoặc GPU), các processing element của nó sẽ bị thiếu dữ liệu. Cuối cùng, nếu muốn đồng bộ hóa nhiều máy tính qua network, network không được làm chậm quá trình tính toán. Một lựa chọn là xen kẽ việc giao tiếp và tính toán. Hãy xem xét chi tiết hơn các thành phần khác nhau.

## Memory

Ở mức cơ bản nhất, memory được dùng để lưu trữ dữ liệu cần có thể truy cập nhanh. Hiện nay, RAM của CPU thường thuộc loại [DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM), cung cấp bandwidth 20--25 GB/s cho mỗi module. Mỗi module có bus rộng 64 bit. Thông thường, các cặp module memory được sử dụng để hỗ trợ nhiều channel. CPU có từ 2 đến 4 channel memory, tức là có peak memory bandwidth từ 4 0GB/s đến 100 GB/s. Thường có hai bank cho mỗi channel. Ví dụ, AMD Zen 3 Threadripper có 8 slot.

Mặc dù những con số này rất ấn tượng, nhưng thực tế chúng chỉ cho biết một phần câu chuyện. Khi muốn đọc một phần dữ liệu từ memory, trước tiên chúng ta cần cho module memory biết thông tin đó nằm ở đâu. Nói cách khác, trước tiên chúng ta cần gửi *địa chỉ* đến RAM. Sau khi hoàn tất, chúng ta có thể chọn đọc chỉ một record 64 bit hoặc một chuỗi record dài. Cách sau được gọi là *burst read*. Tóm lại, việc gửi địa chỉ đến memory và thiết lập quá trình truyền mất khoảng 100 ns (chi tiết phụ thuộc vào các hệ số timing cụ thể của các chip memory được sử dụng), còn mỗi lần truyền tiếp theo chỉ mất 0.2 ns. Nói ngắn gọn, lần đọc đầu tiên tốn kém hơn các lần đọc tiếp theo 500 lần! Lưu ý rằng chúng ta có thể thực hiện tối đa 10,000,000 lần đọc ngẫu nhiên mỗi giây. Điều này cho thấy chúng ta nên tránh truy cập memory ngẫu nhiên nhiều nhất có thể và thay vào đó sử dụng burst read (và burst write).

Mọi việc phức tạp hơn một chút khi tính đến việc chúng ta có nhiều *bank*. Mỗi bank có thể đọc memory gần như độc lập. Điều này dẫn đến hai hệ quả.
Một mặt, số lần đọc ngẫu nhiên hiệu dụng có thể cao hơn tối đa 4 lần, với điều kiện chúng được phân bổ đồng đều trên memory. Điều đó cũng có nghĩa là việc đọc ngẫu nhiên vẫn là một ý tưởng tồi, vì burst read cũng nhanh hơn 4 lần. Mặt khác, do memory được căn chỉnh theo các ranh giới 64 bit, việc căn chỉnh mọi data structure theo cùng các ranh giới đó là một ý tưởng tốt. Compiler thực hiện việc này khá [tự động](https://en.wikipedia.org/wiki/Data_structure_alignment) khi các flag thích hợp được thiết lập. Những độc giả tò mò được khuyến khích xem một bài giảng về DRAM, chẳng hạn như bài giảng của [Zeshan Chishti](http://web.cecs.pdx.edu/%7Ezeshan/ece585_lec5.pdf).

Memory của GPU phải đáp ứng yêu cầu bandwidth thậm chí còn cao hơn vì GPU có nhiều processing element hơn CPU. Nhìn chung, có hai lựa chọn để đáp ứng yêu cầu này. Lựa chọn đầu tiên là làm bus memory rộng hơn đáng kể. Ví dụ, NVIDIA RTX 2080 Ti có bus rộng 352 bit. Điều này cho phép truyền nhiều thông tin hơn cùng lúc. Thứ hai, GPU sử dụng loại memory hiệu năng cao chuyên dụng. Các thiết bị cấp consumer, chẳng hạn như dòng RTX và Titan của NVIDIA, thường sử dụng chip [GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM) với aggregate bandwidth trên 500 GB/s. Một lựa chọn khác là sử dụng module HBM (high bandwidth memory). Chúng sử dụng một interface rất khác và kết nối trực tiếp với GPU trên một silicon wafer chuyên dụng. Điều này khiến chúng rất đắt, và việc sử dụng chúng thường chỉ giới hạn ở các chip server cao cấp, chẳng hạn như dòng accelerator NVIDIA Volta V100. Không có gì đáng ngạc nhiên khi memory của GPU thường *nhỏ hơn nhiều* so với memory của CPU do chi phí cao hơn của loại memory này. Với mục đích của chúng ta, nhìn chung các đặc tính hiệu năng của chúng tương tự nhau, chỉ là nhanh hơn rất nhiều. Chúng ta có thể bỏ qua các chi tiết này trong phạm vi cuốn sách. Chúng chỉ quan trọng khi tuning GPU kernel để đạt throughput cao.

## Storage

Chúng ta đã thấy rằng một số đặc tính chính của RAM là *bandwidth* và *latency*. Điều tương tự cũng đúng với các thiết bị storage, chỉ có điều sự khác biệt có thể còn lớn hơn nhiều.

### Hard Disk Drive

*Hard disk drive* (HDD) đã được sử dụng trong hơn nửa thế kỷ. Tóm lại, chúng chứa một số platter quay cùng các head có thể được định vị để đọc hoặc ghi trên bất kỳ track nào. Các disk cao cấp có thể lưu trữ tối đa 16 TB trên 9 platter. Một trong những lợi ích chính của HDD là giá tương đối rẻ. Một trong nhiều nhược điểm của chúng là các chế độ hỏng hóc thường mang tính thảm họa và read latency tương đối cao.

Để hiểu điều sau, hãy xét thực tế rằng HDD quay ở khoảng 7.200 RPM (vòng/phút). Nếu quay nhanh hơn nhiều, chúng sẽ vỡ do lực ly tâm tác động lên các platter. Điều này gây ra một nhược điểm lớn khi truy cập một sector cụ thể trên đĩa: chúng ta cần chờ platter quay đến đúng vị trí (có thể di chuyển đầu đọc nhưng không thể tăng tốc các đĩa thực tế). Vì vậy, có thể mất hơn 8 ms trước khi dữ liệu được yêu cầu sẵn sàng. Một cách diễn đạt phổ biến là HDD có thể hoạt động ở khoảng 100 IOPs (thao tác input/output mỗi giây). Con số này về cơ bản vẫn không thay đổi trong hai thập kỷ qua. Tệ hơn nữa, việc tăng bandwidth cũng khó không kém (ở mức khoảng 100--200 MB/s). Xét cho cùng, mỗi đầu đọc một track bit, do đó tốc độ bit chỉ tăng theo căn bậc hai của mật độ thông tin. Vì vậy, HDD đang nhanh chóng bị giới hạn vào vai trò lưu trữ lưu trữ dài hạn và lưu trữ cấp thấp cho các tập dữ liệu rất lớn.

### Ổ đĩa thể rắn

Ổ đĩa thể rắn (SSD) sử dụng flash memory để lưu trữ thông tin một cách bền vững. Điều này cho phép *truy cập nhanh hơn nhiều* vào các bản ghi đã lưu trữ. SSD hiện đại có thể hoạt động ở mức 100.000 đến 500.000 IOPs, tức là nhanh hơn HDD tới 3 bậc độ lớn. Hơn nữa, bandwidth của chúng có thể đạt 1--3GB/s, tức là nhanh hơn HDD một bậc độ lớn. Những cải thiện này nghe gần như quá tốt để có thể là sự thật. Thực tế, chúng đi kèm với các điểm cần lưu ý sau đây, do cách SSD được thiết kế.

* SSD lưu trữ thông tin trong các block (256 KB hoặc lớn hơn). Chúng chỉ có thể được ghi toàn bộ, việc này mất khá nhiều thời gian. Do đó, các thao tác ghi ngẫu nhiên theo bit trên SSD có hiệu năng rất kém. Tương tự, việc ghi dữ liệu nói chung cũng mất nhiều thời gian vì block phải được đọc, xóa rồi ghi lại với thông tin mới. Hiện nay, các bộ điều khiển và firmware của SSD đã phát triển các thuật toán để giảm thiểu vấn đề này. Tuy vậy, thao tác ghi có thể chậm hơn nhiều, đặc biệt đối với SSD QLC (quad level cell). Chìa khóa để cải thiện hiệu năng là duy trì một *queue* các thao tác, ưu tiên thao tác đọc và ghi theo các block lớn nếu có thể.
* Các ô nhớ trong SSD bị hao mòn tương đối nhanh (thường chỉ sau vài nghìn lần ghi). Các thuật toán bảo vệ phân bổ mức hao mòn có thể phân tán sự suy giảm trên nhiều ô nhớ. Tuy nhiên, không nên sử dụng SSD cho các file swap hoặc cho việc tổng hợp lượng lớn log file.
* Cuối cùng, sự gia tăng bandwidth rất lớn đã buộc các nhà thiết kế máy tính phải gắn SSD trực tiếp vào bus PCIe. Các ổ đĩa có khả năng xử lý việc này, được gọi là NVMe (Non Volatile Memory enhanced), có thể sử dụng tối đa 4 lane PCIe. Điều này tương đương tối đa 8GB/s trên PCIe 4.0.

### Lưu trữ đám mây

Lưu trữ đám mây cung cấp một khoảng hiệu năng có thể cấu hình. Nghĩa là, việc phân bổ bộ nhớ cho các máy ảo là động, cả về dung lượng lẫn tốc độ, tùy theo lựa chọn của người dùng. Chúng tôi khuyến nghị người dùng tăng số IOPs được cấp bất cứ khi nào độ trễ quá cao, chẳng hạn trong quá trình training với nhiều bản ghi nhỏ.

## CPU

Bộ xử lý trung tâm (CPU) là thành phần trung tâm của mọi máy tính. Chúng bao gồm một số thành phần chính: *các lõi processor* có khả năng thực thi machine code, một *bus* kết nối chúng (topology cụ thể khác nhau đáng kể giữa các model processor, thế hệ và nhà cung cấp), và *các cache* cho phép truy cập bộ nhớ với bandwidth cao hơn và độ trễ thấp hơn so với việc đọc từ main memory. Cuối cùng, gần như tất cả CPU hiện đại đều chứa *các đơn vị xử lý vector* để hỗ trợ đại số tuyến tính hiệu năng cao và các phép tích chập, vì chúng thường được sử dụng trong xử lý media và machine learning.

![CPU quad-core dành cho người dùng Intel Skylake.](../img/skylake.svg)
:label:`fig_skylake`

:numref:`fig_skylake` mô tả một CPU quad-core cấp người dùng Intel Skylake. CPU này có GPU tích hợp, các cache và một ringbus kết nối bốn lõi. Các thiết bị ngoại vi, chẳng hạn như Ethernet, WiFi, Bluetooth, bộ điều khiển SSD và USB, либо là một phần của chipset hoặc được gắn trực tiếp (PCIe) vào CPU.


### Microarchitecture

Mỗi lõi processor bao gồm một tập hợp thành phần khá phức tạp. Mặc dù chi tiết khác nhau giữa các thế hệ và nhà cung cấp, chức năng cơ bản nhìn chung khá tiêu chuẩn. Front-end tải các instruction và cố gắng dự đoán nhánh nào sẽ được chọn (ví dụ: đối với control flow). Sau đó, instruction được giải mã từ assembly code thành microinstruction. Assembly code thường không phải là code ở cấp thấp nhất mà processor thực thi. Thay vào đó, các instruction phức tạp có thể được giải mã thành một tập hợp các operation ở cấp thấp hơn. Sau đó, chúng được xử lý bởi execution core thực tế. Execution core thường có khả năng thực hiện nhiều operation đồng thời. Chẳng hạn, lõi ARM Cortex A77 của :numref:`fig_cortexa77` có thể thực hiện tối đa 8 operation đồng thời.

![Microarchitecture ARM Cortex A77.](../img/a77.svg)
:label:`fig_cortexa77`

Điều này có nghĩa là các chương trình hiệu quả có thể thực hiện nhiều hơn một instruction trong mỗi clock cycle, với điều kiện chúng có thể được thực hiện độc lập. Không phải tất cả unit đều giống nhau. Một số unit chuyên xử lý instruction số nguyên, trong khi các unit khác được tối ưu cho hiệu năng số thực. Để tăng throughput, processor cũng có thể đồng thời đi theo nhiều code path trong một instruction rẽ nhánh, sau đó loại bỏ kết quả của các nhánh không được chọn. Đây là lý do các unit dự đoán nhánh rất quan trọng (ở front-end), để chỉ theo đuổi những path có triển vọng nhất.

### Vectorization

Deep learning đòi hỏi năng lực tính toán cực lớn. Vì vậy, để CPU phù hợp với machine learning, cần thực hiện nhiều operation trong một clock cycle. Điều này đạt được nhờ các vector unit. Chúng có những tên gọi khác nhau: trên ARM, chúng được gọi là NEON; trên x86, chúng (ở thế hệ gần đây) được gọi là [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) unit. Một điểm chung là chúng có khả năng thực hiện các operation SIMD (single instruction multiple data). :numref:`fig_neon128` cho thấy cách cộng 8 số nguyên ngắn trong một clock cycle trên ARM.

![Vectorization NEON 128 bit.](../img/neon128.svg)
:label:`fig_neon128`

Tùy thuộc vào các lựa chọn về architecture, những register như vậy có thể dài tới 512 bit, cho phép kết hợp tối đa 64 cặp số. Chẳng hạn, ta có thể nhân hai số rồi cộng chúng với một số thứ ba; thao tác này còn được gọi là fused multiply-add. [OpenVino](https://01.org/openvinotoolkit) của Intel sử dụng chúng để đạt throughput đáng kể cho deep learning trên CPU cấp server. Tuy nhiên, cần lưu ý rằng con số này hoàn toàn bị lấn át bởi khả năng của GPU. Chẳng hạn, RTX 2080 Ti của NVIDIA có 4.352 CUDA core, mỗi core đều có khả năng xử lý operation như vậy tại bất kỳ thời điểm nào.

### Cache

Hãy xem xét tình huống sau: ta có một CPU khiêm tốn với 4 lõi như được minh họa ở :numref:`fig_skylake` phía trên, chạy ở tần số 2 GHz. Hơn nữa, giả sử ta có chỉ số IPC (instruction per clock) bằng 1 và các unit đã bật AVX2 với độ rộng 256 bit. Ngoài ra, giả sử ít nhất một trong các register được sử dụng cho operation AVX2 cần được lấy từ memory. Điều này có nghĩa là CPU tiêu thụ $4 \times 256 \textrm{ bit} = 128 \textrm{ bytes}$ dữ liệu trong mỗi clock cycle. Nếu không thể truyền $2 \times 10^9 \times 128 = 256 \times 10^9$ byte mỗi giây đến processor, các phần tử xử lý sẽ bị thiếu dữ liệu. Đáng tiếc là memory interface của một chip như vậy chỉ hỗ trợ truyền dữ liệu ở mức 20--40 GB/s, tức là thấp hơn một bậc độ lớn. Cách khắc phục là tránh tải *dữ liệu mới* từ memory nhiều nhất có thể, thay vào đó cache dữ liệu cục bộ trên CPU. Đây là lúc cache phát huy tác dụng. Các tên gọi hoặc khái niệm sau thường được sử dụng:

* **Register** nói chính xác thì không phải là một phần của cache. Chúng giúp chuẩn bị instruction. Tuy nhiên, register của CPU là các vị trí memory mà CPU có thể truy cập ở tốc độ clock mà không chịu thêm độ trễ. CPU có hàng chục register. Việc sử dụng register hiệu quả là trách nhiệm của compiler (hoặc programmer). Chẳng hạn, ngôn ngữ lập trình C có từ khóa `register`.
* **L1 cache** là tuyến phòng thủ đầu tiên trước yêu cầu bandwidth memory cao. L1 cache có kích thước rất nhỏ (thường là 32--64 KB) và thường được tách thành data cache và instruction cache. Khi dữ liệu được tìm thấy trong L1 cache, việc truy cập rất nhanh. Nếu không tìm thấy, quá trình tìm kiếm sẽ tiếp tục xuống các cấp cache thấp hơn.
* **L2 cache** là điểm dừng tiếp theo. Tùy thuộc vào thiết kế architecture và kích thước processor, chúng có thể là exclusive. Chúng có thể chỉ được một lõi nhất định truy cập hoặc được nhiều lõi dùng chung. L2 cache lớn hơn (thường là 256--512 KB trên mỗi lõi) và chậm hơn L1. Hơn nữa, để truy cập một thứ gì đó trong L2, trước tiên chúng ta phải kiểm tra để xác nhận dữ liệu không nằm trong L1, việc này làm tăng thêm một lượng nhỏ độ trễ.
* **L3 cache** được nhiều lõi dùng chung và có thể khá lớn. CPU server Epyc 3 của AMD có tới 256 MB cache được phân bổ trên nhiều chiplet. Các con số phổ biến hơn nằm trong khoảng 4--8 MB.

Dự đoán những phần tử memory nào sẽ cần tiếp theo là một trong các tham số tối ưu hóa then chốt trong thiết kế chip. Chẳng hạn, nên duyệt memory theo hướng *tiến* vì hầu hết thuật toán caching sẽ cố gắng *read ahead* thay vì đọc ngược. Tương tự, duy trì các pattern truy cập memory mang tính cục bộ là một cách tốt để cải thiện hiệu năng.

Việc bổ sung cache là con dao hai lưỡi. Một mặt, cache đảm bảo các lõi xử lý không bị thiếu dữ liệu. Đồng thời, chúng làm tăng kích thước chip, chiếm dụng phần diện tích vốn có thể được dùng để tăng năng lực xử lý. Hơn nữa, *cache miss* có thể gây tốn kém. Hãy xem xét kịch bản xấu nhất, *false sharing*, như minh họa trong :numref:`fig_falsesharing`. Một vị trí bộ nhớ được cache trên processor 0 khi một thread trên processor 1 yêu cầu dữ liệu đó. Để lấy dữ liệu, processor 0 phải dừng công việc đang làm, ghi thông tin trở lại main memory, rồi để processor 1 đọc thông tin đó từ bộ nhớ. Trong quá trình này, cả hai processor đều phải chờ. Rất có thể, đoạn code như vậy chạy *chậm hơn* trên nhiều processor so với một triển khai hiệu quả trên một processor. Đây là thêm một lý do khiến kích thước cache trên thực tế có giới hạn (ngoài giới hạn về kích thước vật lý).

![False sharing (hình ảnh do Intel cung cấp).](../img/falsesharing.svg)
:label:`fig_falsesharing`

## GPU và các accelerator khác

Không quá lời khi nói rằng deep learning sẽ không thể thành công nếu không có GPU. Tương tự, hoàn toàn hợp lý khi cho rằng vận may của các nhà sản xuất GPU đã tăng lên đáng kể nhờ deep learning. Sự đồng tiến hóa của phần cứng và thuật toán này đã dẫn đến tình trạng, dù tốt hay xấu, deep learning trở thành paradigm mô hình hóa thống kê được ưu tiên. Vì vậy, việc hiểu những lợi ích cụ thể của GPU và các accelerator liên quan như TPU :cite:`Jouppi.Young.Patil.ea.2017` là rất đáng giá.

Một điểm đáng chú ý là sự phân biệt thường được sử dụng trong thực tế: accelerator được tối ưu hóa cho training hoặc inference. Với inference, chúng ta chỉ cần tính forward propagation trong một network. Không cần lưu trữ dữ liệu trung gian cho backpropagation. Hơn nữa, chúng ta có thể không cần phép tính quá chính xác (FP16 hoặc INT8 thường là đủ). Mặt khác, trong quá trình training, tất cả kết quả trung gian cần được lưu trữ để tính gradient. Ngoài ra, việc tích lũy gradient đòi hỏi độ chính xác cao hơn để tránh underflow (hoặc overflow) số học. Điều này có nghĩa là FP16 (hoặc mixed precision với FP32) là yêu cầu tối thiểu. Tất cả những điều này đòi hỏi bộ nhớ nhanh hơn và lớn hơn (HBM2 so với GDDR6), cùng năng lực xử lý cao hơn. Chẳng hạn, GPU NVIDIA [Turing](https://devblogs.nvidia.com/nvidia-turing-architecture-in-depth/) T4 được tối ưu hóa cho inference, trong khi GPU V100 phù hợp hơn cho training.

Hãy nhớ lại vectorization như minh họa trong :numref:`fig_neon128`. Việc bổ sung các vector unit vào một lõi processor cho phép chúng ta tăng đáng kể throughput. Ví dụ, trong ví dụ ở :numref:`fig_neon128`, chúng ta có thể thực hiện đồng thời 16 phép toán.
Thứ nhất,
điều gì sẽ xảy ra nếu chúng ta bổ sung các phép toán được tối ưu hóa không chỉ cho các phép toán giữa các vector mà còn cho các phép toán giữa các matrix?
Chiến lược này đã dẫn đến tensor core (sẽ được trình bày ngay sau đây).
Thứ hai, điều gì sẽ xảy ra nếu chúng ta bổ sung nhiều lõi hơn nữa? Tóm lại, hai chiến lược này khái quát các quyết định thiết kế trong GPU. :numref:`fig_turing_processing_block` trình bày tổng quan về một processing block cơ bản. Nó chứa 16 integer unit và 16 floating-point unit. Ngoài ra, hai tensor core tăng tốc một tập con hẹp gồm các phép toán bổ sung liên quan đến deep learning. Mỗi streaming multiprocessor gồm bốn block như vậy.

![NVIDIA Turing processing block (hình ảnh do NVIDIA cung cấp).](../img/turing-processing-block.png)
:width:`150px`
:label:`fig_turing_processing_block`

Tiếp theo, 12 streaming multiprocessor được nhóm thành các graphics processing cluster, tạo nên các processor TU102 cao cấp. Các memory channel dồi dào và một L2 cache bổ sung cho cấu hình này. :numref:`fig_turing` trình bày các chi tiết liên quan. Một trong những lý do để thiết kế một thiết bị như vậy là các block riêng lẻ có thể được thêm vào hoặc loại bỏ khi cần, nhằm cho phép tạo ra các chip nhỏ gọn hơn và xử lý các vấn đề về yield (các module lỗi có thể không được kích hoạt). May mắn là việc lập trình các thiết bị như vậy được che giấu rất kỹ đối với các nhà nghiên cứu deep learning thông thường, bên dưới nhiều lớp code CUDA và framework. Cụ thể, nhiều chương trình hoàn toàn có thể được thực thi đồng thời trên GPU, miễn là có đủ tài nguyên. Tuy nhiên, việc nắm rõ các giới hạn của thiết bị vẫn rất hữu ích để tránh chọn những model không vừa với device memory.

![NVIDIA Turing architecture (hình ảnh do NVIDIA cung cấp)](../img/turing.png)
:width:`350px`
:label:`fig_turing`

Một khía cạnh cuối cùng đáng được đề cập chi tiết hơn là *tensor core*. Đây là ví dụ về xu hướng gần đây trong việc bổ sung các mạch được tối ưu hóa hơn, đặc biệt hiệu quả cho deep learning. Chẳng hạn, TPU đã bổ sung một systolic array :cite:`Kung.1988` để nhân matrix nhanh. Trong trường hợp đó, thiết kế nhằm hỗ trợ một số lượng rất nhỏ các phép toán lớn (một phép toán đối với thế hệ TPU đầu tiên). Tensor core nằm ở đầu đối lập. Chúng được tối ưu hóa cho các phép toán nhỏ liên quan đến các matrix có kích thước từ $4 \times 4$ đến $16 \times 16$, tùy thuộc vào độ chính xác số học. :numref:`fig_tensorcore` trình bày tổng quan về các tối ưu hóa này.

![NVIDIA tensor cores in Turing (hình ảnh do NVIDIA cung cấp).](../img/tensorcore.jpg)
:width:`400px`
:label:`fig_tensorcore`

Rõ ràng, khi tối ưu hóa cho việc tính toán, chúng ta phải chấp nhận một số thỏa hiệp. Một trong số đó là GPU không xử lý tốt interrupt và dữ liệu sparse. Dù có những ngoại lệ đáng chú ý, chẳng hạn như [Gunrock](https://github.com/gunrock/gunrock) :cite:`Wang.Davidson.Pan.ea.2016`, access pattern của sparse matrix và vector không phù hợp với các thao tác đọc burst có bandwidth cao, vốn là điểm mạnh của GPU. Việc đáp ứng đồng thời cả hai mục tiêu là một lĩnh vực đang được tích cực nghiên cứu. Xem למשל [DGL](http://dgl.ai), một library được tinh chỉnh cho deep learning trên graph.


## Network và Bus

Bất cứ khi nào một thiết bị đơn lẻ không đủ cho việc tối ưu hóa, chúng ta cần truyền dữ liệu đến và đi từ thiết bị đó để đồng bộ hóa quá trình xử lý. Đây là lúc network và bus phát huy tác dụng. Chúng ta có một số tham số thiết kế: bandwidth, cost, distance và flexibility.
Ở một đầu, chúng ta có WiFi với phạm vi khá tốt, rất dễ sử dụng (suy cho cùng thì không cần dây), có chi phí thấp nhưng cung cấp bandwidth và latency tương đối tầm thường. Không một nhà nghiên cứu machine learning nào tỉnh táo lại sử dụng WiFi để xây dựng một cluster server. Phần tiếp theo tập trung vào các interconnect phù hợp cho deep learning.

* **PCIe** là một bus chuyên dụng cho các kết nối point-to-point có bandwidth rất cao (lên đến 32 GB/s trên PCIe 4.0 trong một khe 16-lane) trên mỗi lane. Latency vào khoảng vài microsecond (5 μs). Các liên kết PCIe rất quý giá. Processor chỉ có số lượng liên kết hữu hạn: AMD's EPYC 3 có 128 lane, Intel's Xeon có tối đa 48 lane mỗi chip; trên các CPU cấp desktop, các con số lần lượt là 20 (Ryzen 9) và 16 (Core i9). Vì GPU thường có 16 lane, điều này giới hạn số GPU có thể kết nối với CPU ở full bandwidth. Dù sao thì chúng cũng phải chia sẻ các liên kết với những thiết bị ngoại vi có bandwidth cao khác như storage và Ethernet. Cũng giống như khi truy cập RAM, nên ưu tiên các lần truyền dữ liệu bulk lớn do giảm được packet overhead.
* **Ethernet** là cách phổ biến nhất để kết nối các máy tính. Dù chậm hơn đáng kể so với PCIe, Ethernet rất rẻ, bền bỉ khi triển khai và hỗ trợ khoảng cách lớn hơn nhiều. Bandwidth điển hình cho các server cấp thấp là 1 GBit/s. Các thiết bị cao cấp hơn (ví dụ: [C5 instances](https://aws.amazon.com/ec2/instance-types/c5/) trên cloud) cung cấp bandwidth từ 10 đến 100 GBit/s. Như trong tất cả các trường hợp trước, việc truyền dữ liệu có overhead đáng kể. Lưu ý rằng chúng ta hầu như không bao giờ sử dụng Ethernet thô trực tiếp, mà thường sử dụng một protocol chạy bên trên interconnect vật lý (chẳng hạn UDP hoặc TCP/IP). Điều này tạo thêm overhead. Giống như PCIe, Ethernet được thiết kế để kết nối hai thiết bị, ví dụ một máy tính và một switch.
* **Switch** cho phép chúng ta kết nối nhiều thiết bị theo cách mà mọi cặp thiết bị đều có thể đồng thời thực hiện kết nối point-to-point (thường ở full bandwidth). Chẳng hạn, switch Ethernet có thể kết nối 40 server với bandwidth cross-sectional cao. Lưu ý rằng switch không chỉ có trong các network máy tính truyền thống. Ngay cả lane PCIe cũng có thể được [switched](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). Điều này xảy ra, chẳng hạn, khi kết nối số lượng lớn GPU với host processor, như trong trường hợp [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/).
* **NVLink** là một giải pháp thay thế cho PCIe khi cần các interconnect có bandwidth rất cao. Nó cung cấp tốc độ truyền dữ liệu lên đến 300 Gbit/s trên mỗi link. GPU server (Volta V100) có sáu link, trong khi GPU cấp consumer (RTX 2080 Ti) chỉ có một link, hoạt động ở tốc độ giảm còn 100 Gbit/s. Chúng tôi khuyến nghị sử dụng [NCCL](https://github.com/NVIDIA/nccl) để đạt tốc độ truyền dữ liệu cao giữa các GPU.



## Thêm các số liệu về Latency

Bản tóm tắt trong :numref:`table_latency_numbers` và :numref:`table_latency_numbers_tesla` được lấy từ [Eliot Eshelman](https://gist.github.com/eshelman), người duy trì phiên bản cập nhật của các số liệu này dưới dạng một [GitHub gist](https://gist.github.com/eshelman/343a1c46cb3fba142c1afdcdeec17646).

:Common Latency Numbers.

| Thao tác | Thời gian | Ghi chú |
| :----------------------------------------- | -----: | :---------------------------------------------- |
| Tham chiếu/trúng cache L1                  | 1.5 ns | 4 chu kỳ                                        |
| Phép cộng/nhân dấu phẩy động/FMA           | 1.5 ns | 4 chu kỳ                                        |
| Tham chiếu/trúng cache L2                  |   5 ns | 12 ~ 17 chu kỳ                                  |
| Dự đoán nhánh sai                          |   6 ns | 15 ~ 20 chu kỳ                                  |
| Trúng cache L3 (cache không được chia sẻ)  |  16 ns | 42 chu kỳ                                       |
| Trúng cache L3 (được chia sẻ trên một core khác) |  25 ns | 65 chu kỳ                                       |
| Khóa/mở khóa mutex                         |  25 ns |                                                 |
| Trúng cache L3 (đã sửa đổi trên một core khác) |  29 ns | 75 chu kỳ                                       |
| Trúng cache L3 (trên một socket CPU từ xa) |  40 ns | 100 ~ 300 chu kỳ (40 ~ 116 ns)                  |
| Một bước nhảy QPI đến CPU khác (mỗi bước)  |  40 ns |                                                 |
| Tham chiếu bộ nhớ 64MB (CPU cục bộ)        |  46 ns | TinyMemBench trên Broadwell E5-2690v4           |
| Tham chiếu bộ nhớ 64MB (CPU từ xa)         |  70 ns | TinyMemBench trên Broadwell E5-2690v4           |
| Tham chiếu bộ nhớ 256MB (CPU cục bộ)       |  75 ns | TinyMemBench trên Broadwell E5-2690v4           |
| Ghi ngẫu nhiên Intel Optane                |  94 ns | UCSD Non-Volatile Systems Lab                   |
| Tham chiếu bộ nhớ 256MB (CPU từ xa)        | 120 ns | TinyMemBench trên Broadwell E5-2690v4           |
| Đọc ngẫu nhiên Intel Optane                | 305 ns | UCSD Non-Volatile Systems Lab                   |
| Gửi 4KB qua fabric HPC 100 Gbps            |   1 μs | MVAPICH2 qua Intel Omni-Path                   |
| Nén 1KB bằng Google Snappy                 |   3 μs |                                                 |
| Gửi 4KB qua Ethernet 10 Gbps               |  10 μs |                                                 |
| Ghi ngẫu nhiên 4KB vào SSD NVMe            |  30 μs | SSD NVMe DC P3608 (QOS 99% là 500μs)            |
| Truyền 1MB đến/từ GPU NVLink               |  30 μs | ~33GB/s trên NVIDIA 40GB NVLink                 |
| Truyền 1MB đến/từ GPU PCI-E                |  80 μs | ~12GB/s trên liên kết PCIe 3.0 x16             |
| Đọc ngẫu nhiên 4KB từ SSD NVMe             | 120 μs | SSD NVMe DC P3608 (QOS 99%)                     |
| Đọc tuần tự 1MB từ SSD NVMe                | 208 μs | ~4.8GB/s SSD NVMe DC P3608                    |
| Ghi ngẫu nhiên 4KB vào SSD SATA            | 500 μs | SSD SATA DC S3510 (QOS 99.9%)                   |
| Đọc ngẫu nhiên 4KB từ SSD SATA             | 500 μs | SSD SATA DC S3510 (QOS 99.9%)                   |
| Round trip trong cùng data center          | 500 μs | Ping một chiều khoảng ~250μs                    |
| Đọc tuần tự 1MB từ SSD SATA                |   2 ms | ~550MB/s SSD SATA DC S3510                    |
| Đọc tuần tự 1MB từ disk                    |   5 ms | HDD máy chủ ~200MB/s                           |
| Truy cập disk ngẫu nhiên (seek+rotation)   |  10 ms |                                                 |
| Gửi packet CA->Netherlands->CA             | 150 ms |                                                 |
:label:`table_latency_numbers`

:Latency Numbers for NVIDIA Tesla GPUs.

| Thao tác | Thời gian | Ghi chú |
| :------------------------------ | -----: | :---------------------------------------- |
| Truy cập Shared Memory của GPU |  30 ns | 30~90 chu kỳ (xung đột bank làm tăng latency) |
| Truy cập Global Memory của GPU | 200 ns | 200~800 chu kỳ                            |
| Khởi chạy CUDA kernel trên GPU |  10 μs | CPU máy chủ yêu cầu GPU khởi động kernel    |
| Truyền 1MB đến/từ GPU NVLink |  30 μs | ~33GB/s trên NVIDIA 40GB NVLink           |
| Truyền 1MB đến/từ GPU PCI-E  |  80 μs | ~12GB/s trên liên kết PCI-Express x16         |
:label:`table_latency_numbers_tesla`

## Tóm tắt

* Các thiết bị có overhead cho các thao tác. Vì vậy, điều quan trọng là hướng đến một số lượng nhỏ các lần truyền lớn thay vì nhiều lần truyền nhỏ. Điều này áp dụng cho RAM, SSD, network và GPU.
* Vectorization là yếu tố then chốt đối với hiệu năng. Hãy đảm bảo bạn nắm được các khả năng cụ thể của accelerator. Ví dụ, một số CPU Intel Xeon đặc biệt tốt với các phép toán INT8, GPU NVIDIA Volta nổi trội trong các phép toán matrix-matrix FP16, còn NVIDIA Turing phát huy thế mạnh với các phép toán FP16, INT8 và INT4.
* Tràn số do các kiểu dữ liệu nhỏ có thể gây vấn đề trong quá trình training (và ở mức độ thấp hơn trong inference).
* Aliasing có thể làm suy giảm đáng kể hiệu năng. Chẳng hạn, trên các CPU 64 bit, việc căn chỉnh bộ nhớ nên dựa trên các boundary 64 bit. Trên GPU, nên giữ cho kích thước convolution được căn chỉnh, chẳng hạn theo tensor cores.
* Hãy điều chỉnh các algorithm cho phù hợp với hardware (ví dụ: memory footprint và bandwidth). Có thể đạt được speedup rất lớn (tăng nhiều bậc độ lớn) khi đưa các parameter vào trong cache.
* Chúng tôi khuyến nghị bạn phác thảo hiệu năng của một algorithm mới trên giấy trước khi kiểm chứng các kết quả thực nghiệm. Những sai lệch từ một bậc độ lớn trở lên là lý do cần quan tâm.
* Dùng profiler để debug các bottleneck về hiệu năng.
* Hardware dùng cho training và inference có các điểm tối ưu khác nhau xét về giá cả và hiệu năng.

## Bài tập

1. Viết code C để kiểm tra xem tốc độ truy cập bộ nhớ có khác nhau hay không khi bộ nhớ được căn chỉnh hoặc không được căn chỉnh so với memory interface bên ngoài. Gợi ý: hãy cẩn thận với các ảnh hưởng của caching.
1. Kiểm tra sự khác biệt về tốc độ giữa việc truy cập bộ nhớ theo thứ tự hoặc với một stride cho trước.
1. Làm thế nào bạn có thể đo kích thước cache trên CPU?
1. Bạn sẽ bố trí dữ liệu trên nhiều memory channel như thế nào để đạt bandwidth tối đa? Bạn sẽ bố trí như thế nào nếu có nhiều thread nhỏ?
1. Một HDD cấp enterprise quay ở tốc độ 10.000 vòng/phút. Trong trường hợp xấu nhất, thời gian tuyệt đối tối thiểu mà HDD cần trước khi có thể đọc dữ liệu là bao lâu (giả sử đầu đọc di chuyển gần như tức thời)? Vì sao HDD 2.5" đang trở nên phổ biến đối với server thương mại (so với các ổ 3.5" và 5.25")?
1. Giả sử một nhà sản xuất HDD tăng mật độ lưu trữ từ 1 Tbit trên mỗi inch vuông lên 5 Tbit trên mỗi inch vuông. Bạn có thể lưu trữ bao nhiêu thông tin trên một vòng của HDD 2.5"? Có sự khác biệt nào giữa các track bên trong và bên ngoài không?
1. Chuyển từ kiểu dữ liệu 8 bit sang 16 bit làm tăng lượng silicon lên khoảng bốn lần. Vì sao? Vì sao NVIDIA có thể đã thêm các phép toán INT4 vào GPU Turing của họ?
1. Đọc tiến qua bộ nhớ nhanh hơn đọc lùi bao nhiêu? Con số này có khác nhau giữa các máy tính và nhà cung cấp CPU khác nhau không? Vì sao? Hãy viết code C và thử nghiệm.
1. Bạn có thể đo kích thước cache của disk không? Kích thước này là bao nhiêu đối với một HDD điển hình? SSD có cần cache không?
1. Đo overhead của packet khi gửi message qua Ethernet. Tìm hiểu sự khác biệt giữa các kết nối UDP và TCP/IP.
1. Direct memory access cho phép các thiết bị khác ngoài CPU ghi (và đọc) trực tiếp vào (từ) bộ nhớ. Vì sao đây là một ý tưởng tốt?
1. Xem các con số hiệu năng của GPU Turing T4. Vì sao hiệu năng chỉ tăng gấp đôi khi chuyển từ FP16 sang INT8 và INT4?
1. Thời gian ngắn nhất mà một packet cần để round trip giữa San Francisco và Amsterdam là bao lâu? Gợi ý: bạn có thể giả sử khoảng cách là 10.000 km.

[Thảo luận](https://discuss.d2l.ai/t/363)