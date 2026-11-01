# Sử dụng Jupyter Notebook
:label:`sec_jupyter`


Phần này mô tả cách chỉnh sửa và chạy mã
trong từng phần của cuốn sách này
bằng Jupyter Notebook. Hãy đảm bảo bạn đã
cài đặt Jupyter và tải xuống
mã nguồn như được mô tả trong
:ref:`chap_installation`.
Nếu muốn tìm hiểu thêm về Jupyter, hãy xem hướng dẫn tuyệt vời trong
[tài liệu](https://jupyter.readthedocs.io/en/latest/) của họ.


## Chỉnh sửa và chạy mã cục bộ

Giả sử đường dẫn cục bộ của mã nguồn cuốn sách là `xx/yy/d2l-en/`. Sử dụng shell để chuyển thư mục đến đường dẫn này (`cd xx/yy/d2l-en`) và chạy lệnh `jupyter notebook`. Nếu trình duyệt của bạn không tự động thực hiện việc này, hãy mở http://localhost:8888 và bạn sẽ thấy giao diện của Jupyter cùng tất cả các thư mục chứa mã nguồn của cuốn sách, như minh họa trong :numref:`fig_jupyter00`.

![Các thư mục chứa mã nguồn của cuốn sách này.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


Bạn có thể truy cập các tệp notebook bằng cách nhấp vào thư mục hiển thị trên trang web.
Chúng thường có hậu tố ".ipynb".
Để ngắn gọn, chúng ta tạo một tệp "test.ipynb" tạm thời.
Nội dung hiển thị sau khi bạn nhấp vào tệp này
được minh họa trong :numref:`fig_jupyter01`.
Notebook này bao gồm một ô markdown và một ô mã. Nội dung trong ô markdown bao gồm "This Is a Title" và "This is text.".
Ô mã chứa hai dòng mã Python.

![Các ô markdown và mã trong tệp "text.ipynb".](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


Nhấp đúp vào ô markdown để chuyển sang chế độ chỉnh sửa.
Thêm một chuỗi văn bản mới "Hello world." vào cuối ô, như minh họa trong :numref:`fig_jupyter02`.

![Chỉnh sửa ô markdown.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


Như minh họa trong :numref:`fig_jupyter03`,
nhấp vào "Cell" $\rightarrow$ "Run Cells" trên thanh menu để chạy ô đã chỉnh sửa.

![Chạy ô.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

Sau khi chạy, ô markdown được hiển thị trong :numref:`fig_jupyter04`.

![Ô markdown sau khi chạy.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


Tiếp theo, nhấp vào ô mã. Nhân các phần tử với 2 sau dòng mã cuối cùng, như minh họa trong :numref:`fig_jupyter05`.

![Chỉnh sửa ô mã.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


Bạn cũng có thể chạy ô bằng phím tắt (mặc định là "Ctrl + Enter") và nhận được kết quả đầu ra như trong :numref:`fig_jupyter06`.

![Chạy ô mã để nhận kết quả đầu ra.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


Khi notebook chứa nhiều ô hơn, chúng ta có thể nhấp vào "Kernel" $\rightarrow$ "Restart & Run All" trên thanh menu để chạy tất cả các ô trong toàn bộ notebook. Bằng cách nhấp vào "Help" $\rightarrow$ "Edit Keyboard Shortcuts" trên thanh menu, bạn có thể chỉnh sửa các phím tắt theo sở thích của mình.

## Tùy chọn nâng cao

Ngoài việc chỉnh sửa cục bộ, có hai vấn đề khá quan trọng: chỉnh sửa notebook ở định dạng markdown và chạy Jupyter từ xa.
Vấn đề sau hữu ích khi chúng ta muốn chạy mã trên một máy chủ nhanh hơn.
Vấn đề trước quan trọng vì định dạng ipynb gốc của Jupyter lưu trữ rất nhiều dữ liệu phụ trợ không liên quan đến nội dung,
chủ yếu liên quan đến cách thức và nơi mã được chạy.
Điều này gây khó khăn cho Git, khiến
việc xem xét các đóng góp trở nên rất khó khăn.
May mắn là có một lựa chọn khác---chỉnh sửa trực tiếp ở định dạng markdown.

### Các tệp Markdown trong Jupyter

Nếu muốn đóng góp cho nội dung của cuốn sách này, bạn cần sửa đổi
tệp nguồn (tệp md, không phải tệp ipynb) trên GitHub.
Bằng plugin notedown, chúng ta
có thể trực tiếp chỉnh sửa notebook ở định dạng md trong Jupyter.


Trước tiên, cài đặt plugin notedown, chạy Jupyter Notebook và nạp plugin:

```
pip install d2l-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


Bạn cũng có thể bật plugin notedown theo mặc định mỗi khi chạy Jupyter Notebook.
Trước tiên, tạo một tệp cấu hình Jupyter Notebook (nếu tệp này đã được tạo, bạn có thể bỏ qua bước này).

```
jupyter notebook --generate-config
```


Sau đó, thêm dòng sau vào cuối tệp cấu hình Jupyter Notebook (đối với Linux hoặc macOS, thường nằm tại đường dẫn `~/.jupyter/jupyter_notebook_config.py`):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


Sau đó, bạn chỉ cần chạy lệnh `jupyter notebook` để bật plugin notedown theo mặc định.

### Chạy Jupyter Notebook trên máy chủ từ xa

Đôi khi, bạn có thể muốn chạy Jupyter notebook trên một máy chủ từ xa và truy cập máy chủ đó thông qua trình duyệt trên máy tính cục bộ. Nếu máy cục bộ của bạn đã cài Linux hoặc macOS (Windows cũng hỗ trợ chức năng này thông qua phần mềm bên thứ ba như PuTTY), bạn có thể sử dụng chuyển tiếp cổng:

```
ssh myserver -L 8888:localhost:8888
```


Chuỗi `myserver` ở trên là địa chỉ của máy chủ từ xa.
Sau đó, chúng ta có thể sử dụng http://localhost:8888 để truy cập máy chủ từ xa `myserver` đang chạy Jupyter notebook. Chúng tôi sẽ trình bày chi tiết cách chạy Jupyter notebook trên các instance AWS
ở phần sau của phụ lục này.

### Đo thời gian

Chúng ta có thể sử dụng plugin `ExecuteTime` để đo thời gian thực thi của từng ô mã trong Jupyter notebook.
Sử dụng các lệnh sau để cài đặt plugin:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## Tóm tắt

* Bằng công cụ Jupyter Notebook, chúng ta có thể chỉnh sửa, chạy và đóng góp cho từng phần của cuốn sách.
* Chúng ta có thể chạy Jupyter notebook trên các máy chủ từ xa bằng chuyển tiếp cổng.


## Bài tập

1. Chỉnh sửa và chạy mã trong cuốn sách này bằng Jupyter Notebook trên máy cục bộ của bạn.
1. Chỉnh sửa và chạy mã trong cuốn sách này bằng Jupyter Notebook *từ xa* thông qua chuyển tiếp cổng.
1. So sánh thời gian chạy của các phép toán $\mathbf{A}^\top \mathbf{B}$ và $\mathbf{A} \mathbf{B}$ đối với hai ma trận vuông trong $\mathbb{R}^{1024 \times 1024}$. Phép toán nào nhanh hơn?


[Thảo luận](https://discuss.d2l.ai/t/421)