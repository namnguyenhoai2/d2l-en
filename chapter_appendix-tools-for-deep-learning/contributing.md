# Đóng góp cho cuốn sách
:label:`sec_how_to_contribute`

Những đóng góp từ [độc giả](https://github.com/d2l-ai/d2l-en/graphs/contributors) giúp chúng tôi cải thiện cuốn sách này. Nếu bạn phát hiện lỗi chính tả, liên kết lỗi thời, nội dung mà bạn cho rằng chúng tôi đã bỏ sót trích dẫn, đoạn code chưa được viết một cách thanh lịch, hoặc phần giải thích chưa rõ ràng, hãy đóng góp và giúp chúng tôi hỗ trợ độc giả tốt hơn. Trong khi với những cuốn sách thông thường, khoảng thời gian giữa các lần in (và do đó giữa các lần sửa lỗi chính tả) có thể tính bằng năm, thì thông thường chỉ mất vài giờ đến vài ngày để đưa một cải tiến vào cuốn sách này. Tất cả điều đó đều có thể thực hiện được nhờ version control và kiểm thử continuous integration (CI). Để thực hiện việc này, bạn cần gửi một [pull request](https://github.com/d2l-ai/d2l-en/pulls) đến repository GitHub. Khi pull request của bạn được các tác giả merge vào code repository, bạn sẽ trở thành một contributor.

## Gửi các thay đổi nhỏ

Những đóng góp phổ biến nhất là chỉnh sửa một câu hoặc sửa lỗi chính tả. Chúng tôi khuyên bạn tìm file nguồn trong [repository GitHub](https://github.com/d2l-ai/d2l-en) và chỉnh sửa file trực tiếp. Ví dụ: bạn có thể tìm kiếm file thông qua nút [Find file](https://github.com/d2l-ai/d2l-en/find/master) (:numref:`fig_edit_file`) để xác định file nguồn (một file markdown). Sau đó, bạn nhấp vào nút "Edit this file" ở góc trên bên phải để thực hiện thay đổi trong file markdown.

![Chỉnh sửa file trên Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

Sau khi hoàn tất, hãy điền mô tả thay đổi vào bảng "Propose file change" ở cuối trang, rồi nhấp vào nút "Propose file change". Bạn sẽ được chuyển hướng đến một trang mới để xem lại các thay đổi của mình (:numref:`fig_git_createpr`). Nếu mọi thứ đều ổn, bạn có thể gửi pull request bằng cách nhấp vào nút "Create pull request".

## Đề xuất các thay đổi lớn

Nếu bạn dự định cập nhật một phần lớn nội dung hoặc code, bạn cần biết thêm một chút về định dạng mà cuốn sách này sử dụng. File nguồn dựa trên định dạng [markdown](https://daringfireball.net/projects/markdown/syntax) cùng một tập hợp extension thông qua package [D2L-Book](http://book.d2l.ai/user/markdown.html), chẳng hạn như tham chiếu đến equation, image, chapter và citation. Bạn có thể dùng bất kỳ markdown editor nào để mở các file này và thực hiện thay đổi.

Nếu muốn thay đổi code, chúng tôi khuyên bạn sử dụng Jupyter Notebook để mở các file markdown như mô tả trong :numref:`sec_jupyter`, để có thể chạy và kiểm thử các thay đổi. Hãy nhớ xóa toàn bộ output trước khi gửi thay đổi, vì hệ thống CI của chúng tôi sẽ thực thi các section mà bạn đã cập nhật để tạo output.

Một số section có thể hỗ trợ nhiều triển khai framework.
Nếu thêm một code block mới, hãy sử dụng `%%tab` để đánh dấu block này ở dòng đầu tiên. Ví dụ, `%%tab pytorch` cho code block PyTorch, `%%tab tensorflow` cho code block TensorFlow, hoặc `%%tab all` cho code block dùng chung cho mọi triển khai. Bạn có thể tham khảo package `d2lbook` để biết thêm thông tin.

## Gửi các thay đổi lớn

Chúng tôi đề xuất bạn sử dụng quy trình Git tiêu chuẩn để gửi một thay đổi lớn. Tóm lại, quy trình này diễn ra như mô tả trong :numref:`fig_contribute`.

![Đóng góp cho cuốn sách.](../img/contribute.svg)
:label:`fig_contribute`

Chúng tôi sẽ hướng dẫn chi tiết từng bước. Nếu đã quen thuộc với Git, bạn có thể bỏ qua section này. Để làm rõ, chúng tôi giả định rằng user name của contributor là "astonzhang".

### Cài đặt Git

Cuốn sách Git mã nguồn mở mô tả [cách cài đặt Git](https://git-scm.com/book/en/v2). Thông thường, cách này hoạt động thông qua `apt install git` trên Ubuntu Linux, bằng cách cài đặt bộ công cụ dành cho developer Xcode trên macOS, hoặc bằng cách sử dụng [desktop client](https://desktop.github.com) của GitHub. Nếu chưa có tài khoản GitHub, bạn cần đăng ký một tài khoản.

### Đăng nhập vào GitHub

Nhập [địa chỉ](https://github.com/d2l-ai/d2l-en/) của code repository của cuốn sách vào trình duyệt. Nhấp vào nút `Fork` trong khung màu đỏ ở góc trên bên phải của :numref:`fig_git_fork` để tạo một bản sao của repository của cuốn sách này. Đây sẽ là *bản sao của bạn* và bạn có thể thay đổi tùy ý.

![Trang code repository.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


Bây giờ, code repository của cuốn sách này sẽ được fork (tức là sao chép) vào user name của bạn, chẳng hạn như `astonzhang/d2l-en` được hiển thị ở góc trên bên trái của :numref:`fig_git_forked`.

![Code repository đã được fork.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### Clone Repository

Để clone repository (tức là tạo một bản sao cục bộ), chúng ta cần lấy địa chỉ repository của nó. Nút màu xanh lá cây trong :numref:`fig_git_clone` hiển thị địa chỉ này. Nếu quyết định giữ fork này trong thời gian dài hơn, hãy đảm bảo rằng bản sao cục bộ của bạn được cập nhật với repository chính. Hiện tại, chỉ cần làm theo hướng dẫn trong :ref:`chap_installation` để bắt đầu. Điểm khác biệt chính là bây giờ bạn đang tải xuống *fork của chính mình* của repository.

![Clone repository.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```


### Chỉnh sửa và Push

Bây giờ là lúc chỉnh sửa cuốn sách. Tốt nhất là chỉnh sửa trong Jupyter Notebook theo hướng dẫn ở :numref:`sec_jupyter`. Thực hiện các thay đổi và kiểm tra để đảm bảo chúng đúng. Giả sử chúng ta đã sửa một lỗi chính tả trong file `~/d2l-en/chapter_appendix-tools-for-deep-learning/contributing.md`.
Sau đó, bạn có thể kiểm tra những file đã thay đổi.

Lúc này Git sẽ thông báo rằng file `chapter_appendix-tools-for-deep-learning/contributing.md` đã được chỉnh sửa.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix-tools-for-deep-learning/contributing.md
```


Sau khi xác nhận rằng đây là điều bạn muốn, hãy thực thi command sau:

```
git add chapter_appendix-tools-for-deep-learning/contributing.md
git commit -m 'Fix a typo in git documentation'
git push
```


Code đã thay đổi sẽ nằm trong fork cá nhân của bạn của repository. Để yêu cầu thêm thay đổi của mình, bạn phải tạo một pull request cho repository chính thức của cuốn sách.

### Gửi Pull Request

Như minh họa trong :numref:`fig_git_newpr`, hãy truy cập fork của repository trên GitHub và chọn "New pull request". Thao tác này sẽ mở một màn hình hiển thị những thay đổi giữa các chỉnh sửa của bạn và phiên bản hiện tại trong repository chính của cuốn sách.

![Pull request mới.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`


Cuối cùng, gửi một pull request bằng cách nhấp vào nút như minh họa trong :numref:`fig_git_createpr`. Hãy đảm bảo mô tả những thay đổi bạn đã thực hiện trong pull request.
Điều này sẽ giúp các tác giả dễ dàng xem xét và merge thay đổi vào cuốn sách hơn. Tùy thuộc vào các thay đổi, pull request có thể được chấp nhận ngay, bị từ chối, hoặc nhiều khả năng hơn là bạn sẽ nhận được phản hồi về các thay đổi. Sau khi tiếp thu các phản hồi đó, bạn đã hoàn tất.

![Tạo pull request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`


## Tóm tắt

* Bạn có thể sử dụng GitHub để đóng góp cho cuốn sách này.
* Bạn có thể chỉnh sửa trực tiếp file trên GitHub đối với các thay đổi nhỏ.
* Đối với một thay đổi lớn, hãy fork repository, chỉnh sửa cục bộ, và chỉ gửi đóng góp khi bạn đã sẵn sàng.
* Pull request là cách các đóng góp được tập hợp. Cố gắng không gửi pull request quá lớn, vì chúng sẽ khó hiểu và khó tích hợp. Tốt hơn hết là gửi nhiều pull request nhỏ hơn.


## Bài tập

1. Star và fork repository `d2l-ai/d2l-en`.
1. Nếu phát hiện điều gì cần cải thiện (ví dụ: thiếu tài liệu tham khảo), hãy gửi một pull request.
1. Thông thường, tốt hơn là tạo pull request bằng một branch mới. Tìm hiểu cách thực hiện trong [Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell).

[Thảo luận](https://discuss.d2l.ai/t/426)
