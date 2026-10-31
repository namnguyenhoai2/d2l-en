# Ký hiệu
:label:`chap_notation`

Trong suốt cuốn sách này, chúng tôi tuân theo 
các quy ước ký hiệu sau.
Lưu ý rằng một số ký hiệu trong đó là placeholder,
trong khi những ký hiệu khác đề cập đến các đối tượng cụ thể.
Theo quy tắc chung, 
mạo từ không xác định "a" thường cho biết
ký hiệu đó là một placeholder
và các ký hiệu có định dạng tương tự
có thể biểu thị những đối tượng khác cùng kiểu.
Ví dụ, "$x$: một scalar" có nghĩa là 
các chữ cái viết thường thường
biểu diễn các giá trị scalar,
nhưng "$\mathbb{Z}$: tập số nguyên"
cụ thể đề cập đến ký hiệu $\mathbb{Z}$.



## Đối tượng số

* $x$: một scalar
* $\mathbf{x}$: một vector
* $\mathbf{X}$: một ma trận
* $\mathsf{X}$: một tensor tổng quát
* $\mathbf{I}$: ma trận đơn vị (với một số chiều nhất định), tức là một ma trận vuông có $1$ trên mọi phần tử đường chéo và $0$ trên mọi phần tử ngoài đường chéo
* $x_i$, $[\mathbf{x}]_i$: phần tử thứ $i^\textrm{th}$ của vector $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: phần tử của ma trận $\mathbf{X}$ tại hàng $i$ và cột $j$.



## Lý thuyết tập hợp


* $\mathcal{X}$: một tập hợp
* $\mathbb{Z}$: tập số nguyên
* $\mathbb{Z}^+$: tập số nguyên dương
* $\mathbb{R}$: tập số thực
* $\mathbb{R}^n$: tập các vector $n$ chiều của số thực
* $\mathbb{R}^{a\times b}$: tập các ma trận số thực có $a$ hàng và $b$ cột
* $|\mathcal{X}|$: lực lượng (số phần tử) của tập $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: hợp của các tập hợp $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: giao của các tập hợp $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: hiệu của tập hợp $\mathcal{B}$ với tập hợp $\mathcal{A}$ (chỉ chứa những phần tử của $\mathcal{A}$ không thuộc $\mathcal{B}$)



## Hàm và toán tử


* $f(\cdot)$: một hàm
* $\log(\cdot)$: logarithm tự nhiên (cơ số $e$)
* $\log_2(\cdot)$: logarithm cơ số $2$
* $\exp(\cdot)$: hàm mũ
* $\mathbf{1}(\cdot)$: hàm chỉ báo; có giá trị bằng $1$ nếu đối số boolean là đúng và bằng $0$ trong các trường hợp khác
* $\mathbf{1}_{\mathcal{X}}(z)$: hàm chỉ báo phần tử thuộc tập hợp; có giá trị bằng $1$ nếu phần tử $z$ thuộc tập hợp $\mathcal{X}$ và bằng $0$ trong các trường hợp khác
* $\mathbf{(\cdot)}^\top$: chuyển vị của một vector hoặc ma trận
* $\mathbf{X}^{-1}$: nghịch đảo của ma trận $\mathbf{X}$
* $\odot$: tích Hadamard (theo từng phần tử)
* $[\cdot, \cdot]$: phép nối
* $\|\cdot\|_p$: chuẩn $\ell_p$
* $\|\cdot\|$: chuẩn $\ell_2$
* $\langle \mathbf{x}, \mathbf{y} \rangle$: tích trong (tích vô hướng) của các vector $\mathbf{x}$ và $\mathbf{y}$
* $\sum$: tổng trên một tập hợp các phần tử
* $\prod$: tích trên một tập hợp các phần tử
* $\stackrel{\textrm{def}}{=}$: một đẳng thức được khẳng định là định nghĩa của ký hiệu ở vế trái



## Giải tích

* $\frac{dy}{dx}$: đạo hàm của $y$ theo $x$
* $\frac{\partial y}{\partial x}$: đạo hàm riêng của $y$ theo $x$
* $\nabla_{\mathbf{x}} y$: gradient của $y$ theo $\mathbf{x}$
* $\int_a^b f(x) \;dx$: tích phân xác định của $f$ từ $a$ đến $b$ theo $x$
* $\int f(x) \;dx$: tích phân bất định của $f$ theo $x$



## Xác suất và lý thuyết thông tin

* $X$: một biến ngẫu nhiên
* $P$: một phân phối xác suất
* $X \sim P$: biến ngẫu nhiên $X$ tuân theo phân phối $P$
* $P(X=x)$: xác suất được gán cho biến cố trong đó biến ngẫu nhiên $X$ nhận giá trị $x$
* $P(X \mid Y)$: phân phối xác suất có điều kiện của $X$ khi biết $Y$
* $p(\cdot)$: hàm mật độ xác suất (PDF) gắn với phân phối $P$
* ${E}[X]$: kỳ vọng của biến ngẫu nhiên $X$
* $X \perp Y$: các biến ngẫu nhiên $X$ và $Y$ độc lập
* $X \perp Y \mid Z$: các biến ngẫu nhiên $X$ và $Y$ độc lập có điều kiện khi biết $Z$
* $\sigma_X$: độ lệch chuẩn của biến ngẫu nhiên $X$
* $\textrm{Var}(X)$: phương sai của biến ngẫu nhiên $X$, bằng $\sigma^2_X$
* $\textrm{Cov}(X, Y)$: hiệp phương sai của các biến ngẫu nhiên $X$ và $Y$
* $\rho(X, Y)$: hệ số tương quan Pearson giữa $X$ và $Y$, bằng $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: entropy của biến ngẫu nhiên $X$
* $D_{\textrm{KL}}(P\|Q)$: độ phân kỳ KL (hay entropy tương đối) từ phân phối $Q$ đến phân phối $P$



[Thảo luận](https://discuss.d2l.ai/t/25)