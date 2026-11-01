# Tích phân
:label:`sec_integral_calculus`

Phép vi phân chỉ chiếm một nửa nội dung của chương trình giải tích truyền thống. Trụ cột còn lại, phép tích phân, thoạt đầu dường như là một câu hỏi khá rời rạc: “Diện tích bên dưới đường cong này là bao nhiêu?” Mặc dù có vẻ không liên quan, phép tích phân gắn bó chặt chẽ với phép vi phân thông qua điều được gọi là *định lý cơ bản của giải tích*.

Ở mức độ machine learning được trình bày trong cuốn sách này, chúng ta không cần hiểu sâu về phép tích phân. Tuy nhiên, chúng tôi sẽ giới thiệu ngắn gọn để đặt nền tảng cho bất kỳ ứng dụng nào khác mà chúng ta sẽ gặp về sau.

## Diễn giải hình học
Giả sử chúng ta có một hàm $f(x)$. Để đơn giản, hãy giả sử rằng $f(x)$ không âm (không bao giờ nhận giá trị nhỏ hơn 0). Điều chúng ta muốn tìm hiểu là: diện tích nằm giữa $f(x)$ và trục $x$ là bao nhiêu?

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

Trong hầu hết các trường hợp, diện tích này sẽ là vô hạn hoặc không xác định (hãy xét diện tích bên dưới $f(x) = x^{2}$), vì vậy người ta thường nói về diện tích giữa một cặp đầu mút, chẳng hạn $a$ và $b$.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

Chúng ta sẽ ký hiệu diện tích này bằng ký hiệu tích phân dưới đây:

$$
\textrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

Biến bên trong là một biến giả, tương tự như chỉ số của một tổng trong $\sum$, vì vậy ta có thể viết tương đương bằng bất kỳ biến bên trong nào:

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

Có một cách truyền thống để tìm hiểu cách xấp xỉ các tích phân như vậy: ta có thể hình dung việc lấy vùng nằm giữa $a$ và $b$, rồi chia nó thành $N$ lát cắt dọc. Nếu $N$ lớn, ta có thể xấp xỉ diện tích của mỗi lát cắt bằng một hình chữ nhật, sau đó cộng diện tích của chúng lại để được tổng diện tích bên dưới đường cong. Hãy xem một ví dụ thực hiện điều này bằng code. Chúng ta sẽ thấy cách lấy được giá trị chính xác trong phần sau.

```{.python .input}
#@tab mxnet
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

Vấn đề là mặc dù có thể thực hiện bằng phương pháp số, ta chỉ có thể thực hiện cách tiếp cận này một cách giải tích đối với những hàm đơn giản nhất, chẳng hạn

$$
\int_a^b x \;dx.
$$

Bất kỳ hàm nào phức tạp hơn một chút, chẳng hạn như ví dụ trong đoạn code trên

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$

đều vượt quá khả năng giải bằng phương pháp trực tiếp như vậy.

Thay vào đó, chúng ta sẽ tiếp cận theo một cách khác. Chúng ta sẽ làm việc một cách trực quan với khái niệm diện tích và tìm hiểu công cụ tính toán chính được dùng để tìm tích phân: *định lý cơ bản của giải tích*. Đây sẽ là nền tảng cho việc nghiên cứu phép tích phân của chúng ta.

## Định lý cơ bản của giải tích

Để tìm hiểu sâu hơn về lý thuyết tích phân, hãy giới thiệu một hàm

$$
F(x) = \int_0^x f(y) dy.
$$

Hàm này đo diện tích giữa $0$ và $x$ tùy theo cách chúng ta thay đổi $x$. Lưu ý rằng đây là tất cả những gì chúng ta cần, vì

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

Đây là cách biểu diễn bằng toán học cho thực tế rằng ta có thể đo diện tích đến điểm đầu mút xa rồi trừ đi diện tích đến điểm đầu mút gần, như được minh họa trong :numref:`fig_area-subtract`.

![Hình dung lý do tại sao ta có thể chuyển bài toán tính diện tích bên dưới một đường cong giữa hai điểm thành bài toán tính diện tích bên trái một điểm.](../img/sub-area.svg)
:label:`fig_area-subtract`

Do đó, ta có thể tìm tích phân trên bất kỳ khoảng nào bằng cách xác định $F(x)$.

Để làm vậy, hãy xét một thí nghiệm. Như thường làm trong giải tích, hãy hình dung điều gì xảy ra khi ta dịch chuyển giá trị một lượng rất nhỏ. Từ nhận xét trên, ta biết rằng

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

Điều này cho biết hàm thay đổi một lượng bằng diện tích bên dưới một lát rất mỏng của hàm.

Đây là lúc chúng ta đưa ra một phép xấp xỉ. Nếu nhìn vào một lát diện tích rất mỏng như thế này, ta thấy diện tích đó gần với diện tích hình chữ nhật có chiều cao bằng giá trị của $f(x)$ và chiều rộng đáy bằng $\epsilon$. Thật vậy, có thể chứng minh rằng khi $\epsilon \rightarrow 0$, phép xấp xỉ này ngày càng chính xác hơn. Vì vậy, ta có thể kết luận:

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

Tuy nhiên, bây giờ ta nhận thấy: đây chính xác là dạng mà ta kỳ vọng khi tính đạo hàm của $F$! Do đó, ta thấy một sự thật khá đáng ngạc nhiên sau:

$$
\frac{dF}{dx}(x) = f(x).
$$

Đây chính là *định lý cơ bản của giải tích*. Ta có thể viết nó dưới dạng khai triển như sau:
$$\frac{d}{dx}\int_0^x  f(y) \; dy = f(x).$$
:eqlabel:`eq_ftc`

Định lý này lấy khái niệm tìm diện tích (*a priori* vốn khá khó) và rút gọn nó thành một mệnh đề về đạo hàm (thứ đã được hiểu rõ hơn nhiều). Một nhận xét cuối cùng cần nêu là định lý này không cho ta biết chính xác $F(x)$ là gì. Thật vậy, $F(x) + C$ với mọi $C$ đều có cùng đạo hàm. Đây là một thực tế không thể tránh khỏi trong lý thuyết tích phân. May mắn là khi làm việc với các tích phân xác định, các hằng số sẽ triệt tiêu và do đó không ảnh hưởng đến kết quả.

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

Điều này có vẻ như một sự trừu tượng vô nghĩa, nhưng hãy dành chút thời gian để nhận ra rằng nó đã đem lại cho chúng ta một góc nhìn hoàn toàn mới về việc tính tích phân. Mục tiêu của chúng ta không còn là thực hiện một quy trình kiểu chia nhỏ rồi cộng lại để tìm diện tích; thay vào đó, ta chỉ cần tìm một hàm có đạo hàm chính là hàm đã cho! Điều này thật đáng kinh ngạc, vì giờ đây ta có thể liệt kê nhiều tích phân khá khó bằng cách đảo ngược bảng từ :numref:`sec_derivative_table`. Chẳng hạn, ta biết đạo hàm của $x^{n}$ là $nx^{n-1}$. Do đó, sử dụng định lý cơ bản :eqref:`eq_ftc`, ta có thể viết

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

Tương tự, ta biết đạo hàm của $e^{x}$ chính là nó, nên

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

Theo cách này, ta có thể xây dựng toàn bộ lý thuyết tích phân bằng cách tự do vận dụng các ý tưởng từ giải tích vi phân. Mọi quy tắc tích phân đều bắt nguồn từ một sự thật duy nhất này.

## Đổi biến
:label:`subsec_integral_example`

Cũng như với phép vi phân, có một số quy tắc giúp việc tính tích phân trở nên dễ thực hiện hơn. Trên thực tế, mọi quy tắc của giải tích vi phân (chẳng hạn quy tắc tích, quy tắc tổng và quy tắc chuỗi) đều có một quy tắc tương ứng trong giải tích tích phân (lần lượt là tích phân từng phần, tính tuyến tính của phép tích phân và công thức đổi biến). Trong phần này, chúng ta sẽ tìm hiểu quy tắc được cho là quan trọng nhất trong danh sách: công thức đổi biến.

Trước hết, giả sử ta có một hàm bản thân nó là một tích phân:

$$
F(x) = \int_0^x f(y) \; dy.
$$

Giả sử ta muốn biết hàm này có dạng như thế nào khi hợp thành với một hàm khác để thu được $F(u(x))$. Theo quy tắc chuỗi, ta biết

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

Ta có thể chuyển điều này thành một mệnh đề về tích phân bằng cách sử dụng định lý cơ bản :eqref:`eq_ftc` như trên. Ta được

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Nhớ rằng $F$ bản thân là một tích phân, nên vế trái có thể được viết lại thành

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Tương tự, việc nhớ rằng $F$ là một tích phân cho phép ta nhận ra rằng $\frac{dF}{dx} = f$ bằng cách sử dụng định lý cơ bản :eqref:`eq_ftc`, và do đó ta có thể kết luận

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

Đây là công thức *đổi biến*.

Để có một cách suy ra trực quan hơn, hãy xét điều xảy ra khi ta lấy tích phân của $f(u(x))$ từ $x$ đến $x+\epsilon$. Với một $\epsilon$ nhỏ, tích phân này xấp xỉ $\epsilon f(u(x))$, tức là diện tích của hình chữ nhật tương ứng. Bây giờ, hãy so sánh điều này với tích phân của $f(y)$ từ $u(x)$ đến $u(x+\epsilon)$. Ta biết rằng $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$, vì vậy diện tích của hình chữ nhật này xấp xỉ $\epsilon \frac{du}{dx}(x)f(u(x))$. Do đó, để diện tích của hai hình chữ nhật này bằng nhau, ta cần nhân hình chữ nhật thứ nhất với $\frac{du}{dx}(x)$, như được minh họa trong :numref:`fig_rect-transform`.

![Hình dung sự biến đổi của một hình chữ nhật mỏng đơn lẻ dưới phép đổi biến.](../img/rect-trans.svg)
:label:`fig_rect-transform`

Điều này cho ta

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$

Đây là công thức đổi biến được biểu diễn cho một hình chữ nhật nhỏ đơn lẻ.

Nếu $u(x)$ và $f(x)$ được chọn phù hợp, công thức này có thể cho phép tính những tích phân cực kỳ phức tạp. Chẳng hạn, nếu ta chọn $f(y) = 1$ và $u(x) = e^{-x^{2}}$ (điều này có nghĩa là $\frac{du}{dx}(x) = -2xe^{-x^{2}}$), ta có thể chỉ ra, chẳng hạn, rằng

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

và do đó, bằng cách sắp xếp lại, ta có

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## Nhận xét về quy ước dấu

Những độc giả tinh ý sẽ nhận thấy một điều kỳ lạ trong các phép tính trên. Cụ thể, những phép tính như

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

có thể cho ra các số âm. Khi nghĩ về diện tích, việc thấy một giá trị âm có thể gây khó hiểu, vì vậy ta nên tìm hiểu kỹ hơn về quy ước này.

Các nhà toán học sử dụng khái niệm diện tích có dấu. Điều này thể hiện theo hai cách. Thứ nhất, nếu xét một hàm $f(x)$ đôi khi nhỏ hơn không, thì diện tích cũng sẽ âm. Chẳng hạn,

$$
\int_0^{1} (-1)\;dx = -1.
$$

Tương tự, các tích phân tiến từ phải sang trái thay vì từ trái sang phải cũng được xem là các diện tích âm:

$$
\int_0^{-1} 1\; dx = -1.
$$

Diện tích thông thường (từ trái sang phải của một hàm dương) luôn dương. Bất cứ điều gì có được bằng cách lật nó (chẳng hạn lật qua trục $x$ để lấy tích phân của một số âm, hoặc lật qua trục $y$ để lấy tích phân theo thứ tự ngược) sẽ tạo ra một diện tích âm. Và đúng vậy, lật hai lần sẽ tạo ra hai dấu âm triệt tiêu lẫn nhau, cho diện tích dương:

$$
\int_0^{-1} (-1)\;dx =  1.
$$

Nếu phần thảo luận này nghe có vẻ quen thuộc thì đúng là như vậy! Trong :numref:`sec_geometry-linear-algebraic-ops`, chúng ta đã thảo luận về cách định thức biểu diễn diện tích có dấu theo cách tương tự.

## Tích phân nhiều lớp

Trong một số trường hợp, chúng ta sẽ cần làm việc trong các không gian nhiều chiều hơn. Chẳng hạn, giả sử ta có một hàm của hai biến, như $f(x, y)$, và muốn biết thể tích dưới $f$ khi $x$ chạy trên $[a, b]$ còn $y$ chạy trên $[c, d]$.

```{.python .input}
#@tab mxnet
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy())
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

Ta viết điều này như sau:

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Giả sử ta muốn tính tích phân này. Tôi khẳng định rằng ta có thể thực hiện bằng cách tính lần lượt: trước hết tính tích phân theo $x$, sau đó chuyển sang tích phân theo $y$, tức là

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

Hãy xem tại sao điều này đúng.

Xét hình minh họa ở trên, trong đó ta chia hàm thành các hình vuông $\epsilon \times \epsilon$ được đánh chỉ số bằng các tọa độ nguyên $i, j$. Khi đó, tích phân của ta xấp xỉ

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

Sau khi rời rạc hóa bài toán, ta có thể cộng các giá trị trên những hình vuông này theo bất kỳ thứ tự nào mà không cần lo lắng về việc thay đổi kết quả. Điều này được minh họa trong :numref:`fig_sum-order`. Cụ thể, ta có thể viết

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![Minh họa cách phân rã một tổng trên nhiều hình vuông thành tổng theo các cột trước (1), sau đó cộng các tổng của từng cột lại với nhau (2).](../img/sum-order.svg)
:label:`fig_sum-order`

Tổng ở bên trong chính là phép rời rạc hóa của tích phân

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

Cuối cùng, hãy lưu ý rằng khi kết hợp hai biểu thức này, ta được

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Như vậy, gộp tất cả lại, ta có

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

Lưu ý rằng sau khi rời rạc hóa, tất cả những gì ta làm chỉ là sắp xếp lại thứ tự cộng một danh sách các số. Điều này có thể khiến ta nghĩ rằng kết quả này không có gì đặc biệt, nhưng kết quả này (được gọi là *Định lý Fubini*) không phải lúc nào cũng đúng! Đối với loại toán học thường gặp trong machine learning (các hàm liên tục), không có gì đáng lo ngại. Tuy nhiên, có thể xây dựng những ví dụ mà định lý thất bại (chẳng hạn hàm $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$ trên hình chữ nhật $[0,2]\times[0,1]$).

Lưu ý rằng việc chọn tính tích phân theo $x$ trước rồi theo $y$ là tùy ý. Ta cũng có thể chọn tính theo $y$ trước rồi theo $x$ để được

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

Thông thường, ta sẽ rút gọn bằng ký hiệu vector và nói rằng với $U = [a, b]\times [c, d]$, biểu thức này là

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## Đổi biến trong tích phân nhiều lớp

Cũng như với biến đơn trong :eqref:`eq_change_var`, khả năng đổi biến trong một tích phân nhiều chiều là một công cụ quan trọng. Hãy tóm tắt kết quả mà không dẫn xuất.

Ta cần một hàm tái tham số hóa miền tích phân. Ta có thể chọn $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$, tức là một hàm nhận vào $n$ biến thực và trả về một bộ gồm $n$ biến thực khác. Để các biểu thức gọn hơn, ta giả sử $\phi$ là *đơn ánh* (*injective*), nghĩa là nó không bao giờ tự gấp lên chính nó ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$).

Trong trường hợp này, ta có thể viết

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

trong đó $D\phi$ là *Jacobian* của $\phi$, tức ma trận các đạo hàm riêng của $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$,

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

Quan sát kỹ, ta thấy điều này tương tự quy tắc chain rule cho một biến :eqref:`eq_change_var`, ngoại trừ việc ta thay số hạng $\frac{du}{dx}(x)$ bằng $\left|\det(D\phi(\mathbf{x}))\right|$. Hãy xem cách diễn giải số hạng này. Nhớ lại rằng số hạng $\frac{du}{dx}(x)$ cho biết ta kéo giãn trục $x$ bao nhiêu khi áp dụng $u$. Trong các chiều cao hơn, quy trình tương tự là xác định ta kéo giãn diện tích (hoặc thể tích, hay siêu thể tích) của một hình vuông nhỏ (hoặc một *hyper-cube* nhỏ) bao nhiêu khi áp dụng $\boldsymbol{\phi}$. Nếu $\boldsymbol{\phi}$ là phép nhân với một ma trận, thì ta đã biết định thức cho ta câu trả lời như thế nào.

Với một chút nỗ lực, ta có thể chứng minh rằng *Jacobian* cung cấp xấp xỉ tốt nhất cho một hàm nhiều biến $\boldsymbol{\phi}$ tại một điểm bằng một ma trận, tương tự như cách ta có thể xấp xỉ bằng các đường thẳng hoặc mặt phẳng nhờ các đạo hàm và gradient. Vì vậy, định thức của Jacobian chính xác phản ánh hệ số co giãn mà ta đã xác định trong một chiều.

Việc bổ sung đầy đủ các chi tiết đòi hỏi một chút công sức, vì vậy đừng lo nếu hiện giờ chúng chưa rõ ràng. Hãy xem ít nhất một ví dụ mà sau này ta sẽ sử dụng. Xét tích phân

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

Tính trực tiếp tích phân này sẽ không đưa ta đến đâu, nhưng nếu đổi biến, ta có thể tiến triển đáng kể. Nếu đặt $\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$ (nghĩa là $x = r \cos(\theta)$, $y = r \sin(\theta)$), thì ta có thể áp dụng công thức đổi biến để thấy rằng tích phân này tương đương với

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

trong đó

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

Do đó, tích phân là

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

trong đó đẳng thức cuối cùng được suy ra bằng phép tính tương tự như phép tính ta đã sử dụng trong phần :numref:`subsec_integral_example`.

Ta sẽ gặp lại tích phân này khi nghiên cứu các biến ngẫu nhiên liên tục trong :numref:`sec_random_variables`.

## Tóm tắt

* Lý thuyết tích phân cho phép chúng ta trả lời các câu hỏi về diện tích hoặc thể tích.
* Định lý cơ bản của giải tích cho phép ta tận dụng kiến thức về đạo hàm để tính diện tích, thông qua nhận xét rằng đạo hàm của diện tích tính đến một điểm nào đó bằng giá trị của hàm đang được tích phân.
* Các tích phân trong không gian nhiều chiều có thể được tính bằng cách lặp lại các tích phân một biến.

## Bài tập
1. $\int_1^2 \frac{1}{x} \;dx$ bằng bao nhiêu?
2. Sử dụng công thức đổi biến để tính $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$.
3. $\int_{[0,1]^2} xy \;dx\;dy$ bằng bao nhiêu?
4. Sử dụng công thức đổi biến để tính $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ và $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$ để thấy rằng chúng khác nhau.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1092)
:end_tab:


:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1093)
:end_tab: