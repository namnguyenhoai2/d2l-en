```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Lặp giá trị
:label:`sec_valueiter`

Trong phần này, chúng ta sẽ thảo luận cách chọn hành động tốt nhất cho robot tại mỗi trạng thái để tối đa hóa *return* của quỹ đạo. Chúng ta sẽ mô tả một thuật toán có tên là Value Iteration và triển khai thuật toán này cho một robot mô phỏng di chuyển trên một hồ nước đóng băng.

## Chính sách ngẫu nhiên

Một chính sách ngẫu nhiên được ký hiệu là $\pi(a \mid s)$ (gọi tắt là policy), là một phân phối có điều kiện trên các hành động $a \in \mathcal{A}$ khi biết trạng thái $s \in \mathcal{S}$, $\pi(a \mid s) \equiv P(a \mid s)$. Ví dụ, nếu robot có bốn hành động $\mathcal{A}=$ {go left, go down, go right, go up}. Chính sách tại một trạng thái $s \in \mathcal{S}$ với tập hành động như vậy $\mathcal{A}$ là một phân phối categorical, trong đó xác suất của bốn hành động có thể là $[0.4, 0.2, 0.1, 0.3]$; tại một trạng thái khác $s' \in \mathcal{S}$, xác suất $\pi(a \mid s')$ của cùng bốn hành động có thể là $[0.1, 0.1, 0.2, 0.6]$. Lưu ý rằng ta cần có $\sum_a \pi(a \mid s) = 1$ với mọi trạng thái $s$. Chính sách tất định là một trường hợp đặc biệt của chính sách ngẫu nhiên, trong đó phân phối $\pi(a \mid s)$ chỉ gán xác suất khác 0 cho một hành động cụ thể, chẳng hạn $[1, 0, 0, 0]$ trong ví dụ có bốn hành động của chúng ta.

Để ký hiệu gọn hơn, chúng ta thường viết $\pi(s)$ thay cho phân phối có điều kiện $\pi(a \mid s)$.

## Hàm giá trị

Bây giờ hãy tưởng tượng robot bắt đầu tại trạng thái $s_0$ và ở mỗi thời điểm, trước tiên lấy mẫu một hành động từ policy $a_t \sim \pi(s_t)$ rồi thực hiện hành động này để chuyển sang trạng thái tiếp theo $s_{t+1}$. Quỹ đạo $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ có thể khác nhau tùy thuộc vào hành động cụ thể $a_t$ được lấy mẫu tại các thời điểm trung gian. Chúng ta định nghĩa *return* trung bình $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$ của tất cả các quỹ đạo như vậy là
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

trong đó $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ là trạng thái tiếp theo của robot và $r(s_t, a_t)$ là reward tức thời nhận được khi thực hiện hành động $a_t$ tại trạng thái $s_t$ ở thời điểm $t$. Đây được gọi là “hàm giá trị” của policy $\pi$. Nói một cách đơn giản, giá trị của trạng thái $s_0$ đối với policy $\pi$, được ký hiệu là $V^\pi(s_0)$, là *return* được chiết khấu theo $\gamma$ kỳ vọng mà robot nhận được nếu bắt đầu tại trạng thái $s_0$ và thực hiện các hành động từ policy $\pi$ ở mỗi thời điểm.

Tiếp theo, chúng ta chia quỹ đạo thành hai giai đoạn: (i) giai đoạn đầu tương ứng với $s_0 \to s_1$ khi thực hiện hành động $a_0$, và (ii) giai đoạn thứ hai là quỹ đạo $\tau' = (s_1, a_1, r_1, \ldots)$ sau đó. Ý tưởng cốt lõi đằng sau mọi thuật toán trong reinforcement learning là giá trị của trạng thái $s_0$ có thể được viết thành reward trung bình nhận được trong giai đoạn đầu cộng với hàm giá trị lấy trung bình trên tất cả các trạng thái tiếp theo có thể có $s_1$. Điều này khá trực quan và xuất phát từ giả định Markov của chúng ta: *return* trung bình từ trạng thái hiện tại là tổng của *return* trung bình từ trạng thái tiếp theo và reward trung bình của việc chuyển đến trạng thái tiếp theo. Về mặt toán học, ta viết hai giai đoạn như sau:

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$
:eqlabel:`eq_dynamic_programming`

Phân rã này rất mạnh: nó là nền tảng của nguyên lý dynamic programming, trên đó mọi thuật toán reinforcement learning đều dựa vào. Lưu ý rằng giai đoạn thứ hai có hai kỳ vọng: một kỳ vọng trên các lựa chọn của hành động $a_0$ được thực hiện trong giai đoạn đầu bằng policy ngẫu nhiên, và một kỳ vọng khác trên các trạng thái có thể có $s_1$ nhận được từ hành động đã chọn. Chúng ta có thể viết :eqref:`eq_dynamic_programming` bằng cách sử dụng các xác suất chuyển tiếp trong Markov decision process (MDP) như sau:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$
:eqlabel:`eq_dynamic_programming_val`

Một điều quan trọng cần nhận thấy là đẳng thức trên đúng với mọi trạng thái $s \in \mathcal{S}$, vì ta có thể xem bất kỳ quỹ đạo nào bắt đầu từ trạng thái đó và chia quỹ đạo thành hai giai đoạn.

## Hàm giá trị hành động

Trong các triển khai, thường hữu ích khi duy trì một đại lượng có tên là hàm “giá trị hành động”, có liên hệ chặt chẽ với hàm giá trị. Hàm này được định nghĩa là *return* trung bình của một quỹ đạo bắt đầu tại $s_0$, nhưng hành động của giai đoạn đầu được cố định là $a_0$

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

lưu ý rằng tổng bên trong kỳ vọng bắt đầu từ $t=1,\ldots, \infty$ vì reward của giai đoạn đầu được cố định trong trường hợp này. Một lần nữa, chúng ta có thể chia quỹ đạo thành hai phần và viết

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$
:eqlabel:`eq_dynamic_programming_q`

Phiên bản này tương tự :eqref:`eq_dynamic_programming_val` đối với hàm giá trị hành động.

## Chính sách ngẫu nhiên tối ưu

Cả hàm giá trị và hàm giá trị hành động đều phụ thuộc vào policy mà robot lựa chọn. Tiếp theo, chúng ta sẽ xem xét “policy tối ưu” đạt được *return* trung bình lớn nhất
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

Trong tất cả các policy ngẫu nhiên có thể được robot thực hiện, policy tối ưu $\pi^*$ đạt *return* trung bình được chiết khấu lớn nhất cho các quỹ đạo bắt đầu từ trạng thái $s_0$. Ta ký hiệu hàm giá trị và hàm giá trị hành động của policy tối ưu lần lượt là $V^* \equiv V^{\pi^*}$ và $Q^* \equiv Q^{\pi^*}$.

Hãy nhận thấy rằng với một policy tất định, tại mỗi trạng thái bất kỳ chỉ có một hành động có thể được thực hiện theo policy. Khi đó, ta có

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

Một cách ghi nhớ hữu ích là hành động tối ưu tại trạng thái $s$ (đối với policy tất định) là hành động tối đa hóa tổng của reward $r(s, a)$ từ giai đoạn đầu và *return* trung bình của các quỹ đạo bắt đầu từ trạng thái tiếp theo $s'$, được lấy trung bình trên tất cả các trạng thái tiếp theo $s'$ có thể có trong giai đoạn thứ hai.

## Nguyên lý Dynamic Programming

Phát triển của chúng ta trong phần trước tại :eqref:`eq_dynamic_programming` hoặc :eqref:`eq_dynamic_programming_q` có thể được chuyển thành một thuật toán để tính hàm giá trị tối ưu $V^*$ hoặc hàm giá trị hành động $Q^*$ tương ứng. Hãy quan sát rằng
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Đối với policy tối ưu tất định $\pi^*$, vì chỉ có một hành động có thể được thực hiện tại trạng thái $s$, ta cũng có thể viết

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

với mọi trạng thái $s \in \mathcal{S}$. Đẳng thức này được gọi là “nguyên lý dynamic programming” :cite:`BellmanDPPaper,BellmanDPBook`. Nguyên lý này được Richard Bellman xây dựng vào những năm 1950 và có thể được ghi nhớ là “phần còn lại của một quỹ đạo tối ưu cũng là tối ưu”.

## Lặp giá trị

Chúng ta có thể chuyển nguyên lý dynamic programming thành một thuật toán để tìm hàm giá trị tối ưu, có tên là value iteration. Ý tưởng cốt lõi của value iteration là xem đẳng thức này như một tập hợp các ràng buộc liên kết $V^*(s)$ tại các trạng thái khác nhau $s \in \mathcal{S}$. Chúng ta khởi tạo hàm giá trị bằng một số giá trị tùy ý $V_0(s)$ cho mọi trạng thái $s \in \mathcal{S}$. Ở lần lặp thứ $k^{\textrm{th}}$, thuật toán Value Iteration cập nhật hàm giá trị như sau:

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

Khi $k \to \infty$, hàm giá trị được ước lượng bởi thuật toán Value Iteration hội tụ đến hàm giá trị tối ưu bất kể khởi tạo $V_0$,
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Thuật toán Value Iteration tương tự có thể được viết bằng hàm giá trị hành động như sau:
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

Trong trường hợp này, chúng ta khởi tạo $Q_0(s, a)$ bằng một số giá trị tùy ý cho mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$. Một lần nữa, ta có $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ với mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$.

## Đánh giá policy

Value Iteration cho phép chúng ta tính hàm giá trị tối ưu, tức là $V^{\pi^*}$ của policy xác định tối ưu $\pi^*$. Chúng ta cũng có thể sử dụng các phép cập nhật lặp tương tự để tính hàm giá trị gắn với bất kỳ policy nào khác, có thể là policy stochastic $\pi$. Một lần nữa, chúng ta khởi tạo $V^\pi_0(s)$ bằng một số giá trị tùy ý cho mọi state $s \in \mathcal{S}$ và tại lần lặp thứ $k^{\textrm{th}}$, thực hiện các phép cập nhật

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Thuật toán này được gọi là policy evaluation và hữu ích để tính hàm giá trị khi đã biết policy. Một lần nữa, ta có thể chứng minh rằng khi $k \to \infty$, các phép cập nhật này hội tụ về hàm giá trị chính xác bất kể khởi tạo $V_0$ là gì,

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Thuật toán tính hàm giá trị hành động $Q^\pi(s, a)$ của một policy $\pi$ cũng tương tự.

## Triển khai Value Iteration
:label:`subsec_valueitercode`
Tiếp theo, chúng ta sẽ trình bày cách triển khai Value Iteration cho một bài toán điều hướng có tên là FrozenLake từ [Open AI Gym](https://gym.openai.com). Trước tiên, chúng ta cần thiết lập environment như trong đoạn code sau.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations
random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

Trong environment FrozenLake, robot di chuyển trên một lưới $4 \times 4$ (đây là các state) với các action là "up" ($\uparrow$), "down" ($\rightarrow$), "left" ($\leftarrow$) và "right" ($\rightarrow$). Environment chứa một số ô hố (H) và ô băng (F), cũng như một ô đích (G), tất cả đều chưa được robot biết. Để bài toán đơn giản, chúng ta giả sử robot có các action đáng tin cậy, tức là $P(s' \mid s, a) = 1$ với mọi $s \in \mathcal{S}, a \in \mathcal{A}$. Nếu robot đến được đích, trial kết thúc và robot nhận reward bằng $1$ bất kể action nào; reward tại mọi state khác là $0$ đối với tất cả action. Mục tiêu của robot là học một policy đi từ vị trí bắt đầu cho trước (S) (đây là $s_0$) đến vị trí đích (G) để tối đa hóa *return*.

Hàm sau triển khai Value Iteration, trong đó `env_info` chứa thông tin liên quan đến MDP và environment, còn `gamma` là discount factor:

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

Các hình trên hiển thị policy (mũi tên biểu thị action) và hàm giá trị (sự thay đổi màu sắc cho thấy hàm giá trị thay đổi theo thời gian từ giá trị ban đầu được biểu thị bằng màu tối đến giá trị tối ưu được biểu thị bằng các màu sáng). Như chúng ta thấy, Value Iteration tìm được hàm giá trị tối ưu sau 10 lần lặp và có thể đến state đích (G) từ bất kỳ state nào, miễn là state đó không phải là ô H. Một khía cạnh thú vị khác của triển khai này là ngoài việc tìm được hàm giá trị tối ưu, chúng ta còn tự động tìm được policy tối ưu $\pi^*$ tương ứng với hàm giá trị này.


## Tóm tắt
Ý tưởng chính đằng sau thuật toán Value Iteration là sử dụng nguyên lý dynamic programming để tìm return trung bình tối ưu nhận được từ một state cho trước. Lưu ý rằng để triển khai thuật toán Value Iteration, chúng ta cần biết đầy đủ Markov decision process (MDP), chẳng hạn như các hàm transition và reward.


## Bài tập

1. Hãy thử tăng kích thước lưới lên $8 \times 8$. So với lưới $4 \times 4$, cần bao nhiêu lần lặp để tìm được hàm giá trị tối ưu?
1. Độ phức tạp tính toán của thuật toán Value Iteration là bao nhiêu?
1. Chạy lại thuật toán Value Iteration với $\gamma$ (tức là "gamma" trong đoạn code trên) lần lượt bằng $0$, $0.5$ và $1$, rồi phân tích kết quả.
1. Giá trị của $\gamma$ ảnh hưởng thế nào đến số lần lặp cần thiết để Value Iteration hội tụ? Điều gì xảy ra khi $\gamma=1$?

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12005)
:end_tab: