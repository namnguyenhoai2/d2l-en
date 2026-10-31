```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q-Learning
:label:`sec_qlearning`

Trong phần trước, chúng ta đã thảo luận về thuật toán Value Iteration, vốn yêu cầu truy cập vào toàn bộ quy trình quyết định Markov (MDP), chẳng hạn như các hàm chuyển trạng thái và phần thưởng. Trong phần này, chúng ta sẽ tìm hiểu Q-Learning :cite:`Watkins.Dayan.1992`, một thuật toán học hàm giá trị mà không nhất thiết phải biết MDP. Thuật toán này thể hiện ý tưởng cốt lõi đằng sau reinforcement learning: nó cho phép robot tự thu thập dữ liệu.
<!-- , thay vì phụ thuộc vào chuyên gia. -->

## Thuật toán Q-Learning

Hãy nhớ rằng value iteration cho hàm action-value trong :ref:`sec_valueiter` tương ứng với phép cập nhật

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

Như đã thảo luận, việc triển khai thuật toán này yêu cầu biết MDP, cụ thể là hàm chuyển trạng thái $P(s' \mid s, a)$. Ý tưởng then chốt của Q-Learning là thay tổng theo mọi $s' \in \mathcal{S}$ trong biểu thức trên bằng tổng theo các trạng thái mà robot đã đi qua. Điều này giúp loại bỏ nhu cầu phải biết hàm chuyển trạng thái.

## Bài toán tối ưu đằng sau Q-Learning

Hãy giả sử robot sử dụng một policy $\pi_e(a \mid s)$ để thực hiện các hành động. Giống như trong chương trước, nó thu thập một dataset gồm $n$ trajectory, mỗi trajectory có $T$ timestep $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$. Hãy nhớ rằng value iteration thực chất là một tập các ràng buộc liên kết action-value $Q^*(s, a)$ của những trạng thái và hành động khác nhau với nhau. Chúng ta có thể triển khai một phiên bản xấp xỉ của value iteration bằng dữ liệu robot đã thu thập khi sử dụng $\pi_e$ như sau:

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

Trước tiên, hãy quan sát những điểm tương đồng và khác biệt giữa biểu thức này với value iteration ở trên. Nếu policy $\pi_e$ của robot bằng policy tối ưu $\pi^*$ và nếu nó thu thập một lượng dữ liệu vô hạn, thì bài toán tối ưu này sẽ giống hệt bài toán tối ưu nằm bên dưới value iteration. Tuy nhiên, trong khi value iteration yêu cầu chúng ta biết $P(s' \mid s, a)$, hàm mục tiêu tối ưu không chứa hạng tử này. Chúng ta không gian lận: khi robot sử dụng policy $\pi_e$ để thực hiện hành động $a_t^i$ tại trạng thái $s_t^i$, trạng thái tiếp theo $s_{t+1}^i$ là một mẫu được lấy từ hàm chuyển trạng thái. Vì vậy, hàm mục tiêu tối ưu cũng có quyền truy cập vào hàm chuyển trạng thái, nhưng một cách ẩn thông qua dữ liệu robot đã thu thập.

Các biến của bài toán tối ưu là $Q(s, a)$ với mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$. Chúng ta có thể tối thiểu hóa hàm mục tiêu bằng gradient descent. Với mỗi cặp $(s_t^i, a_t^i)$ trong dataset, ta có thể viết

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$
:eqlabel:`q_learning`

trong đó $\alpha$ là learning rate. Thông thường, trong các bài toán thực tế, khi robot đến vị trí mục tiêu, trajectory sẽ kết thúc. Giá trị của trạng thái terminal như vậy bằng 0 vì robot không thực hiện thêm hành động nào sau trạng thái này. Chúng ta nên điều chỉnh phép cập nhật để xử lý các trạng thái như vậy:

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

trong đó $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ là một biến chỉ báo có giá trị bằng một nếu $s_{t+1}^i$ là trạng thái terminal và bằng không trong các trường hợp khác. Giá trị của các bộ state-action $(s, a)$ không có trong dataset được đặt bằng $-\infty$. Thuật toán này được gọi là Q-Learning.

Với nghiệm của các phép cập nhật này là $\hat{Q}$, một giá trị xấp xỉ của hàm giá trị tối ưu $Q^*$, chúng ta có thể dễ dàng thu được policy tất định tối ưu tương ứng với hàm giá trị này bằng

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

Có thể xảy ra trường hợp nhiều policy tất định tương ứng với cùng một hàm giá trị tối ưu; các trường hợp hòa như vậy có thể được phá vỡ tùy ý vì chúng có cùng hàm giá trị.

## Exploration trong Q-Learning

Policy mà robot sử dụng để thu thập dữ liệu $\pi_e$ đóng vai trò then chốt trong việc đảm bảo Q-Learning hoạt động tốt. Sau cùng, chúng ta đã thay thế kỳ vọng theo $s'$ sử dụng hàm chuyển trạng thái $P(s' \mid s, a)$ bằng dữ liệu robot đã thu thập. Nếu policy $\pi_e$ không đi đến những phần đa dạng của không gian state-action, thì dễ hình dung rằng ước lượng $\hat{Q}$ của chúng ta sẽ là một xấp xỉ kém của $Q^*$ tối ưu. Cũng cần lưu ý rằng trong tình huống này, ước lượng của $Q^*$ tại *mọi trạng thái* $s \in \mathcal{S}$ đều sẽ kém, chứ không chỉ tại những trạng thái mà $\pi_e$ đã đi qua. Điều này là do hàm mục tiêu Q-Learning (hoặc value iteration) là một ràng buộc liên kết giá trị của mọi cặp state-action. Vì vậy, việc chọn đúng policy $\pi_e$ để thu thập dữ liệu là vô cùng quan trọng.

Chúng ta có thể giảm bớt vấn đề này bằng cách chọn một policy hoàn toàn ngẫu nhiên $\pi_e$, policy này lấy mẫu các hành động đồng đều ngẫu nhiên từ $\mathcal{A}$. Một policy như vậy sẽ đi qua mọi trạng thái, nhưng sẽ cần một số lượng lớn trajectory trước khi làm được điều đó.

Do đó, chúng ta đi đến ý tưởng then chốt thứ hai trong Q-Learning, đó là exploration. Các triển khai Q-Learning điển hình liên kết ước lượng hiện tại của $Q$ với policy $\pi_e$ để đặt

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

trong đó $\epsilon$ được gọi là “tham số exploration” và do người dùng lựa chọn. Policy $\pi_e$ được gọi là exploration policy. $\pi_e$ cụ thể này được gọi là exploration policy $\epsilon$-greedy vì nó chọn hành động tối ưu (theo ước lượng hiện tại $\hat{Q}$) với xác suất $1-\epsilon$, nhưng thực hiện exploration ngẫu nhiên với xác suất còn lại là $\epsilon$. Chúng ta cũng có thể sử dụng exploration policy softmax, còn gọi là

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

trong đó hyper-parameter $T$ được gọi là temperature. Giá trị lớn của $\epsilon$ trong policy $\epsilon$-greedy có tác dụng tương tự giá trị temperature $T$ lớn trong policy softmax.

Điều quan trọng cần lưu ý là khi chọn một exploration phụ thuộc vào ước lượng hiện tại của hàm action-value $\hat{Q}$, chúng ta cần giải lại bài toán tối ưu theo định kỳ. Các triển khai Q-Learning điển hình thực hiện một mini-batch update bằng một vài cặp state-action trong dataset đã thu thập (thường là các cặp được thu thập từ timestep trước đó của robot) sau mỗi lần thực hiện hành động bằng $\pi_e$.

## Tính chất “Tự hiệu chỉnh” của Q-Learning

Dataset do robot thu thập trong quá trình Q-Learning tăng lên theo thời gian. Cả exploration policy $\pi_e$ và ước lượng $\hat{Q}$ đều thay đổi khi robot thu thập thêm dữ liệu. Điều này đem lại cho chúng ta một insight quan trọng về lý do Q-Learning hoạt động tốt. Xét một trạng thái $s$: nếu một hành động cụ thể $a$ có giá trị lớn theo ước lượng hiện tại $\hat{Q}(s,a)$, thì cả exploration policy $\epsilon$-greedy và softmax đều có xác suất chọn hành động này cao hơn. Nếu hành động này thực sự *không phải* là hành động lý tưởng, thì các trạng thái tương lai phát sinh từ hành động này sẽ có phần thưởng thấp. Do đó, lần cập nhật tiếp theo của hàm mục tiêu Q-Learning sẽ làm giảm giá trị $\hat{Q}(s,a)$, từ đó làm giảm xác suất chọn hành động này vào lần tiếp theo robot đến trạng thái $s$. Các hành động không tốt, chẳng hạn những hành động bị ước lượng quá cao trong $\hat{Q}(s,a)$, sẽ được robot khám phá nhưng giá trị của chúng sẽ được hiệu chỉnh trong lần cập nhật tiếp theo của hàm mục tiêu Q-Learning. Các hành động tốt, chẳng hạn những hành động có giá trị $\hat{Q}(s, a)$ lớn, sẽ được robot khám phá thường xuyên hơn và nhờ đó được củng cố. Tính chất này có thể được dùng để chứng minh rằng Q-Learning có thể hội tụ đến policy tối ưu ngay cả khi bắt đầu với một policy ngẫu nhiên $\pi_e$ :cite:`Watkins.Dayan.1992`.

Khả năng không chỉ thu thập dữ liệu mới mà còn thu thập đúng loại dữ liệu là đặc điểm trung tâm của các thuật toán reinforcement learning, và đây là điều phân biệt chúng với supervised learning. Q-Learning, khi sử dụng deep neural network (mà chúng ta sẽ tìm hiểu trong chương DQN sau), là yếu tố dẫn đến sự hồi sinh của reinforcement learning :cite:`mnih2013playing`.

## Triển khai Q-Learning

Bây giờ, chúng ta sẽ trình bày cách triển khai Q-Learning trên FrozenLake từ [Open AI Gym](https://gym.openai.com). Lưu ý rằng đây cũng là thiết lập mà chúng ta xem xét trong thí nghiệm :ref:`sec_valueiter`.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

Trong môi trường FrozenLake, robot di chuyển trên một lưới $4 \times 4$ (đây là các trạng thái) với các hành động là "lên" ($\uparrow$), "xuống" ($\rightarrow$), "trái" ($\leftarrow$) và "phải" ($\rightarrow$). Môi trường chứa một số ô hố (H) và ô băng (F), cũng như một ô đích (G); tất cả đều chưa được biết đối với robot. Để bài toán đơn giản, ta giả sử robot thực hiện các hành động đáng tin cậy, tức là $P(s' \mid s, a) = 1$ với mọi $s \in \mathcal{S}, a \in \mathcal{A}$. Nếu robot đến được đích, lượt thử kết thúc và robot nhận phần thưởng bằng $1$ bất kể hành động nào; phần thưởng tại mọi trạng thái khác là $0$ đối với tất cả các hành động. Mục tiêu của robot là học một policy đưa robot từ vị trí bắt đầu cho trước (S) (đây là $s_0$) đến vị trí đích (G) để tối đa hóa *return*.

Trước tiên, ta triển khai phương pháp $\epsilon$-greedy như sau:

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

Bây giờ, ta đã sẵn sàng triển khai Q-learning:

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Reset environment
        state, done = env.reset(), False
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

Kết quả này cho thấy Q-learning có thể tìm ra nghiệm tối ưu cho bài toán này sau khoảng 250 vòng lặp. Tuy nhiên, khi so sánh kết quả này với kết quả của thuật toán Value Iteration (xem :ref:`subsec_valueitercode`), ta có thể thấy thuật toán Value Iteration cần ít vòng lặp hơn nhiều để tìm ra nghiệm tối ưu cho bài toán này. Điều này xảy ra vì thuật toán Value Iteration có quyền truy cập vào MDP đầy đủ, trong khi Q-learning thì không.


## Tóm tắt
Q-learning là một trong những thuật toán học tăng cường nền tảng nhất. Nó đã đóng vai trò trung tâm trong thành công gần đây của học tăng cường, đáng chú ý nhất là trong việc học chơi trò chơi điện tử :cite:`mnih2013playing`. Việc triển khai Q-learning không yêu cầu chúng ta biết đầy đủ Markov decision process (MDP), chẳng hạn như các hàm chuyển trạng thái và phần thưởng.

## Bài tập

1. Hãy thử tăng kích thước lưới lên $8 \times 8$. So với lưới $4 \times 4$, cần bao nhiêu vòng lặp để tìm ra hàm giá trị tối ưu?
1. Chạy lại thuật toán Q-learning với $\gamma$ (tức là "gamma" trong đoạn code trên) lần lượt bằng $0$, $0.5$ và $1$, rồi phân tích các kết quả.
1. Chạy lại thuật toán Q-learning với $\epsilon$ (tức là "epsilon" trong đoạn code trên) lần lượt bằng $0$, $0.5$ và $1$, rồi phân tích các kết quả.

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12103)
:end_tab: