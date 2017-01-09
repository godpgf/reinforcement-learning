#Reinforcement Learning

##一、提出问题

###1.介绍

增强学习（reinforcementlearning, RL）又叫做强化学习，是近年来机器学习和智能控制领域的主要方法之一。  
> Reinforcement learning is learning what to do ----how to map situations to actions ---- so as to maximize a numerical reward signal.

也就是说增强学习关注的是智能体(Agent)如何在环境中采取一系列行为，从而获得最大的累积回报。通过增强学习，一个Agent应该知道在什么状态下应该采取什么行为。RL是从环境状态到动作的映射的学习，我们把这个映射称为策略。  
可以看到，增强学习和监督学习的区别主要有以下两点：  

1.  增强学习是试错学习(Trail-and-error)，由于没有直接的指导信息，智能体要以不断与环境进行交互，通过试错的方式来获得最佳策略。  
2.  延迟回报，增强学习的指导信息很少，而且往往是在事后（最后一个状态）才给出的，这就导致了一个问题，就是获得正回报或者负回报以后，如何将回报分配给前面的状态。  

增强学习的其中一个挑战就是在直觉给出的行为（exploitation）和探索新行为（exploration）之间进行试错（trial-and-error）。就像中午选择餐厅吃饭，可以凭借直觉选择已知的好吃的餐厅，也可以探索一个新餐厅（也行更加好吃）。

> Exploitation is the right thing to do to maximize the expected reward on the one play, but exploration may produce the greater total reward in the long run. 

####主要概念:

策略(policy)：state到action的映射

> Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states.

奖励函数（reward function）：某个action得到的直接奖励信号，比如吃个雪糕感觉高兴。所谓的Reward就是Agent执行了动作与环境进行交互后，环境会发生变化，变化的好与坏就用reward来表示。

> Roughly speaking, it maps each perceived state (or state-action pair) of the environment to a single number, a reward, indicating the intrinsic desirability of that state.

值函数（value function）：评价某个state长远来说的好坏，表示一个状态未来的潜在价值期望。

> A value function specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.

###2.评价性反馈（Evaluative Feedback）

评价性反馈（Evaluative Feedback）反映了一个action直接（短期）的好坏，指导性反馈（instructive feedback）反映了一个action的准确性。督导学习就是指导性反馈，这很好区分：评价性反馈依赖已经做过的行为，指导性反馈不依赖。

> Purely evaluative feedback indicates how good the action taken is, but not whether it is the best or the worst action possible. Evaluative feedback is the basis of methods for function optimization, including evolutionary methods. Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken. 

动作值函数（Action-Value Methods）：  
前面我们引出了估值函数，考虑到每个状态之后都有多种动作可以选择，每个动作之下的状态又多不一样，我们更关心在某个状态下的不同动作的估值。显然，如果知道了每个动作的估值，那么就可以选择估值最好的一个动作去执行了。 

> 动作值函数类似于人类在某个情况下采取某种行为的直觉，它不是通过严谨的推理或想象，而仅仅凭借经验来决策的。在试错寻找最大回报策略的过程中，我们不能完全依靠直觉，但可以用直觉来剪枝。 
 
向前t步的估值表示为![qt](file:./RL/inimgtmp86.png)，在经过action a后接下来所有可选择的action的反馈奖励是![rt_all](file:./RL/inimgtmp91.png)，一种最简单的很自然的方法可以用均值评估a，写成：

![qt_formul](file:./RL/numeqtmp0.png)

> where ![rt_all](file:./RL/inimgtmp91.png) are all the rewards received following all selections of action a prior to play 

当然，这只是评估行为的一种方法，也不是最好的。不过目前可以用它来提出行为选择的问题。最简单的行为选择规则就是用贪心法选择![a_select](file:./RL/inimgtmp99.png),即每次选择评估最大值最大的行为![a_select_formul](file:./RL/inimgtmp100.png)。

每次试错时（trial-and-error）选择评估最好的action可以大大降低收索空间，但这种贪心算法效果不一定很好。提出一种![e2](file:./RL/img2.png)-greedy算法，以小概率![e2](file:./RL/img2.png)去探索新的行为（exploration）。

进一步改进，可以让每个action的选中概率正比于评估，即Softmax Action Selection。

不同的策略对应不同的动作值函数，不是每个都需要记录，只需要记录最好的那个就行。
>  It is easy to devise incremental update formulas for computing averages with small, constant computation required to process each new reward.

增量实现（Incremental Implementation）

动作值函数可以增量实现，设它的前k个奖励是![qk](file:./RL/inimgtmp151.png)(注意不要和![qt](file:./RL/inimgtmp86.png)搞混，t是第t步，k使用action a接下来的前k个反馈奖励)，可以得到：

![qk_formal](file:./RL/qk_formul.png)

形式如下：

![qk_format](file:./RL/numeqtmp3.png)  
其中![qk_format2](file:./RL/inimgtmp164.png)可以看成是误差，![qk_step](file:./RL/inimgtmp166.png)可以看成学习率。

跟踪一个不确定的环境模型（Tracking a Nonstationary Problem）：

确定的环境模型是指环境的状态空间很小（比如井字棋就那么多种排列情况）。而我们生活中多数问题的状态空间很大，甚至环境还会改变，是不确定的环境模型（比如围棋，所有排列情况太多，不是计算机能够处理的）  
上面提到的平均法适合确定的环境模型，但在不确定的环境模型中就不准确了，因为越往后的动作估值越不准确。可以用如下公式替代平均法：  

![qk_format2](file:./RL/qk_formul2.png) 

> The averaging methods discussed so far are appropriate in a stationary environment, but not if the bandit is changing over time. As noted earlier, we often encounter reinforcement learning problems that are effectively nonstationary. In such cases it makes sense to weight recent rewards more heavily than long-past ones. One of the most popular ways of doing this is to use a constant step-size parameter.

初始化动作值函数：

为了鼓励exploration，可以把某个state下直接奖励反馈大的行为对应的动作值函数初始化一个小的数值，给直接反馈小的行为对应的动作值函数初始化一个大的数值，这样在试错的时候一开始就会做出更多的尝试。

强化比较（Reinforcement Comparison）：

奖励期望越大的行为试错时能得到更多的复现机会，但我们要怎么去对比奖励期望？
>A central intuition underlying reinforcement learning is that actions followed by large rewards should be made more likely to recur, whereas actions followed by small rewards should be made less likely to recur. But how is the learner to know what constitutes a large or a small reward? 

如果像之前那样用动作值函数来对比不太好，因为它是根据策略来定义的，一开始尝试的策略比较少，所以动作值函数得到的值不是很准确。  
先定义第t次试错选择行为a的概率如下：  
![percent_a](file:./RL/numeqtmp6.png)  
其中![pt](file:./RL/inimgtmp212.png)表示对行为a的偏好(preference)。“强化比较”在这里就可以用来更新这个偏好的取值。假设某个试错策略下的到的奖励数是5，那么我们应该增加![pt](file:./RL/inimgtmp212.png)还是减小？这就需要一个参考的奖励值（reference reward）作为对比，比参考值大的就增加![pt](file:./RL/inimgtmp212.png)，小的就减小。用符号![reference_reward](file:./RL/inimgtmp218.png)表示参考的奖励值，![real_reward](file:./RL/inimgtmp217.png)表示当前奖励。得到偏好的更新公式：  
![pt_formul](file:./RL/numeqtmp7.png)  
以及参考奖励的更新公式：  
![reference_formul](file:./RL/numeqtmp8.png)  
其中![alpha_range](file:./RL/inimgtmp221.png)。

追赶法（Pursuit Methods）：

和强化比较一样，追赶法是另外一种学习的方法。不同的是，在学习（更新偏好）的过程中，它不止考虑了当前的偏好，还考虑了当前的动作值函数。试想一下，如果能在学习初期各个action被选中的概率比较平均，越到后面越按照动作值函数的指导来设置概率，概率值有一定的延迟但不断追赶着动作值函数，这就是追赶法。  
定义当前动作值函数最大的事件是：![argmax](file:./RL/inimgtmp237.png)
试错时某个事件被选中概率如下：  
![argmax_1](file:./RL/numeqtmp9.png)  
![argmax_2](file:./RL/numeqtmp10.png)  
这样就可以做到越往后，动作值函数平均越大的事件被选中的概率越大，但一开始概率每个事件被选中的概率差不多。

Evaluation VS Instruction：

如果某个state下有100个可选择的action，经过督导学习训练后得到最大奖励反馈的action是67（Instruction feedback），但是用动作值函数得到的最大奖励反馈的action是32（Evaluative Feedback），是选择这两个行为中的哪一个？

其实可以试错时随机选择，然后试错结束得到反馈后就知道选67好还是32好，给好的那个加分。最后统计谁的得分高就更加趋向于选谁。

###3.强化学习问题（The Reinforcement Learning Problem）

目标和奖励（Goals and Rewards）：

人类在做某件事的时候是有目标的，但是强化学习并没有引入目标这个概念，这是因为目标和奖励信号可以挂钩，只要给达成目标的上一个state到目标state的行为一个非常大的奖励信号，agent就会有目标（得到最大奖励就是它的目标）

回报期望（Returns）：

上文已经提到，确定的环境模型下回报期望是： 
![return1](file:./RL/numeqtmp11.png)  
其中，T是最终的步骤。agent在和环境交互的过程中，（因为步骤有限）很自然的会被分割成很多子序列（从某个state开始，然后结束，这就是一个子序列）。把每个子序列叫做一个episodic，把这种形式的交互叫做episodic tasks。

不确定的环境模型下回报期望是：  
![return2](file:./RL/numeqnarraytmp2-0-0.png)  
其中，![r_Range](file:./RL/inimgtmp296.png)。可见![lamda](file:./RL/inimgtmp295.png)越大，agent看得越远（但远的不一定准确），当等于1时退化成第一个公式。  
可见，这样的交互下T趋向于无穷大，把这种交互称为continuing tasks。

统一“episodic tasks”和“continuing tasks”：

为了方便计算，需要把episodic tasks和continuing tasks这两种交互方式统一成一种。可以这样，让episodic tasks的每个最终状态都加上一个自己到自己的行为连线，奖励是0，这样有限的episodic tasks就可以变成无限的continuing tasks。

![episodic_demo](file:./RL/imgtmp1.png)

马尔科夫性（The Markov Property）：

马尔科夫性说的是某个状态转到新状态的概率与之前所有经历过的状态没有关系，仅于当前状态有关。即： 
![mp1](file:./RL/numeqtmp13.png)  
等于：  
![mp2](file:./RL/numeqtmp14.png)

马尔科夫决策过程（Markov Decision Processes）：

由MDP的假设，如果这个世界就是MDP的，如果再加上一个假设，每个动作都是由完全的环境（比如人的每个细胞导致的精神状态，意识）决定，那么有一个初始状态，后继状态就是全部确定的。当然，现实情况环境一般不完全可观察，然后有一些随机性stochastic是人类无法确定的。绝大多数的增强学习都可以模型化为MDP的问题。  

如果行为和状态空间是有限的，就叫做有限马尔科夫过程（finite MDP）。

> If the state and action spaces are finite, then it is called a finite Markov decision process (finite MDP). 

设状态![s](file:./RL/inimgtmp357.png)经过行为![a](file:./RL/inimgtmp361.png)转换到![s,](file:./RL/inimgtmp362.png)的概率：  

![pssa](file:./RL/numeqtmp15.png)

得到的奖励期望是：

![rssa](file:./RL/numeqtmp16.png)

值函数（Value Functions）：

值函数用来评估一个状态的好坏。设一个策略是![pi](file:./RL/inimgtmp411.png)，它是状态![state](file:./RL/inimgtmp412.png)到行为![action](file:./RL/inimgtmp413.png)的映射。某状态下采取某行为的概率是![psa](file:./RL/inimgtmp414.png)，当前策略下的值函数![vpi](file:./RL/inimgtmp419.png)可以表示为：  
![vpi_formul](file:./RL/numeqtmp17.png)

类似的，动作值函数可以表示为：  
![qsa_formul](file:./RL/numeqtmp18.png)  
表示是当前策略下（某个状态的）某个行为到评估值的映射。

可以把这个方程写成“动态规划方程”的形式（Bellman equation）：

![v_bellman](file:./RL/v_bellman.png)

值函数优化：

强化学习就是找到一个策略让（长远来说）得到的奖励最大化。前文已经说了某个策略下的值函数怎么求，那么对于所有策略![pi_all](file:./RL/inimgtmp512.png)呢？

对于估值函数：  
![v_all_formul](file:./RL/numeqtmp19.png)

对于动作值函数：  
![q_all_formul](file:./RL/numeqtmp20.png)

结合起来：  
![q_all_formul2](file:./RL/numeqtmp21.png)  
其中![s_](file:./RL/inimgtmp362.png)表示s经过a有可能转移到的所有新状态（当然一般可能只有一个，所以不要被累加符号吓到）。

因为对所有策略的估值函数等于产生奖励最大的那个行为的动作值函数，所以可以写成：  
![v_formul2](file:./RL/v_formul2.png)

写成贝尔曼方程(动态规划方程)：  
![q_bellman](file:./RL/numeqtmp21.png)

##二、基本的实现算法

###1.动态规划法(dynamic programming methods)

####策略评估（Policy Evaluation）：
首先，先算出在某个策略![pi](file:./RL/inimgtmp411.png)下的所有值函数。按照之前给出的值函数bellman方程：  
![vbellman](file:./RL/vbellman.png)  
有如下算法：  
![pseudotmp0](file:./RL/pseudotmp0.png)

####策略改进（Policy Improvement）：  
计算值函数的一个原因是为了找到更好的策略。在状态s下，是否存在一个更好的行为![bettle_a](file:./RL/inimgtmp667.png)？要想判断行为的好坏，我们就需要计算动作值函数。  
![qpi](file:./RL/qpi.png)
已知估值函数V是评估某个状态的回报期望，动作值函数是评估某个状态下使用行为a的回报期望。所以，如果在新的策略下![new_pi](file:./RL/inimgtmp689.png)有![Q>V](file:./RL/inimgtmp691.png)就说明新的策略比当前的好。这里的不等式等价于:  
![v>v](file:./RL/numeqtmp23.png)  
有了策略改进定理，我们可以遍历所有状态和所有可能的动作，并采用贪心策略来获得新策略：  
![best_pi](file:./RL/bestpi.png)  
我们不止要更新策略，还需要更新估值函数：  
![v_new](file:./RL/v_new.png)

####策略迭代（Policy Iteration）：  
当有了一个策略![pi](file:./RL/inimgtmp411.png)，通过策略评估得到![v_pi](file:./RL/inimgtmp729.png)，再通过策略改进得到新的策略![pi_2](file:./RL/inimgtmp730.png)并可以计算出新的估值函数![v_2](file:./RL/inimgtmp731.png)，再次通过策略改进得到![pi_3](file:./RL/inimgtmp732.png)...

![pi_2_v](file:./RL/imgtmp35.png)

这样通过不断的迭代，就会收敛到一个很好的策略。

![pseudotmp1](file:./RL/pseudotmp1.png)

####值迭代（Value Iteration）：

策略迭代需要遍历所有的状态若干次，其中巨大的计算量直接影响了策略迭代算法的效率。我们必须要获得精确的![v_pi](file:./RL/inimgtmp729.png)值吗？事实上不必，有几种方法可以在保证算法收敛的情况下，缩短策略估计的过程。  
值函数的bellman方程如下：  
![v_k1](file:./RL/v_k1.png)  
值迭代算法直接用下一步![s_](file:./RL/inimgtmp362.png)的估值函数来更新当前s的估值函数，不断迭代，最后通过估值函数来得到最优策略。算法如下：  
![pseudotmp2](file:./RL/pseudotmp2.png)

####异步动态规划（Asynchronous Dynamic Programming）：  
对于值迭代，在s很多的情况下，我们可以多开一些线程，同时并行的更新多个s以提高效率；对于策略迭代，我们也没必要等待一个策略评估完成再开始下一个策略评估，可以多线程的更新。

####总结：
可以看到，动态规划去求解强化学习问题需要遍历所有状态，对于不确定的环境模型（比如围棋的排列组合方式太多，不可能遍历每个状态）并不适用。

###2.蒙特卡罗方法(Monte Carlo methods)

####基本思想：
蒙特卡洛的思想很简单，就是反复测试求平均。

一个简单的例子可以解释蒙特卡罗方法，假设我们需要计算一个不规则图形的面积，那么图形的不规则程度和分析性计算（比如积分）的复杂程度是成正比的。而采用蒙特卡罗方法是怎么计算的呢？首先你把图形放到一个已知面积的方框内，然后假想你有一些豆子，把豆子均匀地朝这个方框内撒，散好后数这个图形之中有多少颗豆子，再根据图形内外豆子的比例来计算面积。当你的豆子越小，撒的越多的时候，结果就越精确。

需要注意的是，我们仅仅将蒙特卡洛方法定义在episodic tasks上（就是指不管采取哪种策略都会在有限时间内到达终止状态并获得回报的任务）。

####蒙特卡洛策略评估（Monte Carlo Policy Evaluation）：

首先考虑用蒙特卡洛方法学习估值函数，方法很简单，就是先初始化一个策略，在这个策略下随机产生很多行为序列，针对每个行为序列计算出对应估值函数的取值，这样某个状态下估值函数针对不同的行为序列就得到不同的估值，只要再一平均就可以算出最后的估值。  
![pseudotmp3](file:./RL/pseudotmp3.png)

####蒙特卡洛动作函数评估（Monte Carlo Estimation of Action Values）：

为了像动态规划那样做策略改进，必须得到动作函数。这和前文类似，某个状态下动作是a，然后再随机尝试很多行为序列，得到反馈奖励后计算平均值（不是最大值，可能是考虑在不确定的环境模型下，使用平均最优更加靠谱）就是当前策略的动作函数。

####蒙特卡洛控制（Monte Carlo Control）

即蒙特卡洛版本的策略迭代：生成动作函数，改进策略，循环前两步。  
![imgtmp6](file:./RL/imgtmp6.png)  
过程如下：  
![imgtmp36](file:./RL/imgtmp36.png)  
具体到MC control，就是在每个episode后都重新估计下动作值函数（尽管不是真实值），然后根据近似的动作值函数，进行策略更新。这是一个episode by episode的过程。  
![pseudotmp4](file:./RL/pseudotmp4.png)

####在策略MC（On-Policy Monte Carlo Control）：


为了更好的探测状态空间避免陷入局部最优解，加入![e2](file:./RL/img2.png)-greedy算法：  
![20160512105522475](file:./RL/20160512105522475.png)  
训练越长时间![e2](file:./RL/img2.png)越小，可以提高收敛速度。

####离策略MC（Off-Policy Monte Carlo Control）：

为了让更加接近结果的步骤得到更多的调整机会，而不是一个episode中每一步都有一样多的调整机会，于是提出了Off-Policy方法。这就类似人类下棋，一开始变数太大没有必要花太多精力去学习（调整动作值函数），但越接近结果时越需要更准确的行为估值（需要花费更多的运算去调整动作值函数以便更加准确）。  
在off-policy中，使用两个策略。一个策略用来生成行为，叫做行为策略（behavior policy）；另外一个策略用来（从终点向起点）评估行为序列准确性的，叫做评估策略（estimation policy）。当第一次发现这两个策略指示的行为出现不同时，开始更新动作值函数，可见越靠近终点的得到调整的概率越大。算法如下：  
![pseudotmp6](file:./RL/pseudotmp6.png)  
其中，行为策略![inimgtmp908](file:./RL/inimgtmp908.png)是一种soft policy，评估策略![inimgtmp909](file:./RL/inimgtmp909.png)是一种greedy policy。

增量实现（Incremental Implementation）：

在第一章第二节已经提出过动作值函数的增量实现，这里就可以用上了。以值函数为例的增量实现如下：  
![numeqtmp27](file:./RL/numeqtmp27.png)  
其中：  
![imgtmp39](file:./RL/imgtmp39.png)，![inimgtmp925](file:./RL/inimgtmp925.png)


###3.时间差分法(temporal difference)

####基本思想：

动态规划优点：  
动态规划使用估值函数，已知动态规划在值迭代的时候可以直接用下一个状态的值更新当前状态的值；蒙特卡洛方法使用动作值函数，它只有在等待随机生成的episode所有行为执行完后，才能从后向前更新动作值函数。而且，蒙特卡洛法局限于episode task，不可以用于连续的任务。

蒙特卡洛优点：  
动态规划在值迭代的时候需要遍历所有state，这就需要一个确定的环境模型（state不能太多），但现实往往是state的数量特别多；蒙特卡洛方法可以从经验中学习不需要环境模型。

时序差分法可以看成是动态规划和蒙特卡洛方法的结合。和蒙特卡洛方法一样，它不需要遍历所有state；而它又与动态规划相似，可以基于对其他状态的估计来更新对当前状态估值函数的估计，不用等待最后的结果。  


>If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap). The relationship between TD, DP, and Monte Carlo methods is a recurring theme in the theory of reinforcement learning. 

####TD prediction算法：

TD在更新当前状态函数时可以像动态规划那样，使用下一个已经存在的状态v(t+1)和真实的立即反馈rt来更新当前状态v(t)，把这种更新方式叫做bootstrapping方法。由于我们没有状态转移概率，所以要利用多次实验来得到期望状态值函数估值。类似MC方法，在足够多的实验后，状态值函数的估计是能够收敛于真实值的。由于不必等待所有的state都走完就可以更新当前state，所以TD不局限于episode task，可以用于连续的任务。

回忆蒙特卡洛方法值函数更新策略：  
![numeqtmp28](file:./RL/numeqtmp28.png)  
其中![inimgtmp936](file:./RL/inimgtmp936.png)是每个episode结束后获得的实际累积回报.这个式子的直观的理解就是用实际累积回报作为状态值函数的估计值。  
现在我们将公式修改就得到TD(0)的值函数更新公式：  
![numeqtmp29](file:./RL/numeqtmp29.png)  
为什么修改成这种形式呢，我们回忆一下状态值函数的定义：  
![mcupdate](file:./RL/mcupdate.png)  
很容易发现，利用真实的立即回报和下个状态的值函数来更新当前值函数，这种方式就称为时序差分。

策略评估算法如下：

![pseudotmp7](file:./RL/pseudotmp7.png)

举例说明：  
假设我们有这么一条路径：A-->B-->end，随机生成8个episode:(A:0-->B:0)、（B:1）、（B:1）、（B:1）、（B:1）、（B:1）、（B:1）、（B:0）。其中A:0表示经过A得到的直接奖励是0。

蒙特卡洛法求解：  
V(A)=ra+a\*rb=0\+a\*0=0  
V(B)=average(0+1+1+1+1+1+1+0)=0.75

TD法求解：  
V(A)=a\*0.75  
V(B)=0.75

####TD算法优化：

假设只能得到少量的经验（比如10个episodes），很自然的办法就是反复使用它们直到状态值函数收敛。每次使用这些经验都等待全部episodes增量计算完成，再去更新状态值函数。把这种方式叫做批量更新（batch updating）。它可以提高收敛速度，因为如果每次尝试一个episodes得到增量都去更新状态值函数，抖动必然特别严重，因为下一个episode计算出来的增量是受到当前状态取值的影响的。

####Sarsa算法（On-Policy TD Control）：

现在我们利用TD prediction组成新的强化学习算法，用到决策/控制问题中。在这里，强化学习算法可以分为在策略(on-policy)和离策略(off-policy)两类。首先要介绍的sarsa算法属于on-policy算法。  
与前面DP方法稍微有些区别的是，sarsa算法估计的是动作值函数(Q函数)而非状态值函数。也就是说，我们估计的是策略![pi](file:./RL/inimgtmp411.png)下，任意状态![s](file:./RL/inimgtmp357.png)上所有可执行的动作a的动作值函数![inimgtmp1010](file:./RL/inimgtmp1010.png)，Q函数同样可以利用TD Prediction算法估计。可以把下图的黑点当做一个state：  
![imgtmp8](file:./RL/imgtmp8.png)  
给出sarsa的动作值函数更新公式如下：  
![numeqtmp30](file:./RL/numeqtmp30.png)  
由于算法每次更新都与s(t)、a(t)、r(t+1)、s(t+1)、a(t+1)有关，所以叫做sarsa算法。完整的流程如下：  
![pseudotmp8](file:./RL/pseudotmp8.png)

####Q-Learning算法（Off-Policy TD Control）：

在sarsa算法中，选择动作时遵循的策略和更新动作值函数时遵循的策略是相同的，即ϵ−greedy的策略，而在接下来介绍的Q-learning中，动作值函数更新则不同于选取动作时遵循的策略，这种方式称为离策略(Off-Policy)。Q-learning的动作值函数更新公式如下：

![numeqtmp31](file:./RL/numeqtmp31.png)

可以看到，Q-learning与sarsa算法最大的不同在于更新Q值的时候，直接使用了下一个最大的Q值——相当于采用了Q(st+1,a)值最大的动作，并且与当前执行的策略，即选取动作at时采用的策略无关。 Off-Policy方式简化了证明算法分析和收敛性证明的难度，使得它的收敛性很早就得到了证明。Q-learning的完整流程图如下：

![pseudotmp9](file:./RL/pseudotmp9.png)

####Actor-Critic算法：

回忆第一章提出的“强化比较（Reinforcement Comparison）”，策略选择和动作值函数一点关系都没有，完全是一个数据结构去记录动作值函数，另外一个全新的数据结构去记录选择某个试错行为的概率。Actor-Critic算法就是在TD算法的基础上扩展出来的产生试错行为时不依赖于值函数的一种算法。如下图：

![figtmp34](file:./RL/figtmp34.png)

其中Actor是一个产生试错行为的概率函数，而标准的Critic就是一个用来评价行为好坏的值函数（注意不是Q函数）。在值函数更新的时候，会有一个误差值扔给Actor去更新概率函数，这个误差值叫做TD error，公式如下：

![imgtmp41](file:./RL/imgtmp41.png)

假设Actor用softmax算法（Gibbs softmax method）产生试错行为，公式如下：

![imgtmp42](file:./RL/imgtmp42.png)

这里面的概率函数就可以用TD error来更新。方法如下：

![imgtmp43](file:./RL/imgtmp43.png)

其中![inimgtmp1047](file:./RL/inimgtmp1047.png)是学习率，可以不变，也可以一开始很大后来变小。  
当然也可以用其他版本的概率函数更新方式，比如想让概率越小的行为对应的概率函数更新越快，可以用如下公式：

![imgtmp44](file:./RL/imgtmp44.png)

更多的优化会在未来章节提出。

####R-Learning算法（解决Undiscounted Continuing Tasks）：

R-Learning是为了处理那种无限步骤的问题，这类问题下试错经验不能被划分成episodes并得到最终的回报。

>R-learning is an off-policy control method for the advanced version of the reinforcement learning problem in which one neither discounts nor divides experience into distinct episodes with finite returns. 

这种情况下，某个策略![inimgtmp411](file:./RL/inimgtmp411.png)下的值函数和长时间下的平均回报![inimgtmp1051](file:./RL/inimgtmp1051.png)有关：

![imgtmp45](file:./RL/imgtmp45.png)

经过长期运行以后，平均回报![inimgtmp1051](file:./RL/inimgtmp1051.png)会趋向于一个固定的值，把于平均回报的差值作为真实的立即回报。

![imgtmp46](file:./RL/imgtmp46.png)

类似的动作值函数是：

![imgtmp47](file:./RL/imgtmp47.png)

把这个些值函数称为相对值因为都是相对于策略下的平均回报的。
>We call these relative values because they are relative to the average reward under the current policy.

让然不仅限于相对值，R-Learning也是基于off-policy的，它会同时维护评估策略和（试错）行为策略。具体算法如下：

![pseudotmp10](file:./RL/pseudotmp10.png)

##三、学习方法的融合

动态规划、蒙特卡洛、时序差这三套算法并不是独立的，很多时候可以联合起来使用会使学习效率大大提升，这就是这一章节的目的所在。

####贡献度

贡献度在所有TD算法中都可以使用，包括Q-learning和Sarsa。

#####n步TD predict算法（n-Step TD Prediction）

在蒙特卡洛方法中，agent是用一个完整的学习阶段获得整个样本序列来估计价值函数的，考虑的是直到终止状态多不以后的反馈。从这一点说，它关 心的是多步以后的反馈之和，是deepbackups，而时序差方法中，agent是基于立 即奖励和后继状态的价值函数的。考虑的是一步之后的反馈，是shallow backups。因此可以把蒙特卡罗方法和时序差方法的差别仅仅看作反馈的深 浅程度不同。  
最容易想到的蒙特卡罗方法和时序差方法的折中就是：考虑n步的反馈，n 介于1到终止步数之间。这种方法不用等到最后一步就开始对价值函数的更新， 故仍可看作时序差方法，称为n步时序差方法，显然，我们以前介绍的时序差方法，现在该改称l步TD方法了；而当n趋向无穷大时，就是蒙特卡罗方法。

![figtmp36](file:./RL/figtmp36.png)

回忆蒙特卡洛方法，某个值函数![inimgtmp1080](file:./RL/inimgtmp1080.png)使用某个策略计算出的值![inimgtmp1081](file:./RL/inimgtmp1081.png)来更新：  
![imgtmp48](file:./RL/imgtmp48.png)
之前的TD算法奖励期望如下：  
![imgtmp49](file:./RL/imgtmp49.png)  
2步TD算法奖励期望如下：
![imgtmp50](file:./RL/imgtmp50.png)  
n步TD算法奖励期望如下：  
![numeqtmp32](file:./RL/numeqtmp32.png)

#####前向n步TD算法（The Forward View of TD(![img1](file:./RL/img1.png))）:

之前提到的n步TD算法就是前向的，它会向前看并得到反馈。这里对它进行一下扩展。  
对于某个奖励期望，我们可以混合多种步长的反馈来得到，比如混合2步和4步：![inimgtmp1127](file:./RL/inimgtmp1127.png)，限制就是每个混合元素的权重和必须是1。把这种方式叫做混合反馈。混合2步和4步的权重都是0.5，如下图：  
![imgtmp10](file:./RL/imgtmp10.png)  
下图是另外一种混合方式（每个元素的权重展示）：  
![figtmp38](file:./RL/figtmp38.png)  
奖励期望的计算公式：  
![imgtmp52](file:./RL/imgtmp52.png)  
于是定义![img1](file:./RL/img1.png)-return算法，每次更新的增量：  
![numeqtmp35](file:./RL/numeqtmp35.png)  
这就像人一样，越远的东西看得越不清楚，如下图展示：  
![figtmp40](file:./RL/figtmp40.png)  
当然，得更具具体的需求来选择使用具体的方式，千万不能认死理。

#####反向n步TD算法（The Backward View of TD(![img1](file:./RL/img1.png))）：

反向n步TD算法从概念上和计算上更加简单。前向n步TD算法更新时使用了未来将要发生的值，这些值只是一个估计，肯定是不准确的。  
>In particular, the forward view itself is not directly implementable because it is acausal, using at each step knowledge of what will happen many steps later. 

反向n步TD算法在这方面做了改进，在off-line的情况下更加准确。

反向n步TD算法会为每个访问过的状态![inimgtmp357](file:./RL/inimgtmp357.png)分配一个新的内存空间来保存贡献度（eligibility trace），当循环计算到下一个状态![inimgtmp362](file:./RL/inimgtmp362.png)时，之前所有状态的贡献度都会减小，然后在更具当前状态的TD error来更新（包括当前状态在内的）之前所有状态的值函数，更新方法当然是TD error乘以贡献度。而前向的n步TD算法循环计算到下一个状态时，前面计算过的状态不会再去更新了。

贡献度的更新公式：  
![numeqtmp36](file:./RL/numeqtmp36.png)  
其中，![inimgtmp1154](file:./RL/inimgtmp1154.png)是折扣率，![inimgtmp1155](file:./RL/inimgtmp1155.png)是混合权重参数（trace-decay）。把这种贡献度更新公式叫做累积贡献（accumulating trace）因为状态每次被访问时贡献度会累积（甚至大于1），但没被访问时会下降。
![imgtmp15](file:./RL/imgtmp15.png)
某个状态下的TD error计算公式：  
![numeqtmp37](file:./RL/numeqtmp37.png)  
某个状态下的更新值公式：  
![numeqtmp38](file:./RL/numeqtmp38.png)  
具体的算法流程：  
![pseudotmp11](file:./RL/pseudotmp11.png)  
更具结果来更新之前状态的理解如下：  
![figtmp42](file:./RL/figtmp42.png)

#####前向和反向的等价（Equivalence of Forward and Backward Views）：

按理说，前向和反向算法的区别仅在于TD error按照不同的比例分配给各个state，所以它们的累加值应该相同。具体证明参见[sutton book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node76.html)

#####Sarsa(![img1](file:./RL/img1.png))算法：

很简单，仅仅把值函数变成动作值函数，类似的更新公式如下：  
![imgtmp56](file:./RL/imgtmp56.png)  
其中：  
![imgtmp57](file:./RL/imgtmp57.png)  
以及  
![numeqtmp40](file:./RL/numeqtmp40.png)  
如图：  
![figtmp44](file:./RL/figtmp44.png)  
具体的算法如下：  
![pseudotmp12](file:./RL/pseudotmp12.png)  

#####Q(![img1](file:./RL/img1.png))算法：

把贡献度和Q-learning算法结合起来，有两种不同的方法：Watkins's Q(![img1](file:./RL/img1.png))、Peng's Q(![img1](file:./RL/img1.png))，先描述前者。

由于Q-learning更新当前动作值函数时，是使用下一个取值最大的动作值函数，要使用贡献度就得想一种特殊的方式。  
假设在第t步反馈动作值函数的奖励，再假设选择行为时agent在后两步按照贪心法选择的（即后两步就是反馈最大奖励的行为），而在后三步是随机产生的（即后三步不是反馈最大奖励的行为），我们可以只跟踪到最后一个反馈最大奖励的行为那里，后面的（从后三步开始的）就不再跟踪而是直接遍历所有可选的行为找到一个反馈奖励最大的作为终点。  

> Thus, unlike TD(![img1](file:./RL/img1.png)) or Sarsa(![img1](file:./RL/img1.png)), Watkins's Q(![img1](file:./RL/img1.png)) does not look ahead all the way to the end of the episode in its backup. It only looks ahead as far as the next exploratory action.

设![inimgtmp1218](file:./RL/inimgtmp1218.png)是第一个随机选出的非最大奖励的行为（exploratory action），反馈奖励公式如下：  
![imgtmp58](file:./RL/imgtmp58.png)  
反馈奖励流程如图：  
![figtmp46](file:./RL/figtmp46.png)  

第t步的贡献度计算如下：  
![imgtmp59](file:./RL/imgtmp59.png)  
其中![inimgtmp1222](file:./RL/inimgtmp1222.png)是指示函数，表示当![inimgtmp1224](file:./RL/inimgtmp1224.png)时是0，其他情况是1。

动作值函数更新公式如下：  
![imgtmp60](file:./RL/imgtmp60.png)  
其中  
![imgtmp61](file:./RL/imgtmp61.png)

具体算法流程如下：  
![pseudotmp13](file:./RL/pseudotmp13.png)

以上算法有一个小缺陷，就是当exploratory action被频繁的选中时，算法的效果就非常接近于Q(0)，失去了对未来一定时间内奖励的平均这个功能。于是Peng's Q(![img1](file:./RL/img1.png))作为另外一个版本出现了，可以把它看成是Sarsa(![img1](file:./RL/img1.png))和Watkins's Q(![img1](file:./RL/img1.png))的结合。  
![figtmp47](file:./RL/figtmp47.png)  
如图，Peng's Q(![img1](file:./RL/img1.png))仅仅在最后一步使用最大的反馈奖励，前面的步骤试错行为的产生和更新都是用同样的策略，所以它即不是on-policy也不是off-policy。简单的说，就是在前面的步骤中都使用on-policy仅在最后一步使用off-policy（即遍历所有可选行为找到最大累积反馈那个来更新）

#####在Actor-Critic算法中使用贡献度：

回忆actor的更新公式：  
![imgtmp62](file:./RL/imgtmp62.png)  
改进一下就可以变成：  
![numeqtmp41](file:./RL/numeqtmp41.png)

或者用另外一个版本：  
![imgtmp63](file:./RL/imgtmp63.png)  
在这个版本中，贡献度的更新方式如下：  
![numeqtmp42](file:./RL/numeqtmp42.png)

#####替换贡献度（Replacing Traces）：

在更新贡献度时，如果某个状态被反复访问，这样它的贡献度因为累积会大于1，为了解决这个问题，可以把贡献度的更新公式改改：  
![numeqtmp43](file:./RL/numeqtmp43.png)  
效果如图：  
![figtmp48](file:./RL/figtmp48.png)  
于之前的累积贡献度有轻微的不同，把这种方法叫做替换贡献度算法。

用这种方法改进Sarsa(![img1](file:./RL/img1.png))的贡献度更新，如下：  
![numeqtmp44](file:./RL/numeqtmp44.png)

####生成试和函数试的近似

在前文的描述中，我们必须记录每一个state或者每对state-action的值函数，如果state或state-action数量巨大，就必须找一种近似算法来实现。在这一章节，我将不再按照sutton book的结构，而是结合自己的想法以及深度学习来完成。

#####用长短记忆法记录值函数（我的想法）

可以给当前访问过的每个state或者state-action分配一个内存表示记忆，用c(state)表示记忆的强度。影响记忆强度的方式有几个：  
1、访问时间：长时间不访问的记忆强度会减弱  
2、值函数的大小：值函数绝对值特别大的表示印象很深刻，记忆强度也会很大，越接近0的表示越小。  
3、访问次数：访问次数多的证明这个state经常要去使用，当然记忆强度应该更大。

每次删掉记忆强度小的内存空间，把这种方法叫做“忘记”，这样就可以解决state数量巨大的问题。

#####归纳总结行为序列和state特征。

某些常用的行为序列可以归纳总结出来，生成一个更高级别的行为，即anew=(a1,a2,...,an)。比如说，走路先迈出左脚再迈出右脚，于是得到：a走路=(a迈出左脚,a迈出右脚)，这样就可以只记录一个高级的行为。

可以想办法总结出state特征，用抽象的特征来表示真实的行为，减少数量。一个状态下可以划分出很多维度的特征，其中的一些特征或者特征组合又可以划分更加抽象的特征。在某个特征出现时，使用某个行为如果能得到很好的反馈就需要被重点记录。此时应该做的改进是将Q(s,a)变成Q(f,a)，其中f表示特征。

举例：比如一个agent想从柳州去鞍山。

传统的增强学习：得到的行为序列是（a前,a左,a前,a右,.........）

归纳总结后的增强学习(a去北京,a去鞍山)

#####用深度学习

参见Deep Q-Learning。  
1、用一个深度神经网络来作为Q值的网络，参数为w：  
![qw](file:./RL/qw.png)  
2、在Q值中使用均方差mean-square error 来定义目标函数objective function也就是loss function  
![lw](file:./RL/lw.png)  
可以看到，这里就是使用了Q-Learning要更新的Q值作为目标值。有了目标值，又有当前值，那么偏差就能通过均方差来进行计算。  
3、计算参数w关于loss function的梯度，这个可以直接计算得到  
![ldw](file:./RL/ldw.png)  
4、使用SGD实现End-to-end的优化目标。有了上面的梯度，而![qdw](file:./RL/qdw.png)可以从深度神经网络中进行计算，因此，就可以使用SGD 随机梯度下降来更新参数，从而得到最优的Q值。

####计划和学习（Planning and Learning）：

动态规划方法更加预知的环境模型来决定策略，它知道环境下所有状态的取值，这可以称为计划（Planning）的方法。而蒙特卡洛和时序差方法是完全的重交互中学习，它不知道每个状态的取值，这可以称为学习（Learning）的方法。那么能不能agent在交互中即对动作值函数进行更新，同时又更具自己的经历建立环境模型呢？

#####模型和计划

所谓的模型，就是agent可以用来对某一行为的结果进行预测的工具。模型可以分为两类：一类描述了各种可能的立即反馈及转移状态，以及它们各自发生的概率，称为分布模型（distribution models）；另一种则只给出依据概率取样得到的一种可能，我们称为取样模型（sample models）。在前面介绍的动态规划方法中，使用的是分布模型。分布模型远比取样模型难以获得，而且增强学习的环境是动态的，预知的分布模型不能跟上环境的变化。这是限制动态规划方法的重要原因。  
实际上，agent在和环境交互过程中，根据经历的状态、获得的反馈等，可以积累对环境的了解，从而构成取样模型。另外，agent会不断地获得新的信息， 这可能会引起对模型的修改，而使之更符合实际情况的变化。因此，agent通过交互获得的经验，至少有两个用处：直接用于改进价值函数和策略，以及用于构造和改进模型。我们以前介绍的各种方法都是为了第一个用处，可以称为直接的增强学习（direct RL）。后者称为模型的学习（model learning）。经验改变模型来间接地改进价值函数和策略，这称为间接的增强学 习（indirect RL），它也是一种“计划”的学习。

#####整合计划、行为和学习

计划学习其实没那么难，当使用Q-Learning去学习时，更新用的下一个状态的取值是估计出来的，它并不准确。如果把当前的s和a记录下来，过一段时间后再去对它更新一次（因为过一段时间后下一个状态的估计肯定比之前更加准确是吧），这就是planning的过程，其实就是用来帮助更新值函数让其更加准确的。如图：  
![figtmp63](file:./RL/figtmp63.png)  
于是提出Dyna-Q算法，交互过程如下：  
![Dyna-Q](file:./RL/figtmp64.png)  
具体的算法流程如下：  
![pseudotmp18](file:./RL/pseudotmp18.png)

##自己的想法
有很多强化学习和督导学习可以结合的地方，比如如何提取有用的输入信号（注意力放在哪），如何将第一次接触的信号不训练直接分类后放到合适的模型中去训练（分区保存数据，比如第一次看到两个轮子的东西知道它是车），如何提取数据间的共性并仅记忆共性（特征提取）  
