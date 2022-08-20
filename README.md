# EM算法的直觉性数学推导

思考如下问题：离散随机变量$Z$有$k$种取值，分别对应m种随机变量$X$的分布，$Z$和$X$的分布的所有参数都未知，记为$\theta$，目的是估计出这些参数。

很自然地，我们想到了用极大似然法（ML）估计这些参数。于是我们抽取$n$个样本，每次抽样的方式为：先抽一个$z$，再从$z$对应的$X$的分布里抽出一个$x$。于是，我们获得了$n$组样本$(z_1, x_1)$, $(z_2, x_2)$, ..., $(z_n, x_n)$。可以写出样本的对数似然函数

$$
\begin{aligned}
    L(\theta) &= \log{\prod_{i=1}^{n}{P(x_i, z_i; \theta)}} \\
    &= \sum_{i=1}^n{\log{P(x_i, z_i; \theta)}} \\
\end{aligned}
$$

显然，我们可以很轻松地估计出$Z$的分布的参数和各种$X$的分布的参数。但问题是，如果我们只能看到抽样结果中的$x$（$Z$为隐变量），是否还可以用ML估计出全部参数$\theta$。我们继续尝试ML，写出样本的对数似然函数

$$
\begin{aligned}
    L(\theta) &= \log{\prod_{i=1}^{n}{P(x_i; \theta)}} \\
    &= \sum_{i=1}^n{\log{P(x_i; \theta)}} \\
    &= \sum_{i=1}^n{\log{\sum_{j=1}^{k}{P(z_j; \theta)P(x_i|z_j; \theta)}}} \\
    &= \sum_{i=1}^n{\log{\sum_{j=1}^{k}{P(x_i,z_j; \theta)}}}
\end{aligned}
$$

问题出现了，我们之前能看到$z_i$，所以知道$x_i$是从哪个分布中取出来的，但是现在我们不知道$z$的取值，所以只能用全概率公式求取到$x_i$的概率。对比上式，全概率公式直接导致了$\log$里出现了求和，为计算似然函数对参数的偏导带来了极大困难，ML不好用了。

但聪明的Dempster等（1977）想到了可以用Jenson不等式，利用$\log$凹函数的特性把求和从$\log$里提出来：

$$
\begin{aligned}
    L(\theta) &= \sum_{i=1}^n{\log{\sum_{j=1}^{k}{P(x_i,z_j; \theta)}}} \\
    &= \sum_{i=1}^n{\log{\sum_{j=1}^{k}{W_j(\cdot)\frac{P(x_i,z_j; \theta)}{W_j(\cdot)}}}}  \\
    &\geq \sum_{i=1}^n{\sum_{j=1}^{k}{W_j(\cdot)}\log{\frac{P(x_i,z_j; \theta)}{W_j(\cdot)}}} = J(\theta)
\end{aligned}
$$

其中，$W_j(\cdot)$满足$\sum_{j=1}^{k}{W_j(\cdot)} = 1$。

这样一来，ML似乎又可行了起来。不过，我们还需要注意：现在只能对$J(\theta)$进行极大化，如果$J(\theta)$和$L(\theta)$不能取等，则对$J(\theta)$极大化的意义也不大。于是，构造$W_j(\cdot)$使得Jensen不等式取等成为了一个努力方向。已知Jensen不等式的在函数方面的取等条件是函数为常值函数，考虑到

$$
\begin{aligned}
    P(x_i; \theta) = \frac{P(x_i,z_j; \theta)}{P(z_j|x_i; \theta)}
\end{aligned}
$$

在$i$固定时为常数，于是令$W_j(\cdot) = P(z_j|x_i; \theta)$，此时$J(\theta) \equiv L(\theta)$。但是我们仍需注意：$W_j(\cdot)$此时含有参数$\theta$，与含参的$\log$有乘积关系，求偏导仍有困难。所以我们只能让$W_j(\cdot)$在$i$和$j$都固定时成为一个常数而非函数，那么就需要预先设置一组参数$\theta^{(0)}$，令$W_j(\cdot) = P(z_j|x_i; \theta^{(0)})$，然后对$J(\theta, \theta^{(0)})$进行极大化，得到一组新的参数估计$\theta^{(1)}$，再令$W_j(\cdot) = P(z_j|x_i; \theta^{(1)})$...不断迭代，直到参数估计收敛，即

$$
\begin{aligned}
    \theta^{(t+1)} &= \underset{\theta}{\arg\max} \,J(\theta, \theta^{(t)}) \\
    &= \underset{\theta}{\arg\max} \, \sum_{i=1}^n{\sum_{j=1}^{k}{P(z_j|x_i; \theta^{(t)})}\log{\frac{P(x_i,z_j; \theta)}{P(z_j|x_i; \theta^{(t)})}}} \\
    &= \underset{\theta}{\arg\max} \, \sum_{i=1}^n{\sum_{j=1}^{k}{P(z_j|x_i; \theta^{(t)})}\log{P(x_i,z_j; \theta)}} \\
    &= \underset{\theta}{\arg\max} \, Q(\theta, \theta^{(t)}) \\
\end{aligned}
$$

于是，EM算法就此诞生：1）先写出$Q$函数，由于$Q$函数可以理解为对数似然函数$\log{P(X, Z; \theta)}$在给定观测数据$X$和当前参数$\theta^{(t)}$对未观测数据$Z$的条件概率分布$P(Z|X; \theta^{(t)})$的期望，于是这一步也叫“E步”（Expectation step）；2）求出使得$Q$函数达到极大值的参数估计$\theta^{(t+1)}$，由于这一步是做极大化，于是也叫“M步”（Maximization step）。两步不断迭代，直到参数估计收敛。另外，EM算法也可理解为，在迭代的过程中不断提高似然函数$L(\theta)$的下界，直至$L(\theta)$收敛到局部最大。

现在只剩最后一个问题：为什么参数估计一定会收敛？因为$L(\theta)$是一个概率，有极大值为1，故只需证明$L(\theta^{(t+1)}) \geq L(\theta^{(t)})$：

因为

$$
\begin{aligned}
    L(\theta^{(t)}) &= \sum_{i=1}^n{\log{\sum_{j=1}^{k}{P(x_i,z_j; \theta^{(t)})}}} \\
    &= \sum_{i=1}^n{\log{\sum_{j=1}^{k}{P(z_j|x_i; \theta^{(t)})\frac{P(x_i,z_j; \theta^{(t)})}{P(z_j|x_i; \theta^{(t)})}}}}  \\
    &\overset{取等}{=} \sum_{i=1}^n{\sum_{j=1}^{k}{P(z_j|x_i; \theta^{(t)})}\log{\frac{P(x_i,z_j; \theta^{(t)})}{P(z_j|x_i; \theta^{(t)})}}} \\
    &= J(\theta^{(t)}, \theta^{(t)})
\end{aligned}
$$

同时，

$$
\begin{aligned}
    L(\theta^{(t+1)}) &\geq J(\theta^{(t+1)}, \theta^{(t)}) \\
    &\geq J(\theta^{(t)}, \theta^{(t)}) \\
    &= L(\theta^{(t)})
\end{aligned}
$$

所以，参数估计$\theta$一定收敛。

# EM算法在高斯混合模型中的应用

## 数学推导
### 模型简介
高斯混合模型（Gaussian Mixture Model，GMM）可以理解为$k$个高斯分布以概率$\alpha_1, \alpha_2, ..., \alpha_k$组合在一起，其中，$\sum_{j=1}^{k}{\alpha_j}=1$。从中高斯混合模型中抽样的过程为：先取一个随机变量$z$（共$k$种结果，每种结果取到的概率为$\alpha_{j=1,2,...,k}$），再从$z$对应的高斯分布$N(\mu_j, \Sigma_j)$中取出一个$x$。其中，$z$为隐变量（latent variable），$x$为观测变量（observed variable），我们只能看到取样结果为$x_{i=1,2,...,n}$。目的是根据这$n$个可观测的取样结果估计出高斯混合模型的全部参数$\alpha_{j=1,2,...,k}$，$\mu_{j=1,2,...,k}$和$\Sigma_{j=1,2,...,k}$。

我们的方法是用EM算法给出GMM中的参数估计，如下：

### E步

写出$Q$函数：

$$
Q(\theta, \theta^{(t)}) = \sum_{i=1}^{n}{\sum_{j=1}^{k}{P(z_j|x_i; \theta^{(t)})}}\log{P(z_j, x_i; \theta)}
$$

令

$$
\begin{aligned}
    \gamma_{ij}^{(t)} &= P(z_j|x_i; \theta^{(t)}) \\
    &= \frac{P(z_j; \theta^{(t)})P(x_i|z_j; \theta^{(t)})}{P(x_i; \theta^{(t)})} \\
    &= \frac{P(z_j; \theta^{(t)})P(x_i|z_j; \theta^{(t)})}{\sum_{h=1}^{k}{P(z_h; \theta^{(t)})P(x_i|z_h; \theta^{(t)})}} \\
    &= \frac{\alpha_j^{(t)}\phi(x_i;\mu_j^{(t)}, \Sigma_j^{(t)})}{\sum_{h=1}^{k}{\alpha_h^{(t)}\phi(x_i;\mu_h^{(t)}, \Sigma_h^{(t)})}}
\end{aligned}
$$

其中，$\gamma_{ij}^{(t)}$满足$\sum_{j=1}^{k}{\gamma_{ij}^{(t)}}=1$，$\phi$为高斯分布的概率密度函数：

$$
\phi(x; \mu, \Sigma) = \frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}\exp[-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)]
$$

于是，

$$
\begin{aligned}
    Q(\theta, \theta^{(t)}) &= \sum_{i=1}^{n}{\sum_{j=1}^{k}{\gamma_{ij}^{(t)}\log{P(z_j, x_i; \theta)}}} \\
    &= \sum_{i=1}^{n}{\sum_{j=1}^{k}{\gamma_{ij}^{(t)}\log{P(z_j; \theta)P(x_i|z_j; \theta)}}} \\
    &= \sum_{i=1}^{n}{\sum_{j=1}^{k}{\gamma_{ij}^{(t)}\log{\alpha_j\phi(x_i; \mu_j, \Sigma_j)}}}
\end{aligned}
$$

### M步

对$Q$函数极大化：

1）求$\alpha_j$：由于$\alpha_j$满足$\sum_{j=1}^{k}{\alpha_j}=1$，所以需要用拉格朗日乘数法。令

$$
\frac{\partial{Q^{\prime}(\theta, \theta^{(t)})}}{\partial{\alpha_j}} = \sum_{i=1}^{n}{\gamma_{ij}^{(t)}\frac{1}{\alpha_j}} - \lambda = 0
$$

其中，$Q^{\prime}(\theta, \theta^{(t)}) = Q(\theta, \theta^{(t)}) - \lambda(\sum_{j=1}^{k}{\alpha_j} - 1)$，$\lambda$为拉格朗日乘数。

又因为$\sum_{j=1}^{k}{\alpha_j}=1$，所以

$$
\begin{aligned}
    \lambda &= \sum_{j=1}^{k}{\sum_{i=1}^{n}{\gamma_{ij}^{(t)}}} \\
    &= \sum_{i=1}^{n}{\sum_{j=1}^{k}{\gamma_{ij}^{(t)}}} \\
    &= n
\end{aligned}
$$

得

$$
\alpha_j = \frac{\sum_{i=1}^{n}{\gamma_{ij}^{(t)}}}{n}
$$

2）求$\mu_j$：令

$$
\frac{\partial{Q(\theta, \theta^{(t)})}}{\partial{\mu_j}} = \sum_{i=1}^{n}{\gamma_{ij}^{(t)}\Sigma_j^{-1}(x_i - \mu_j)} = 0
$$

得

$$
\mu_j = \frac{\sum_{i=1}^{n}{\gamma_{ij}^{(t)}x_i}}{\sum_{i=1}^{n}{\gamma_{ij}^{(t)}}}
$$

3）求$\Sigma_j$：令

$$
\frac{\partial{Q(\theta, \theta^{(t)})}}{\partial{\Sigma_j}} = \sum_{i=1}^{n}{\gamma_{ij}^{(t)}[-\frac{1}{2}\Sigma_j^{-1}+\frac{1}{2}\Sigma_j^{-1}(x_i-\mu_j)(x_i-\mu_j)^T\Sigma_j^{-1}]} = 0
$$

得

$$
\Sigma_j = \frac{\sum_{i=1}^{n}{\gamma_{ij}^{(t)}(x_i-\mu_j)(x_i-\mu_j)^T}}{\sum_{i=1}^{n}{\gamma_{ij}^{(t)}}}
$$

## 代码实现
见gmm.py

# 参考资料

1.Dempster, A. P., et al. “Maximum Likelihood from Incomplete Data via the EM Algorithm.” Journal of the Royal Statistical Society. Series B (Methodological), vol. 39, no. 1, 1977, pp. 1–38. JSTOR

2.《统计学习方法（第二版）》，李航

3.https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm

4.https://mp.weixin.qq.com/s/A1U7dQNL2C7TmSvbse55bA
