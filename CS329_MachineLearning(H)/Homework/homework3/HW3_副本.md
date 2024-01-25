# CS405 Homework #3

12110644 周思呈

## Question 1

**Consider a data set in which each data point $t_n$ is associated with a weighting factor $r_n>0$, so that the sum-of-squares error function becomes **
$$
E_D (\mathbf{w}) = \frac{1}{2}\sum_{n=1}^Nr_n\{t_n-\mathbf{w^T}\phi(\mathbf{x}_n)\}^2.
$$
**Find an expression for the solution $\mathbf{w}^*$ that minimizes this error function. Give two alternative interpretations of the weighted sum-of-squares error function in terms of (i) data dependent noise variance and (ii) replicated data points.**

![image-20231108163706263](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231108163706263.png)

Take the derivative of (3.104) with respect to $\boldsymbol{w}$ and set it equal to 0
$$
\begin{aligned}
\nabla E_{D}(\boldsymbol{w}) &= \sum_{n=1}^{N} r_{n}\left\{t_{n}-\boldsymbol{w}^{T} \boldsymbol{\Phi}\left(\boldsymbol{x}_{n}\right)\right\} \boldsymbol{\Phi}\left(\boldsymbol{x}_{n}\right)^{T} = 0 \\
&= 0=\sum_{n=1}^{N} r_{n} t_{n} \boldsymbol{\Phi}\left(\boldsymbol{x}_{n}\right)^{T}-\boldsymbol{w}^{T}\left(\sum_{n=1}^{N} r_{n} \boldsymbol{\Phi}\left(\boldsymbol{x}_{n}\right) \boldsymbol{\Phi}\left(\boldsymbol{x}_{n}\right)^{T}\right)
\end{aligned}
$$


To achieve a similar form with (3.14), we denote $\sqrt{r_{n}} \boldsymbol{\phi}\left(\boldsymbol{x}_{\boldsymbol{n}}\right)=\boldsymbol{\phi}^{\prime}\left(\boldsymbol{x}_{\boldsymbol{n}}\right)$  and  $\sqrt{r_{n}} t_{n}=t_{n}^{\prime}$ 
$$
0=\sum_{n=1}^{N} t_{n}^{\prime} \Phi^{\prime}\left(\boldsymbol{x}_{n}\right)^{T}-\boldsymbol{w}^{T}\left(\sum_{n=1}^{N} \boldsymbol{\Phi}^{\prime}\left(\boldsymbol{x}_{n}\right) \boldsymbol{\Phi}^{\prime}\left(\boldsymbol{x}_{\boldsymbol{n}}\right)^{T}\right) \\
 \boldsymbol{w}_{M L}=\left(\boldsymbol{\Phi}^{T} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{T} \boldsymbol{t} 
$$
where
$$
\boldsymbol{t}=\left[\sqrt{r_{1}} t_{1}, \sqrt{r_{2}} t_{2}, \ldots, \sqrt{r_{N}} t_{N}\right]^{T} \\
\boldsymbol{\Phi}(i, j)=\sqrt{r_{i}} \phi_{j}\left(x_{i}\right)
$$
If we substitute  $\beta^{-1}$  by  $r_{n} \cdot \beta^{-1}$  in the summation term,  (3.12) will be the same as (3.104).

$r_{n}$  can be viewed as the observation time of $\left(\mathbf{x}_{n}, t_{n}\right)$.

## Question 2

**We saw in Section 2.3.6 that the conjugate prior for a Gaussian distribution with unknown mean and unknown precision (inverse variance) is a normal-gamma distribution. This property also holds for the case of the conditional Gaussian distribution $p(t|\mathbf{x,w},\beta)$ of the linear regression model. If we consider the likelihood function,**
$$
p(\mathbf{t}|\mathbf{X},{\rm w},\beta)=\prod_{n=1}^{N}\mathcal{N}(t_n|{\rm w}^{\rm T}\phi({\rm x}_n),\beta^{-1})
$$
**then the conjugate prior for $\mathbf{w}$ and $\beta$ is given by**
$$
p(\mathbf{w},\beta)=\mathcal{N}(\mathbf{w|m}_0, \beta^{-1}\mathbf{S}_0) {\rm Gam}(\beta|a_0,b_0).
$$
**Show that the corresponding posterior distribution takes the same functional form, so that**
$$
p(\mathbf{w},\beta|\mathbf{t})=\mathcal{N}(\mathbf{w|m}_N, \beta^{-1}\mathbf{S}_N) {\rm Gam}(\beta|a_N,b_N).
$$
**and find expressions for the posterior parameters $\mathbf{m}_N$, $\mathbf{S}_N$, $a_N$, and $b_N$.**

![image-20231108193506510](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231108193506510.png)

From (3.112) we have
$$
\begin{aligned}
p(\boldsymbol{w}, \beta) & =\mathscr{N}\left(\boldsymbol{w} \mid \boldsymbol{m}_{\mathbf{0}}, \beta^{-1} \mathbf{S}_{\mathbf{0}}\right) \operatorname{Gam}\left(\beta \mid a_{0}, b_{0}\right) \\
& \propto\left(\frac{\beta}{\left|\mathbf{S}_{\mathbf{0}}\right|}\right)^{2} \exp \{-\frac{1}{2}\left(\boldsymbol{w}-\boldsymbol{m}_{\mathbf{0}}\right)^{T} \beta \mathbf{S}_{\mathbf{0}}^{-1}\left(\boldsymbol{w}-\boldsymbol{m}_{\mathbf{0}}\right)\} b_{0}^{a_{0}} \beta^{a_{0}-1} \exp \{-b_{0} \beta\}
\end{aligned}
$$
Because
$$
p(\boldsymbol{w}, \beta \mid \mathbf{t}) \propto p(\mathbf{t} \mid \mathbf{X}, \boldsymbol{w}, \beta) \times p(\boldsymbol{w}, \beta)
$$
and we have
$$
\begin{aligned}
p(\mathbf{t} \mid \mathbf{X}, \boldsymbol{w}, \beta) & =\prod_{n=1}^{N} \mathscr{N}\left(t_{n} \mid \boldsymbol{w}^{T} \boldsymbol{\phi}\left(\boldsymbol{x}_{n}\right), \beta^{-1}\right) \\
& \propto \prod_{n=1}^{N} \beta^{1 / 2} \exp \left[-\frac{\beta}{2}\left(t_{n}-\boldsymbol{w}^{T} \boldsymbol{\phi}\left(\boldsymbol{x}_{n}\right)\right)^{2}\right] 
\end{aligned}
$$

$$
\begin{aligned}
\text { quadratic term } & =-\frac{\beta}{2} \boldsymbol{w}^{T} \boldsymbol{S}_{0}^{-1} \boldsymbol{w}+\sum_{n=1}^{N}-\frac{\beta}{2} \boldsymbol{w}^{T} \phi\left(x_{n}\right) \phi\left(x_{n}\right)^{T} \boldsymbol{w} \\
& =-\frac{\beta}{2} \boldsymbol{w}^{T}\left[\boldsymbol{S}_{0}{ }^{-1}+\sum_{n=1}^{N} \phi\left(x_{n}\right) \phi\left(x_{n}\right)^{T}\right] \boldsymbol{w} \\
\Rightarrow \mathbf{S}_{N}^{-1}&=\boldsymbol{S}_{0}{ }^{-1}+\sum_{n=1}^{N} \phi\left(x_{n}\right) \phi\left(x_{n}\right)^{T}
\end{aligned}
$$

$$
\begin{aligned}
\text { linear term } & =\beta \boldsymbol{m}_{\mathbf{0}}{ }^{T} \boldsymbol{S}_{\mathbf{0}}{ }^{-1} \boldsymbol{w}+\sum_{n=1}^{N} \beta t_{n} \boldsymbol{\phi}\left(\boldsymbol{x}_{\boldsymbol{n}}\right)^{T} \boldsymbol{w} \\
& =\beta\left[\boldsymbol{m}_{\mathbf{0}}{ }^{T} \boldsymbol{S}_{\mathbf{0}}{ }^{-1}+\sum_{n=1}^{N} t_{n} \boldsymbol{\phi}\left(\boldsymbol{x}_{\boldsymbol{n}}\right)^{T}\right] \boldsymbol{w}\\
\Rightarrow \boldsymbol{m}_{N}&=\boldsymbol{S}_{\boldsymbol{N}}\left[\boldsymbol{S}_{0}{ }^{-1} \boldsymbol{m}_{0}+\sum_{n=1}^{N} t_{n} \phi\left(\boldsymbol{x}_{n}\right)\right]
\end{aligned}
$$

$$
\begin{aligned}
\text { constant term } & =\left(-\frac{\beta}{2} \boldsymbol{m}_{\mathbf{0}}{ }^{T} \boldsymbol{S}_{\mathbf{0}}{ }^{-1} \boldsymbol{m}_{\mathbf{0}}-b_{0} \beta\right)-\frac{\beta}{2} \sum_{n=1}^{N} t_{n}^{2} \\
& =-\beta\left[\frac{1}{2} \boldsymbol{m}_{0}{ }^{T} \boldsymbol{S}_{\mathbf{0}}{ }^{-1} \boldsymbol{m}_{\mathbf{0}}+b_{0}+\frac{1}{2} \sum_{n=1}^{N} t_{n}^{2}\right] \\
\Rightarrow b_{N}&=\frac{1}{2} \boldsymbol{m}_{0}^{T} \boldsymbol{S}_{0}^{-1} \boldsymbol{m}_{0}+b_{0}+\frac{1}{2} \sum_{n=1}^{N} t_{n}^{2}-\frac{1}{2} \boldsymbol{m}_{N}{ }^{T} \boldsymbol{S}_{\boldsymbol{N}}{ }^{-1} \boldsymbol{m}_{\boldsymbol{N}}
\end{aligned}
$$

$$
\begin{aligned}
\text { exponent term }&=\left(2+a_{0}-1\right)+\frac{N}{2} \\
\Rightarrow a_{N}&=a_{0}+\frac{N}{2}
\end{aligned}
$$

## Question 3

**Show that the integration over $w$ in the Bayesian linear regression model gives the result**
$$
\int \exp\{-E(\mathbf{w})\} {\rm d}\mathbf{w}=\exp\{-E(\mathbf{m}_N)\}(2\pi)^{M/2}|\mathbf{A}|^{-1/2}.
$$
**Hence show that the log marginal likelihood is given by**
$$
\ln p(\mathbf{t}|\alpha,\beta)=\frac{M}{2}\ln\alpha+\frac{N}{2}\ln\beta-E(\mathbf{m}_N)-\frac{1}{2}\ln|\mathbf{A}|-\frac{N}{2}\ln(2\pi)
$$
![image-20231108202140636](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231108202140636.png)

From multivariate normal distribution, we have
$$
\int \frac{1}{(2 \pi)^{M / 2}} \frac{1}{|\mathbf{A}|^{1 / 2}} \exp \left\{-\frac{1}{2}\left(\boldsymbol{w}-\boldsymbol{m}_{N}\right)^{T} \mathbf{A}\left(\boldsymbol{w}-\boldsymbol{m}_{N}\right)\right\} d \boldsymbol{w}=1
$$
Hence
$$
\int \exp \left\{-\frac{1}{2}\left(\boldsymbol{w}-\boldsymbol{m}_{N}\right)^{T} \mathbf{A}\left(\boldsymbol{w}-\boldsymbol{m}_{N}\right)\right\} d \boldsymbol{w}=(2 \pi)^{M / 2}|\mathbf{A}|^{1 / 2}
$$


## Question 4

**Consider real-valued variables $X$ and $Y$. The $Y$ variable is generated, conditional on $X$, from the following process:**
$$
\epsilon\sim N(0,\sigma^2) \\
Y=aX+\epsilon
$$
**where every $\epsilon$ is an independent variable, called a noise term, which is drawn from a Gaussian distribution with mean 0, and standard deviation $\sigma$. This is a one-feature linear regression model, where $a$ is the only weight parameter. The conditional probability of $Y$ has distribution $p(Y|X, a)\sim N(aX, \sigma^2)$, so it can be written as**
$$
p(Y|X,a)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{1}{2\sigma^2}(Y-aX)^2)
$$
**Assume we have a training dataset of $n$ pairs ($X_i, Y_i$) for $i = 1...n$, and $\sigma$ is known.**

**Derive the maximum likelihood estimate of the parameter $a$ in terms of the training example $X_i$'s and $Y_i$'s. We recommend you start with the simplest form of the problem:**
$$
F(a)=\frac{1}{2}\sum_{i}(Y_i-aX_i)^2
$$

$$
\begin{aligned}
\frac{\partial F}{\partial a} &= \sum _i (Y_i - aX_i)(-X_i) \\
&= \sum _i aX_i^2 - X_iY_i \\
\Rightarrow a &= \frac{\sum_i X_iY_i}{\sum X_i^2}
\end{aligned}
$$


## Question 5

**If a data point $y$ follows the Poisson distribution with rate parameter $\theta$, then the probability of a single observation $y$ is**
$$
p(y|\theta)=\frac{\theta^{y}e^{-\theta}}{y!}, {\rm for}\;y = 0, 1, 2,\dots
$$
**You are given data points $y_1, \dots ,y_n$ independently drawn from a Poisson distribution with parameter $\theta$ . Write down the log-likelihood of the data as a function of $\theta$ .**
$$
\begin{aligned}
log \text{ } p(y|\theta) &= ylog\theta - \theta - \sum^y_{i=0}logi \\
\Rightarrow L(\theta) &= \sum^n_{i=1} (y_ilog\theta - \theta - logy_i!)
\end{aligned}
$$


## Question 6

**Suppose you are given $n$ observations, $X_1,\dots,X_n$, independent and identically distributed with a $Gamma(\alpha, \lambda$) distribution. The following information might be useful for the problem.**

* **If $X\sim Gamma(\alpha,\lambda)$, then $\mathbb{E}[X]=\frac{\alpha}{\lambda}$ and $\mathbb{E}[X^2]=\frac{\alpha(\alpha+1)}{\lambda^2}$ **
* **The probability density function of $X\sim Gamma(\alpha,\lambda)$ is $f_X(x)=\frac{1}{\Gamma(\alpha)}\lambda^{\alpha}x^{\alpha-1}e^{-\lambda x}$ , where the function $\Gamma$ is only dependent on $\alpha$ and not $\lambda$.**

**Suppose we are given a known, fixed value for $\alpha$. Compute the maximum likelihood estimator for $\lambda$.**


$$
\begin{aligned}
log \text{ } f_X(x) &= \alpha log\lambda + (\alpha-1)logx - \lambda x - log\Gamma(\alpha)\\
L(\lambda) &= n\alpha log \lambda + (\alpha-1)log\prod^n_{i=1}x_i - \lambda \sum^n_{i=1}x_i - nlog\Gamma(\alpha)\\
\frac{dL(\lambda)}{d\lambda}&= \frac{n\alpha}{\lambda} - \sum^n_{i=1}x_i \\
\Rightarrow \lambda &= \frac{\alpha}{\frac{1}{n}\sum^n_{i=1}x_i}
\end{aligned}
$$
