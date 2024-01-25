# Homework Ⅳ

12110644 周思呈

## Question 1

**Show that maximization of the class separation criterion given by $m_2 - m_1 = \mathbf{w^T(m_2 - m_1)}$ with respect to $\mathbf w$, using a Lagrange multiplier to enforce the constraint $\mathbf{w^T w = 1}$, leads to the result that $\mathbf w \propto \mathbf{(m_2 - m_1)}$.**

![image-20231121103022157](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231121103022157.png)


$$
L(\mathbf{w}) = \mathbf{w^T(m_2-m_1) - \lambda (w^T w-1)} \\
\frac{\part L}{\part \mathbf{w}} = \mathbf{(m_2 - m_1) - 2\lambda w} = 0 \\
\Rightarrow \mathbf{w} = \frac{\mathbf{m_1-m_2}}{2\lambda} \propto \mathbf{(m_2 - m_1)}
$$


## Question 2

**Show that the Fisher criterion**
$$
\mathrm J(\mathbf w) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2}
$$

**can be written in the form**
$$
\mathrm J(\mathbf w) = \mathbf{\frac{w^T S_B w}{w^T S_W w}}
$$

**Hint.**

$$
y = \mathbf{w^T x},\qquad
$$

$$
m_k = \mathbf{w^T m_k},\qquad
$$

$$
s_k^2 = \sum_{n\in\mathcal C_k}(y_n - m_k)^2
$$



Rewrite the formula as
$$
\begin{aligned}
\mathrm J(\mathbf w) &= \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2} \\
&= \mathbf{\frac{(w^T m_2 - w^T m_1)^2}{\sum _{n \in C_1} (y_n - m_1)^2 + \sum _{n \in C_2} (y_n - m_2)^2}}
\end{aligned}
$$

We define 
$$
\boldsymbol{S}_{B}=\left(\boldsymbol{m}_{2}-\boldsymbol{m}_{1}\right)\left(\boldsymbol{m}_{2}-\boldsymbol{m}_{2}\right)^{T} \\
\boldsymbol{S}_{W}=\sum_{n \in \mathcal{C}_{1}}\left(\boldsymbol{x}_{n}-\boldsymbol{m}_{1}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{m}_{1}\right)^{T}+\sum_{n \in \mathcal{C}_{2}}\left(\boldsymbol{x}_{n}-\boldsymbol{m}_{2}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{m}_{2}\right)^{T}
$$
So
$$
\mathrm J(\mathbf w) = \mathbf{\frac{w^T S_B w}{w^T S_W w}}
$$


## Question 3

**Consider a generative classification model for $K$ classes defined by prior class probabilities $p(\mathcal C_k) = \pi_k$ and general class-conditional densities $p(\phi|\mathcal C_k)$ where $\phi$ is the input feature vector. Suppose we are given a training data set \{ $\phi_n, \mathbf t_n$ \} where $n = 1, ..., N$, and $\mathbf t_n$ is a binary target vector of length $K$ that uses the 1-of-K coding scheme, so that it has components $t_{nj} = I_{jk}$ if pattern $n$ is from class $\mathcal C_k$.**

**Assuming that the data points are drawn independently from this model, show that the maximum-likelihood solution for the prior probabilities is given by**
$$
\pi_k = \frac{N_k}{N},
$$

**where $N_k$ is the number of data points assigned to class $\mathcal C_k$.**

![image-20231121113504069](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231121113504069.png)
$$
\begin{aligned}
p\left(\left\{\phi_{\mathbf{n}}, t_{n}\right\} \mid \pi_{1}, \pi_{2}, \ldots, \pi_{K}\right) & =\prod_{n=1}^{N} \prod_{k=1}^{K}\left[p\left(\boldsymbol{\phi}_{\boldsymbol{n}} \mid C_{k}\right) p\left(C_{k}\right)\right]^{t_{n k}} \\
& =\prod_{n=1}^{N} \prod_{k=1}^{K}\left[\pi_{k} p\left(\boldsymbol{\phi}_{\boldsymbol{n}} \mid C_{k}\right)\right]^{t_{n k}}
\end{aligned}
$$

$$
\ln p=\sum_{n=1}^{N} \sum_{k=1}^{K} t_{n k}\left[\ln \pi_{k}+\ln p\left(\phi_{n} \mid C_{k}\right)\right] \propto \sum_{n=1}^{N} \sum_{k=1}^{K} t_{n k} \ln \pi_{k}
$$



Add a Lagrange Multiplier to the expression
$$
L=\sum_{n=1}^{N} \sum_{k=1}^{K} t_{n k} \ln \pi_{k}+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right) \\
\frac{\partial L}{\partial \pi_{k}}=\sum_{n=1}^{N} \frac{t_{n k}}{\pi_{k}}+\lambda \\
\Rightarrow \pi_{k}=-\left(\sum_{n=1}^{N} t_{n k}\right) / \lambda=-\frac{N_{k}}{\lambda}
$$


Because $1=-\left(\sum_{k=1}^{K} N_{k}\right) / \lambda=-\frac{N}{\lambda}$, we can obtain  $\pi_{k}=\frac{N_{k}} {N}$ .



## Question 4 (4.12)

**Verify the relation**
$$
\frac{\mathrm d\sigma}{\mathrm da} = \sigma(1 - \sigma)
$$

**for the derivative of the logistic sigmoid function defined by**
$$
\sigma(a) = \frac{1}{1 + \mathrm{exp}(-a)}
$$

$$
\sigma(a)=\frac{1}{1+e^{-x}}=\frac{e^{x}}{e^{x}+1}=1-\left(e^{x}+1\right)^{-1}
$$

$$
\begin{aligned}
\frac{\mathrm d\sigma}{\mathrm da} & =(-1)(-1)\left(e^{x}+1\right)^{-2} e^{x} \\
& =\left(1+e^{-x}\right)^{-2} e^{-2 x} e^{x} \\
& =\left(1+e^{-x}\right)^{-1} \cdot \frac{e^{-x}}{1+e^{-x}} \\
& = \sigma(1 - \sigma)
\end{aligned}
$$



## Question 5 (4.13)

**By making use of the result**
$$
\frac{\mathrm d\sigma}{\mathrm da} = \sigma(1 - \sigma)
$$

**for the derivative of the logistic sigmoid, show that the derivative of the error function for the logistic regression model is given by**
$$
\nabla \mathbb E(\mathbf w) = \sum^N_{n=1}(y_n - t_n)\phi_n.
$$

**Hint.**

**The error function for the logistic regression model is given by**
$$
\mathbb E(\mathbf w) = -\mathrm{ln}p(\mathbf{t|w}) = -\sum^N_{n=1}\{t_n\mathrm{ln}y_n + (1 - t_n)\mathrm{ln}(1 - y_n)\}.
$$

Define $y_{n}=\sigma\left(a_{n}\right), a_{n}=\mathbf{w}^{T} \boldsymbol{\phi}_{n}$, we have
$$
\begin{aligned}
\nabla E(\boldsymbol{w}) & =-\nabla \sum_{n=1}^{N}\left\{t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right\} \\
& =-\sum_{n=1}^{N} \nabla\left\{t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right\} \\
& =-\sum_{n=1}^{N} \frac{d\left\{t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right\}}{d y_{n}} \frac{d y_{n}}{d a_{n}} \frac{d a_{n}}{d \mathbf{w}} \\
& =-\sum_{n=1}^{N}\left(\frac{t_{n}}{y_{n}}-\frac{1-t_{n}}{1-y_{n}}\right) \cdot y_{n}\left(1-y_{n}\right) \cdot \boldsymbol{\phi}_{n} \\
& =-\sum_{n=1}^{N} \frac{t_{n}-y_{n}}{y_{n}\left(1-y_{n}\right)} \cdot y_{n}\left(1-y_{n}\right) \cdot \boldsymbol{\phi}_{n} \\
& =-\sum_{n=1}^{N}\left(t_{n}-y_{n}\right) \boldsymbol{\phi}_{n} \\
& =\sum_{n=1}^{N}\left(y_{n}-t_{n}\right) \boldsymbol{\phi}_{n}
\end{aligned}
$$


## Question 6

**There are several possible ways in which to generalize the concept of linear discriminant functions from two classes to $c$ classes. One possibility would be to use ( $c-1$ ) linear discriminant functions, such that $y_k(\mathbf x )>0$ for inputs $\mathbf{x}$ in class $C_k$ and $y_k(\mathbf{x})<0$ for inputs not in class $C_k$.**

**By drawing a simple example in two dimensions for $c = 3$, show that this approach can lead to regions of x-space for which the classification is ambiguous.**

**Another approach would be to use one discriminant function $y_{jk}(\mathbf{x})$ for each possible pair of classes $C_j$ and $C_k$ , such that $y_{jk}(\mathbf{x})>0$ for patterns in class $C_j$ and $y_{jk}(\mathbf{x})<0$ for patterns in class $C_k$. For $c$ classes, we would need $c(c-1)/2$ discriminant functions.**

**Again, by drawing a specific example in two dimensions for $c = 3$, show that this approach can also lead to ambiguous regions.**

![image-20231126084038892](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231126084038892.png)



## Question 7

**Given a set of data points { $\{\mathbf{x}^n\}$ } we can define the convex hull to be the set of points $\mathbf{x}$ given by**
$$
\mathbf{x} = \sum_n\alpha_n\mathbf{x}^n
$$

**where $\alpha_n>=0$ and $\sum_n\alpha_n=1$. Consider a second set of points $\{\mathbf{z}^m\}$ and its corresponding convex hull. The two sets of points will be linearly separable if there exists a vector $\hat{\mathbf{w}}$ and a scalar $w_0$ such that $\hat{\mathbf{w}}^T\mathbf{x}^n+w_0>0$ for all $\mathbf{x}^n$, and $\hat{\mathbf{w}}^T\mathbf{z}^m+w_0<0$ for all $\mathbf{z}^m$.**

**Show that, if their convex hulls intersect, the two sets of points cannot be linearly separable, and conversely that, if they are linearly separable, their convex hulls do not intersect.**

![image-20231126084648583](/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231126084648583.png)

If the convex hull of  $\left\{\mathbf{x}_{\mathbf{n}}\right\}$  and $ \left\{\mathbf{z}_{\mathbf{m}}\right\}$  intersects, we know that there will be a point  $\mathbf{y}$  which can be written as $ \mathbf{y}=\sum_{n} \alpha_{n} \mathbf{x}_{\mathbf{n}} $ and also $ \mathbf{y}=\sum_{m} \beta_{m} \mathbf{z}_{\mathbf{m}}$ . Because $\sum_{n} \alpha_{n}=1$ we can obtain:
$$
\begin{aligned}
\widehat{\mathbf{w}}^{T} \mathbf{y}+w_{0} & =\widehat{\mathbf{w}}^{T}\left(\sum_{n} \alpha_{n} \mathbf{x}_{\mathbf{n}}\right)+w_{0} \\
& =\left(\sum_{n} \alpha_{n} \widehat{\mathbf{w}}^{T} \mathbf{x}_{\mathbf{n}}\right)+\left(\sum_{n} \alpha_{n}\right) w_{0} \\
& =\sum_{n} \alpha_{n}\left(\widehat{\mathbf{w}}^{T} \mathbf{x}_{\mathbf{n}}+w_{0}\right) 
\end{aligned}
$$
Similarly we have
$$
\begin{aligned}
\widehat{\mathbf{w}}^{T} \mathbf{y}+w_{0} & =\widehat{\mathbf{w}}^{T}\left(\sum_{m} \alpha_{m} \mathbf{z}_{\mathbf{m}}\right)+w_{0} \\
& =\left(\sum_{m} \alpha_{m} \widehat{\mathbf{w}}^{T} \mathbf{z}_{\mathbf{m}}\right)+\left(\sum_{m} \alpha_{m}\right) w_{0} \\
& =\sum_{m} \alpha_{m}\left(\widehat{\mathbf{w}}^{T} \mathbf{z}_{\mathbf{m}}+w_{0}\right) 
\end{aligned}
$$
 If  $\left\{\mathbf{x}_{\mathbf{n}}\right\}$  and $ \left\{\mathbf{z}_{\mathbf{m}}\right\}$  are linearly separable, for  $\forall \mathbf{x}_{\mathbf{n}}, \mathbf{z}_{\mathbf{m}} $ we have
$$
\widehat{\mathbf{w}}^{T} \mathbf{x}_{\mathbf{n}}+w_{0}>0 \\
 \widehat{\mathbf{w}}^{T} \mathbf{z}_{\mathbf{m}}+w_{0}<0
$$


which leads to the contradiction.
