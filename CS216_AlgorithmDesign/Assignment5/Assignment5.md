12110644周思呈
**Prove 3-SAT ≤ p 3-Color.**

```toc
```


## 1 3-SAT ≤ p Ciucuit
First show that any 3-SAT foumula can be converted to circuit.

- AND gate
$$\begin{array}{c}
x \Longleftrightarrow(a \wedge b) \\
(x \Longrightarrow(a \wedge b)) \wedge((a \wedge b) \Longrightarrow x) \\
(\neg x \vee(a \wedge b)) \wedge(\neg(a \wedge b) \vee x) \\
(\neg x \vee a) \wedge(\neg x \vee b) \wedge(\neg a \vee \neg b \vee x)
\end{array}$$

- OR gate
$$\begin{array}{c}
x \Longleftrightarrow(a \vee b) \\
(x \Longrightarrow(a \vee b)) \wedge((a \vee b) \Longrightarrow x) \\
(\neg x \vee(a \vee b)) \wedge(\neg(a \vee b) \vee x) \\
(\neg x \vee a \vee b) \wedge(\neg a \vee \neg b \vee x)
\end{array}$$

- NOT gate
$$\begin{array}{c}
a \Longleftrightarrow \neg x \\
(\neg x \Longrightarrow a) \wedge(a  \Longrightarrow x) \\
(a \vee x) \wedge(\neg a \vee \neg x)
\end{array}$$

In this way, we can convert every 3-SAT formula into a equivalent ciucuit.

## 2 Circuit ≤ p 3-Color
This part quote # 3 色问题的 NP-Complete 证明 ^[https://soptq.me/2020/06/26/3color-npc/].

Assume our 3 colors are T, F and N(something other than T and F). In the graph, every edge cannot connect vertexes of the same color. Next we will show that AND, OR, NOT gate can be represented by graphs in 3-Color problem context.

- Boolean Variable
Now we connect the original 3 vertexes(T, F, N) to each other. Then suppose we input a variable P. We connect it to N, thus P can only be T or F, which means P is a boolean variable.
![[1.jpeg | 500]]

- NOT gate
Based on the above diagram, we add a new vertex and connect it to both N and P. Because it connect to N, it's also a boolean variable. What's more, it connect to P, so it cannot have the same value as P does. As a result, the new variable must be "not P".
![[2.jpg | 500]]

- OR gate
Now suppose we have 3 variables, namely P, Q and R(P or Q). Similarly we connect them with N to make sure that they are boolean variables. Then we constuct a graph like below.

![[3 2.jpg]]

Here is the variable assign result.

| P | Q | A | B | C | D | R |
|---|---|---|---|---|---|---|
| T | T |   | F | N |   | T |
| T | F | N | F | N | F | T | 

Therefore we have R = P or Q.

Combining OR gate and NOT gate we have NOR gate. Furthermore, from NOR gate we can have any circuit we desire. So we now can convert any circuit into graph which is under the context of 3-Color problem. Thus complete the proof.