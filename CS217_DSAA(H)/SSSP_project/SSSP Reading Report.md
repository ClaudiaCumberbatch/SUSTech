# 1 Problem Definition
在带负权重边的图上解决单源最短路径问题。输入图G和源s，输出最短路径树。现有相关理论主要为Dijkstra和Bellman-Ford算法。前者复杂度O(m+nlogn)近线性，但要求没有负权边；后者只要求没有负权重环，但复杂度O(mn)。

主要贡献为提出了能够计算Price Function $\omega_{\phi}(u, v)=\omega(u, v)+\phi(u)-\phi(v)$ 的ScaleDown算法，ScaleDown$(G=(V, E, w), \Delta, B)$ 输入满足$w(e) \geq-2 B$ 和$\eta\left(G^{B}\right) \leq \Delta$ ，输出price function满足$w_{\phi}(e) \geq-B$ ，即将原图中负权边的权重从-2B提升至-B。在SPmain中调用O(logn)次ScaleDown将原图中负权重边转化为非负权重，同时保证最短路径不变，并在所得结果上运行Dijkstra算法得到近线形时间复杂度。

主要算法流程如下图所示。
![[SPmain(p13).png]]

# 2 The idea illustration example
放在文末

# 3 The time complexity analysis
## LDD 
$O\left(|V(G)| \log ^{3}(n)+|E(G)| \log ^{2}(n)\right)$ 确实没看懂。
将非负权重图分解为直径更小的强连通团。输入非负权重图G和正整数D，输出边集E，使得G\\E中的强连通团距离小于等于D。研究者改进了[BPW20]中提出的算法，用静态集代替动态集，使得算法速度更快。
## FixDAGEdges
按照拓扑序遍历SCC，相同SCC中的顶点赋值相同的price function使进入下一个SCC时边权为正。时间复杂度$O\left(m + n \right)$ 。
## ElimNeg
输入的图出度为常数，输出price function使所有边权均为正数。对于每个顶点都需要$\eta_{G}(v)+1$ 次循环。总时间复杂度$O\left(\log (n) \cdot\left(n+\sum_{v \in V} \eta_{G}(v)\right)\right)$ 。可以看到其时间复杂度与图中存在的负权边数量相关，因此在ScaleDown调用过程中先减少负权边数量，再调用该函数。

定义满足$d(v)+w(v, x)<d(x)$ 的边$(v,x)$ 为active，反之为inactive。所有操作结束后，所有边均为inactive。第一阶段调用Dijkstra，使所有$E \backslash E^{n e g}(G)$ 中的边inactive。第二阶段调用Bellman-Ford，对于所有负权边，如果边$(u,v)$ active，那么$u$ 一定在队列Q当中。

所有顶点的出度都是常数，每次标记一个顶点时，它要么被添加到Q中，要么被从Q中删除。对于在Dijkstra阶段从Q中提取的每个顶点和在Bellman-Ford阶段的外层for循环中处理的每个标记顶点，只存在固定数量的出边，因此处理每个这样的顶点只需要O(log n)时间，主要需要考虑的时间复杂度部分在于Q的更新。由于Q中删除的次数不能超过插入的次数，因此Q中的插入次数为$O\left(\sum_{v \in V} \eta_{G}(v)\right)$ 。

## ScaleDown
### $\Delta \le 2$ 
直接进入phase 3调用ElimNeg，时间复杂度$O\left(m \log m \right)$ 。
### $\Delta >  2$ 
Phase 0，LDD，$O\left(m \log ^{3}(m)\right)$ 。
通过递归调用ScaleDown和FixDAGEdges减少图中负权边数量，然后调用ElimNeg。由于原图中不包含负权环，其子图同样不包含负权环，因此每次递归调用时间复杂度均为$O\left(m \log ^{3}(m)\right)$ ，调用$O\left(\log \Delta\right)$ 次，因此总复杂度$O\left(m \log ^{3}(m) \log \Delta\right)$ 。
$P_{G^{B}}(v)$  是$\left(G_{\phi_{2}}^{B}\right)_{s}$ 中的最短路径，因此
$$\begin{aligned}
\eta_{G_{\phi_{2}}^{B}}(v) & =\min \left\{\left|P \cap E^{n e g}\left(G_{\phi_{2}}^{B}\right)\right|: P \text { is a shortest } s v \text {-path in }\left(G_{\phi_{2}}^{B}\right)_{s}\right\} \\
& \leq\left|P_{G^{B}}(v) \cap E^{n e g}\left(G_{\phi_{2}}^{B}\right)\right|
\end{aligned}$$

$$E\left[\eta_{G_{\phi_{2}}^{B}}(v)\right] \leq E\left[\left|P_{G^{B}}(v) \cap E^{r e m}\right|\right]=O\left(\log ^{2} m\right)$$
因此ElimNeg的时间复杂度为
$$O\left(\left(m+E\left[\sum_{v \in V} \eta_{G_{\phi_{2}}^{B}}(v)\right]\right) \log m\right)=O\left(m \log ^{3} m\right)$$ 
## SPmain
复杂度主要由$\log (B)=O(\log (n))$ 次调用ScaleDown决定，上述准备工作完成后调用Dijkstra计算最短路径树，复杂度为$O\left(m + n\log n \right)$ ，总复杂度$O\left(m \log ^{5}(n)\right).$ 

# 4 Conclusion
最开始读论文完全是一头雾水，于是尝试画了一个example来加深理解。后来还是有很多不理解的地方，于是尝试和同学讨论，于是发现自己画example的时候犯了很多错误。总的来说画example&与同学讨论问题都是加深对整篇论文理解的有效方法。不过最后对于随机算法的理解和许多对时间复杂度的理解还是有很多欠缺，可能需要通过日后的学习进一步加深理解。