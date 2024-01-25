# CS305 Assignment 02

12110644 周思呈

## Q1

**The UDP checksum provides for error detection. Consider the following word with 32 bits：**
$$
01100110011000000101010101010101
$$
**(a) Compute the checksum. (Recall that UDP computes checksum based on 16-bit word.) Break the 32-bit word into two 16-bit words, and sum their up.**

|          | 0    | 1    | 1    | 0    | 0    | 1    | 1    | 0    | 0    | 1    | 1    | 0    | 0    | 0    | 0    | 0    |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|          | 0    | 1    | 0    | 1    | 0    | 1    | 0    | 1    | 0    | 1    | 0    | 1    | 0    | 1    | 0    | 1    |
| +        | 1    | 0    | 1    | 1    | 1    | 0    | 1    | 1    | 1    | 0    | 1    | 1    | 0    | 1    | 0    | 1    |
| Negation | 0    | 1    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 1    | 0    | 0    | 1    | 0    | 1    | 0    |

**(b) How does the receiver detect errors?**

At the receiver, all 16-bit words are added. If no errors are introduced into the packet, then the sum will be 1111_1111_1111_1111. If one of the bits is a 0, then error/errors is/are detected.

**(c) If the receiver does not detect any error using the checksum, does it mean that the message was transmitted without any error? Please explain the reason and provide an example.**

No. There might be more than one error.

For example, if the 15th bit and the 31st bit take their own negation, the receiver will not detect an error.

## Q2

**Fill in the blanks (B1) and (B2) in the below figure for go-back-N and selective repeat. Note that (B1) should be fill in with the action of the receiver, and (B2) should be fill in with the sender window. In the figure, all packets are transmitted successfully without error except pkt1.**

<img src="/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231109075847402.png" alt="image-20231109075847402" style="zoom:50%;" />

| Strategy | B1                  | B2          |
| -------- | ------------------- | ----------- |
| GBN      | Discard, send ack0  | 01**234**56 |
| SR       | Buffered, send ack2 | 0123**456** |

## Q3

**The following figure illustrates the convergence of TCP’s additive-increase multiplicative-decrease (AIMD) algorithm. Suppose that instead of a multiplicative decrease, TCP decreased the window size by a constant amount. Would the resulting additive-increase additive-decrease (AIAD) algorithm converge to an equal share algorithm? Justify your answer using a diagram similar to the above figure. (Note: Simply draw the diagram is not sufficient. You need to explain what the diagram shows.)**

<img src="/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231109081946508.png" alt="image-20231109081946508" style="zoom:50%;" />

<img src="/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231109103803382.png" alt="image-20231109103803382" style="zoom:40%;" />

Start from A(x1, s2), because the amount of link bandwidth jointly consumed by the two connections is less than R, no loss will occur, so both connections will increase their window by 1 MSS per RTT as a result of TCP’s congestion-avoidance algorithm. Thus, the joint throughput of the two connections proceeds along a 45-degree line.

Eventually, the link bandwidth jointly consumed by the two connections will be greater than R, and eventually packet loss will occur. Suppose that connections 1 and 2 experience packet loss when they realize throughputs indicated by point B(x1+a0, x2+a0). Connections 1 and 2 then decrease their windows by a constant amount, let's say a1. The resulting throughputs realized are thus at point C(x1+a0-a1, x2+a0-a1), also proceeds along a 45-degree line. Because the joint bandwidth use is less than R at point C, the two connections again increase their throughputs along a 45-degree line starting from C.

This process repeat and does not converge to fairness.

## Q4

**Draw the TCP connection-establishment procedure (that is, TCP three-way handshaking) between a client host and a server host. Suppose the initial sequence number of the client host is 25, and that of the server host is 89. For each segment exchange between the client and server, please indicate (1) the SYN bit, sequence number, acknowledgement number (if necessary); (2) whether the segment can carry data in the segment payload (that is, in the segment data filed)**

<img src="/Users/zhousicheng/Library/Application Support/typora-user-images/image-20231109093103403.png" alt="image-20231109093103403" style="zoom:40%;" />

## Q5

**Consider the TCP procedure for estimating RTT. Suppose that α = 0.1, and EstimatedRTT is initialized as EstimatedRTT0. Recall that**
$$
EstimatedRTT = (1 −α)EstimatedRTT + αSampleRTT.
$$
**(a) For a given TCP connection, suppose four acknowledgments have been returned in sequence with corresponding sample RTTs: SampleRTT1, SampleRTT2, SampleRTT3, and SampleRTT4. Express EstimatedRTT in terms of EstimatedRTT0 and the four sample RTTs.**
$$
\begin{aligned}
EstimatedRTT_1 &= (1 −α)EstimatedRTT_0 + αSampleRTT_1 \\
EstimatedRTT_2 &= (1 −α)EstimatedRTT_1 + αSampleRTT_2 \\
&= (1 −α)^2EstimatedRTT_0 + α(1 −α)SampleRTT_1 + αSampleRTT_2 \\
EstimatedRTT_3 &= (1 −α)EstimatedRTT_2 + αSampleRTT_3 \\
&= (1 −α)^3EstimatedRTT_0 + α(1 −α)^2SampleRTT_1 + α(1 −α)SampleRTT_2 + αSampleRTT_3\\
EstimatedRTT_4 &= (1 −α)EstimatedRTT_3 + αSampleRTT_4 \\
&= (1 −α)^4EstimatedRTT_0 + α(1 −α)^3SampleRTT_1 + α(1 −α)^2SampleRTT_2 \\
&\text{ }\text{ }+ α(1 −α)SampleRTT_3 + αSampleRTT_4 \\
&= (0.9)^4EstimatedRTT_0 + 0.1(0.9)^3SampleRTT_1 + 0.1(0.9)^2SampleRTT_2 \\
&\text{ }\text{ }+0.1(0.9)SampleRTT_3 + 0.1SampleRTT_4
\end{aligned}
$$
**(b) Generalize your formula for n sample RTTs.**
$$
EstimatedRTT = (1 −α)^nEstimatedRTT_0 + \alpha\sum^n_{i=1} (1 −α)^{n-i}SampleRTT_i
$$
