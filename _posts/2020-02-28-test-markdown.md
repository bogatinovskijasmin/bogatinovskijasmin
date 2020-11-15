---
layout: post
title: Sample blog post
subtitle: Each post also has a subtitle
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---
The linear methods, although simple, they can provide very interesting insights for the data being observevd due to this parasimony. They are usually regarded as methods with high bias, meaning that their hypothesis usually does not corresponds to the real world relationship (due to the greater complexity result from various interactions between the features) of the descriptive attributes. In the case of Neural Networks one allows to learn arbitrary relationships between the data. This makes 

In the case for classification the linear models produce a linear decision boundary between the classes. In this work we are going to discuss:

1) Logistic regression

2) Perceptron and Neural Networks

3) Support Vector Machine (non-kernelzied version for classification)

Additional terms defined: 

1) Newton's method

2) linear separability

3) KKT conditions

4) complementary slackness

### Logistic regression

Logistic regression is a very simple method for classification. It assumes that the data can be fit with a linear function of the form $\beta^TX$, where $\beta \in R^{dx1}$ (+ optional bias term). If you want to add bias always add feature with value 1. That allows to move accoriding to one more degree of freedom. In some sence the linear function is measuring the intensity of belonging of an example to one class or the other. To quantify the intensity the linear regression is wrapped around a sigmoid function. It takes values between 0 and 1 (its form resembles a form similar like a form of the cdf of a Gaussian distribution).
Formally the problem of logistic regression can be defined as following:
Let's assume that we are given a training set $D= \{ (x_1, y_1), \dots (x_n, y_n)\}, n \in N, y \in \{0, 1\}$. Let's assume that the score of $p(z=1|x) = \frac{1}{1+e^{-\beta^Tx}}$, then  $p(z=0|x) = 1-\frac{1}{1+e^{-\beta^Tx}}$. We can write a liklihoodd function in the following form (q is the number of 1's):

\begin{equation}
L(\beta, X) = \prod_{i=1}^N\frac{1}{1+e^{-\beta^Tx_i}}^{y_{i}}(1-\frac{1}{1+e^{-\beta^Tx_i}})^{1-y_{i}}
\end{equation}
Since it is easier to work with log-liklihood we calculate the log likilihood of the equation above:
\begin{equation}
l = log(L(\beta, X)) = \sum_{i=1}^Ny_{i}log(\frac{1}{1+e^{-\beta^Tx_i}})+(1-y_{i})log(1-\frac{1}{1+e^{-\beta^Tx_i}})
\end{equation}

The later function is called. However, the solution of this problem is not existance in a closed form. To this end, one tries to approximate the solution using the Newtown method of calculating the derviative of the update and then using gradient descent to ascent to the minimize the cost function. 

#### Newton's method:

In calculus, Newton's method is an iterative method for finding the roots of a differentiable function F, which are solutions to the equation $F (x) = 0$. In optimization, Newton's method is applied to the derivative $f^′$ of a twice-differentiable function $f$ to find the roots of the derivative (solutions to $f^′(x) = 0$), also known as the stationary points of $f$. These solutions may be minima, maxima, or saddle points. The geometric interpretation of Newton's method is that at each iteration, it amounts to the fitting of a paraboloid to the surface of $f(x)$ at the trial value  $x_{k}$ having the same slopes and curvature as the surface at that point, and then proceeding to the maximum or minimum of that paraboloid (in higher dimensions, this may also be a saddle point). [From Wikipedia].

Using the Netown's method, performing some gradient calculation when we minimize the fucntion we end up in the following updates for the $\beta 's$.
\begin{equation}
\beta^{r+1} <= \beta^{r} - \frac{\delta^2 l}{\delta \beta \delta \beta^T}^{-1}\frac{\delta l}{\delta \beta}
\end{equation}
\begin{equation}
y = [0;1;\dots 0], y \in \{0,1\}
\end{equation}
\begin{equation}
X \in R^{nxd}
\end{equation}
\begin{equation}
p \in R^{nx1}
\end{equation}

W is a diagonal matrix such that $w_{il}=(1-p(x_i|\beta))p(x_i|\beta))$.
\begin{equation}
\frac{\delta l}{\delta \beta} = (y-p)^T
\end{equation}

\begin{equation}
\frac{\delta^2 l}{\delta \beta \beta^T} = -X^TWX
\end{equation}

Then for the $\beta's$ at each iteration we have:
\begin{equation}
\beta^{r+1} <= \beta^{r} - (-X^TW^{r}X)^{-1}(y-p^{r})^TX
\end{equation}

After several iteration this will converge and the appropriate solution for the  corresponding values can be found. It is very interesting to use the corresponding coefficients for interpretation and varinace explanation. Performing senstiivty analysis can make strong interpretation for the models behaviour. For more details on all of this visit the chapters on linear and logistic regression on Introduction on Statistical Learning by Tibshirani and  Hestie. 

The classification problem can be seen as minimization of the zero one loss in the binary classification case. This problem is NP hard to be optimized and it is not differentiable. To this end logistic regression tries to minmize the zero one loss with apprimating the decsion threshold with a smooth boundary of exponential form.

### Perceptron

Additional linear method that fits linear boundary between two classes is the perceptron method. It is buildin based on the similariity with the neural cells. It is applicable if the classes are linearly separable. That means that there exist a postiive quantity $c$ for which a boundary can be found such that the distance for each point from the decision boundary is at least c. It does not find the optimal decision boundary but just the fisrt one that it finds which the separation error is 0.

The model of the perceptron assumes linear relationship between the input variables $\beta^TX + \beta_o$.At this 

It intution can be build from studing the 2D space of linearly separable classes. For the points from both of the classes three properties can be identified:

1) Let's assume that we are given $x_{0}$ that the point belongs to a decision boundary $\beta^TX+\beta_{o}=0$; Then we can say that $\beta^Tx_{o} = -\beta_{o}$;

2) Let's assume that we are given $x_{o1}$ and $x_{o2}$  that are part of the decision boundary, then $\beta^Tx_{o1} + \beta_o=\beta^Tx_{o2} + \beta_o => \beta^T(x_{o1}-x_{o2})=0$. That means that $\beta$ is normal to the decision boundary. 2

3) The distance between $x_{i}$ and any $x_{o}$ on the decision boundary we have $\beta^T(x_{i}-x_{o})=\beta^Tx_{i}-\beta^Tx_{o}=\beta^Tx_{i}+\beta{o}$. 

We assume that $Y \in \{-1, 1\}$ and $X \in R^{dxn}$. The main idea behind the perceptron is to use the geometry of the space. So a point can be classified either in the correct or in the wrong side of the decision boundary. One merit of how good the classification is can be seen in the distance of a point to the decision boundary (property 3)). That said, one way to define the cost function is to assume it is the sum of the distancs of the missclassified points. Since the distance is always positive, to account for that we additionally multiply by the true label $y_i$ and also with $-1$ if the point is missclassifed. 
The cost function is then defined like:
\begin{equation}
J(\beta, X) = \sum_{i \in M} -y_{i}(\beta^Tx_i + \beta_o)
\end{equation}
We can perform gradient descent on this const function with an aim to minimize it. To do this we need the partial derivatives with respect to the parameters $\beta$ and $\beta_o$.
\begin{equation}
\frac{\delta J}{\delta \beta}= \sum_{i \in M} -y_{i}x_i 
\end{equation}

\begin{equation}
\frac{\delta J}{\delta \beta_o}= \sum_{i \in M} -y_{i} 
\end{equation}

After several iterations, if the data is **linearly separable** this algortihm will find one, not necesseraly the best line that separates the points. Also it is sensitive to noise and it has great variance with respect to adding new points, however it is an online method since it can adjust the decision boundary. The **linearly separable** is defined as existing a positive quantity C such that $y_i(\beta^Tx_i+\beta_o) \geq C$ for all the points. To train the method instead batch updateds of the gradinets one  can use stochastic or mini batch gradient. This increases the noise of the gradient estimates however it often makes the computation much much faster, theoreticaly the training cost is constant with respect to the number of steps it needs until it converges (it will fall in some local optimal or minimal at one point or another). The speed of convergence also depends on the size of C, the smaller the slower the perceptron will converge. 

Stacking of several perceptron in a not-constrained grid like form (horziontally or vertically) results in the creation of a method that can create vairous relationships between the inputs and various non-linear transformations. This in fact allows to improve the performance of the methods by a large margin and produce comparable and often state of the art results (in time of writting) for many problems. 


### Neural Network

Stacking of multiple perceptrons and changing the various activations that may arise inbetween allows to build very complex models with rich and powerful representations, we refer to as Neural Network since they are brain inspired methods. There are two strong points of neural networks:

1) A neural network is a universal approximation function. It can reconstruct any function up to an arbtiratry error. This universal approxmation property makes it very powerful method.

2) A neural network is compact. Mening it needs finite number of neurons and states to represent dependance within the data most sutied to optimizing the given objective function.

Fun fact: One layer neural network with infinite amount of neurons is equivalent to a Gaussian Process. 

**Proof of universal approximator**

First we will provide a narrative description of the proof for the universal approximation function. Let's assume we are given a two layered network with inputs $x_1$ and $x_2$. We will show that a two-layered network with a threshold fcn can approximate any function up to a certain accuracy. We observe that any function of $x_2$ ($x_1$ being fixed can be approximated as an infinite Foruier series $y(x_1, x_2) = \sum_s A_s(x_1)cos(sx_2)$. The coefficients themselves can be approximated with inifite Fourier Series: $y(x_1, x_2) = \sum_s \sum_l A_{sl}cos(lx_1)cos(sx_2)$. Then using the trigonometric inequality $cos(\alpha)cos(\beta)=\frac{1}{2}cos(\alpha + \beta) + \frac{1}{2}cos(\alpha - \beta)$ the approximateion function can be written as asum of cosines where each of them is a linear combination of the inputs $y(x_1, x_2) = \sum_{j=1}^{\inf}u_jcos(x_1w_{1j} + x_2w_{2j})$.
The later is in fact a two layer network. Furthermore, the cosine can be approximated with a sum of ste functions (Hevisides step functions). Since the threshold function is in fact a step function, then stacking of many of them will result into approximation of the cosine function, that leads to the proof that a two-layered network with threshold function can approximate any arbitrary function of the input. 

To train a neural network one is using the **Backpropagation** learning algorithm. It is derived and described in the following.
Let's assume we are give the MSE as a cost function where $y^*$ is the prediction of the network and y is the target, then $E = MSE=\frac{1}{2}(y-y^*)^2$. We observe 3 layers in the middle of the architecture. The goal is to find the update of the gradient of the weight in the neural network that is the most suited to the update of the weight. Let the layers be denoted with $l,i$ and $j$. At a neuron at each layer we decompose it at two parts $z_i$ and $a_i$. The $a_i=w_{l1}z_{l1} + \dots + w_{lr}z_{lr}$. The $z_i=\sigma(a_i)$ is some nonlinearity applied on the sum of the inputs. Then, 
\begin{equation}
\frac{\delta E}{\delta w_{il}} = \frac{\delta E}{\delta a_i}\frac{\delta a_i}{\delta w_{il}}
\end{equation}
one of this partial derivatives is straightforward for computation
\begin{equation}
\frac{\delta a_i}{\delta w_{il}} = z_{li}
\end{equation}

The derviative of one of the gradients is quite straight forward since it is the derivative of the activation.
However, for computation of $\frac{\delta E}{\delta a_i}$ we aim to rewrite it in the form that depends on the activation in the next layer. $d_i=\frac{\delta E}{\delta a_i}=\sum_j\frac{\delta E}{\delta a_j}\frac{\delta a_j}{\delta a_i} =\sum_j\frac{\delta a_j}{\delta a_i}d_j$. 

\begin{equation}
\frac{\delta a_j}{\delta a_i} = w_{ij}\sigma^{'}(a_i)
\end{equation}

This reformulation of the loss in fact means that we can rewrite the gradient for the current activation with respect to the next loss of the cost function on the next layer. Assuming that the neural network has a finite width, at one point in time we end up in caluclatiton of the last $d_n$ at layer n. Since we can caluclate that loss we can know the exact form of the last $d_n$ and we can backpropagete it so at the end we know the exact form of all of the gradients. 

---------
Algorithm

Step 1) Propagate the data through the network

Step 2) Calculate the last $\frac{\delta E}{\delta u}=d_k$

Step 3) Run backwards the corresponding calcluation for $d_i= \sigma^{'}(a_i)\sum_jw_{ij}d_j$ and obtain those estimates

Step 4) Calculate the derivative of $\frac{\delta E}{\delta w_{il}}=z_{li}d_i$

Step 5) Calculate the updates ot the weithts $w_{ij}<= w_{ij}-\eta \frac{\delta E}{\delta w_{il}}$

Despite this brief summarization of neural networks, we have devoted another chapter for them describing different architectures and their usfullness in various applications. 

### Support Vector Machines (SVMs)

Support Vector Machines are another very popular method for machine learning. Compared to perceptron they differ in a way that instead of finding the fisrt linear boundary of arbitrary "quality", in the case of linearly separable classes the SVMs find the optimal one. The optimal linear boundary is defined as a hyperplane that is simultaniously at the equal distance to the points in both of the classes. Other words to say that is that it has the largest margin. Margin represents the maximal distance between the hyperplane and the points from both of the classes. The basic idea is to have this margin be as maximal as possible. To build the intuiton we assume that the classes are linearly separable, then we have similar like with the perceptron:

\begin{equation}
d_i = \frac{y_i(\beta^Tx_i + \beta_0)}{||\beta||} \geq c | \frac{1}{c}
\end{equation}

We divide with c, in that way we are making the smallest possible disntae in the space to be 1. 

\begin{equation}
d_i = \frac{y_i(\beta^Tx_i + \beta_0)}{||\beta||c} \geq 1 
\end{equation}

The goal is to maximize the margin for all the points:
\begin{equation}
\max_{\beta} d_i <=> \max_{\beta} \frac{y_i(\beta^Tx_i + \beta_0)}{||\beta||^*}
\end{equation}

The later is ill defined problem. We introduce a corresponding constraint to make it well defined and rewrite the optimization function.

\begin{equation}
\max_{\beta} d_i <=> \min_{\beta} \frac{1}{2}\beta^T\beta
\end{equation}
\begin{equation}
y_i(\beta^Tx_i + \beta_0) \geq 1
\end{equation}
for all the points $1 \dots n$.

To solve this optmization problem we apply the Lagrangian method. We first write the Lagrangian dual form with respect to the dual variables $\alpha \geq 0$.

\begin{equation}
\max_{\beta} d_i <=> \min_{\beta} \frac{1}{2}\beta^T\beta => \max_{\alpha} 1/2\beta^T\beta - \sum_i^N\alpha_i[y_i(\beta^Tx_i + \beta_0) - 1] 
\end{equation}

We calculate now the partial derivatives of the Lagrangian with respect to the parameters $\beta$, $\beta_0$ and $\alpha$.

\begin{equation}
\frac{\delta L}{\delta \beta} = \beta - \sum_i \alpha_iy_ix_i = 0 => \beta = \sum_i \alpha_iy_ix_i
\end{equation}

\begin{equation}
\frac{\delta L}{\delta \beta_o} = \sum_i \alpha_iy_i = 0
\end{equation}

We can now rewrite the cost function in terms only on the parameters $\alpha$.
\begin{equation}
\max_{\beta} d_i <=> \min_{\beta} \frac{1}{2}\beta^T\beta => \max_{\alpha} 1/2\beta^T\beta - \sum_i^N\alpha_i[y_i(\beta^Tx_i + \beta_0) - 1] = \frac{1}{2} \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j -  \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j -  \sum_i\alpha_iy_i\beta_0 + \sum_i \alpha_i = -\frac{1}{2} \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j + \sum_i \alpha_i
\end{equation}
s.t: 
\begin{equation}
\alpha_i \geq 0
\end{equation}

\begin{equation}
\sum_i \alpha_iy_i= 0
\end{equation}
This is a quadratic loss function with linear constraints. As such it can be solved by any solver of quadratic programming problems. 

Further, we need to check if the KKT (Kaurn-Kuhn-Tucker) conditions hold.

1) The derivative of the Lagrangian in the charactersistic point is 0;

2) The primal conditions are met

3) The dual conditions are met

4) Complementary slackness; The product of the dual variable and primal constraint should be equal to 0;
\begin{equation}
\alpha_i y_i(\beta^Tx_i + \beta_o)=0
\end{equation}

For the 4) point to be true, there are two cases, either $\alpha_i = 0$. This means that $y_i(\beta^Tx_i + \beta_o) \geq 0$. So for the points that do not lie on the marginal lines the coefficient $\alpha_i$ are zero.
The second case involves that $y_i(\beta^Tx_i + \beta_o)=0$, which means that the point lie on the margin. In that case $\alpha_i$ > 0. This points are called support vectors. For SVM, those are inedeed the most important points that make the distiction between the different classes. One can see that in that regard SVM is a vert efficient algorithm since it requires just the support vectors to be stored in the memory, as well the corresponding coefficients (if primal formulation of the problem is considered). There are various algortihms that are implementing the SVM paradigm for classification, e.g. John's Platt SMO (sequential maximization optimization). We will discuss them later when putting the SVM in the context of kernels.

In the following we will explain the **soft margin** case for support vector machines that extends the hard margin case allowing for modeling of cases where there is no linear separability between the classes via the introduction of a novel slack variable that amounts for the allowed violotation (overshoot) over the margins.

### Soft margin 

The previously described method is not able to provide a clssification of the points if they are not linearly separable. To overcome this limmitation to all of the points we add additional degree of freedom. We allow for each point to exist some positive quantity such that its missclassification will be tolerable. In such a way we are happy if we make mistakes by small errors.

This yields to the following formulation of the prolbem:

\begin{equation}
\max_{\beta} d_i <=> \min_{\beta} \frac{1}{2}\beta^T\beta + C\sum_i \epsilon_i
\end{equation}
\begin{equation}
y_i(\beta^Tx_i + \beta_0) \geq 1 - \epsilon_i
\end{equation}
for all the points $1 \dots n$ and $\epsilon_i \geq 0$.

To minimize this cost function we also consider the maximiazation of the Lagrangian function. We write it down. It has one more dual variable since there is  one more variable in the optimization fcn to  be considered. 


\begin{equation}
\max_{\beta} d_i <=> \min_{\beta} \frac{1}{2}\beta^T\beta => \max_{\alpha} 1/2\beta^T\beta - \sum_i^N\alpha_i[y_i(\beta^Tx_i + \beta_0) - 1 + \epsilon_i] + C\sum_i \lambda_i \epsilon_i
\end{equation}

We calculate now the partial derivatives of the Lagrangian with respect to the parameters $\beta$, $\beta_0$ and $\alpha$ and $\lambda$.

\begin{equation}
\frac{\delta L}{\delta \beta} = \beta - \sum_i \alpha_iy_ix_i = 0 => \beta = \sum_i \alpha_iy_ix_i 
\end{equation}

\begin{equation}
\frac{\delta L}{\delta \beta_o} = \sum_i \alpha_iy_i = 0
\end{equation}

\begin{equation}
\frac{\delta L}{\delta \epsilon} = C - \lambda_i -\alpha_i = 0
\end{equation}


We can now rewrite the cost function in terms only on the parameters $\alpha$.
\begin{equation}
\max_{\beta} d_i <=> \min_{\beta} \frac{1}{2}\beta^T\beta + C\sum_i \lambda_i\epsilon_i => \max_{\alpha, \lambda} 1/2\beta^T\beta - \sum_i^N\alpha_i[y_i(\beta^Tx_i + \beta_0) - 1 + \epsilon_i] + C \sum_i \lambda_i\epsilon_i - \sum_i \lambda_i \epsilon_i= \frac{1}{2} \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j -  \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j -  \sum_i\alpha_iy_i\beta_0 + \sum_i \alpha_i - \sum_i \alpha_i \epsilon_i+ C \sum_i \lambda_i\epsilon_i - \sum_i \lambda_i \epsilon_i = -\frac{1}{2} \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j + \sum_i \alpha_i + \sum_i (C -\lambda_i - \alpha_i)\epsilon_i = -\frac{1}{2} \sum_i\sum_j \alpha_i\alpha_jy_iy_jx_i^Tx_j
\end{equation}
s.t: 
\begin{equation}
\alpha_i \geq 0
\end{equation}

\begin{equation}
\lambda_i \geq 0
\end{equation}
 

\begin{equation}
0 \leq \alpha_i \leq C
\end{equation}


\begin{equation}
\sum_i \alpha_iy_i = 0
\end{equation}
This is a quadratic loss function with linear constraints. As such it can be solved by any solver of quadratic programming problems. It is baiscally the same as in the previous case with additional constrints. One important thing to note is that in this scenario, following the KKT conditions it turns out that the support vectors are not just the points that are on the margin but also the points that are on the wrong side of the margin.



```python

```

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
