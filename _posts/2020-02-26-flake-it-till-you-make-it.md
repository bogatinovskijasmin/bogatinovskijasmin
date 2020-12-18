---
layout: post
title: Few words on Neural Networks
subtitle: A broad overview on the topic of Neural Networks
cover-img: /assets/img/neural_network_cover_image.png
thumbnail-img: /assets/img/neural_network_cover_image.png
share-img: /assets/img/neural_network_cover_image.png
tags: [neural network, perceptron, universal approximator, compactness, backpropagation, perceptron, training neural network, RNN, CNN, GAN, GNN, RBM, Transformers, DBNs, Ladder networks, Autoencoders, VAE, MLP]
---
### Neural networks

Neural netowrks are one of the most intereseting learning paradigms. They have one very strong advantage in allowing of learning of arbitrary kernels into their layer-wise strucutre. This in allows exponential decresse in the number of feature that are needed for successfully solving a problem. They are state of the art methods for sequential tasks, images, signal processing and NLP. Many industries and researchers have their focus on developing of new neural network architectures and approaches and application of these methods into various domains. This secition is a small aim to explain what does there exist? What is a neural network? What is a perceptron? Why neural networks are called the unvierasla approximator? Why do they have so called property of compactness? as well several architectures that are most commonly used will be discussed. 

In this section are given:

1) Perceptron

2) Feedforward neural network

3) RNN

4) CNN

5) GNN

6) Transformers

7) Restricted Boltzman Machines

8) Deep Belief Networks

9) Energy-based learning

Also are defined:

1) tricks how to train Neural network (Dropout, batching, vanishing/exploidng gradient, covariate shift, domain adaptation, batch normalization)

2) NN as universal and compact approximator of any function

### Perceptron

Additional linear method that fits linear boundary between two classes is the perceptron method. It is buildin based on the similariity with the neural cells. It is applicable if the classes are linearly separable. That means that there exist a postiive quantity $c$ for which a boundary can be found such that the distance for each point from the decision boundary is at least c. It does not find the optimal decision boundary but just the fisrt one that it finds which the separation error is 0.

The model of the perceptron assumes linear relationship between the input variables $\beta^TX + \beta_o$.At this 

It intution can be build from studing the 2D space of linearly separable classes. For the points from both of the classes three properties can be identified:

1) Let's assume that we are given $x_{0}$ that the point belongs to a decision boundary $\beta^TX+\beta_{o}=0$; Then we can say that $\beta^Tx_{o} = -\beta_{o}$;

2) Let's assume that we are given $x_{o1}$ and $x_{o2}$  that are part of the decision boundary, then $\beta^Tx_{o1} + \beta_o=\beta^Tx_{o2} + \beta_o => \beta^T(x_{o1}-x_{o2})=0$. That means that $\beta$ is normal to the decision boundary. 2

3) The distance between $x_{i}$ and any $x_{o}$ on the decision boundary we have $\beta^T(x_{i}-x_{o})=\beta^Tx_{i}-\beta^Tx_{o}=\beta^Tx_{i}+\beta{o}$. 

We assume that $Y \in \{-1, 1\}$ and $X \in R^{dxn}$. The main idea behind the perceptron is to use the geometry of the space. So a point can be classified either in the correct or in the wrong side of the decision boundary. One merit of how good the classification is can be seen in the distance of a point to the decision boundary (property 3)). That said, one way to define the cost function is to assume it is the sum of the distancs of the missclassified points. Since the distance is always positive, to account for that we additionally multiply by the true label $y_i$ and also with $-1$ if the point is missclassifed. 
The cost function is then defined like:

$J(\beta, X) = \sum_{i \in M} -y_{i}(\beta^Tx_i + \beta_o)$

We can perform gradient descent on this const function with an aim to minimize it. To do this we need the partial derivatives with respect to the parameters $\beta$ and $\beta_o$.
$\frac{\delta J}{\delta \beta}= \sum_{i \in M} -y_{i}x_i$

$\frac{\delta J}{\delta \beta_o}= \sum_{i \in M} -y_{i}$

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
Let's assume we are give the MSE as a cost function where $y^{\*}$ is the prediction of the network and y is the target, then $E = MSE=\frac{1}{2}(y-y^{\*})^2$. We observe 3 layers in the middle of the architecture. The goal is to find the update of the gradient of the weight in the neural network that is the most suited to the update of the weight. Let the layers be denoted with $l,i$ and $j$. At a neuron at each layer we decompose it at two parts $z_i$ and $a_i$. The $a_i=w_{l1}z_{l1} + \dots + w_{lr}z_{lr}$. The $z_i=\sigma(a_i)$ is some nonlinearity applied on the sum of the inputs. Then, 
$\frac{\delta E}{\delta w_{il}} = \frac{\delta E}{\delta a_i}\frac{\delta a_i}{\delta w_{il}}$
one of this partial derivatives is straightforward for computation
$\frac{\delta a_i}{\delta w_{il}} = z_{li}$

The derviative of one of the gradients is quite straight forward since it is the derivative of the activation.
However, for computation of $\frac{\delta E}{\delta a_i}$ we aim to rewrite it in the form that depends on the activation in the next layer. 
$d_i=\frac{\delta E}{\delta a_i}=\sum_j\frac{\delta E}{\delta a_j}\frac{\delta a_j}{\delta a_i} =\sum_j\frac{\delta a_j}{\delta a_i}d_j$. 

$\frac{\delta a_j}{\delta a_i} = w_{ij}\sigma^{'}(a_i)$

This reformulation of the loss in fact means that we can rewrite the gradient for the current activation with respect to gradient of the cost function on the next layer. Assuming that the neural network has a finite width and depth, at one point in time we end up in caluclatiton of the last $d_n$ at layer n. Since we can caluclate that value we can know the exact form of the last $d_n$ and we can backpropagete it so at the end we know the exact form of all of the gradients. 

---------
Algorithm

Step 1) Propagate the data through the network

Step 2) Calculate the last $\frac{\delta E}{\delta u}=d_k$

Step 3) Run backwards the corresponding calcluation for $d_i= \sigma^{'}(a_i)\sum_jw_{ij}d_j$ and obtain those estimates

Step 4) Calculate the derivative of $\frac{\delta E}{\delta w_{il}}=z_{li}d_i$

Step 5) Calculate the updates ot the weithts $w_{ij}<= w_{ij}-\eta \frac{\delta E}{\delta w_{il}}$

Despite this brief summarization of neural networks, we have devoted another chapter for them describing different architectures and their usfullness in various applications. 

### Neural network architectures and some practical issues when training

There exist various neural network architectures. For example, RNN (LSTM, GRU), CNN, GNN, Transformers, Feed-forward neural networks, Mixtutre Density Netowrks, (Restricted) Boltzman machines, Deep Belief networks, energy-based neural networks and so on. They all have some intrinisc properties that makes them applicable for different types of data. Some of them are discussed in the following.

RNNs are type of neural networks that utilize **backpropagation through time** algorithm for training and design the architecture in a way that the so called Markov propertiy is preserved. This allows the neural network to model arbitrary number of states. Thus it is utilized in variuous contexts and applications like sequence to sequence learning or seq2seq, machine translation, time series forcasting and so on. One main issue with RNN is their inability to tackle long term dependancies. To address that problem usually additional mechanisms like input, output, forgeting gates are introduced. That is how LSTM's and GRU's have arised.

One important aspect when training neural network in general is concerned with the problems of **vanishing/exploding gradient**. These are problem that arise in deep networks (multiple layers of neurons) or when modeling complex long-term dependancies. When the gradient has a value slightly smaller then 1 (it can happen due to the activation fucntions that squash the values between e.g sigmoid 0-1), if it needs to propagate through many layers, following the chain rule of probability and writting down the derivatives (e.g $g=0.99$ $n\_layesrs=10$, then this term will have  value close to $0.99^100 \sim 0$ because it is calculated as multiple of derivatives through the layers from the final until first) the bakcpropagated gradient will be very small and will have no influence on the updates. In that case we have the problem of **vanishing gradient**. If the value of the gradient is a bit larger then 1, then we have the problem of **exploding gradient**. 
The problem of exploding gradient can be resolved with clipping the norm of the gradient. This bazically means if the norm of the gradient is above some number, set it to a defualt value. On the other side, the problem of vanishing gradient can be resolved in several ways. One way is to use activation functions such as Rectifier Linear Unit. The inutution is that this kind of nonlinearity does not make the gradient small so fast. In that way one can propagate through many layers with the gradfients still being large enough and the information for the update is not lost. Additionally, it **vanishing gradient** problem can be addressed using **residual connections**. This are special kind of connections that to the output of a neuron add the value given to the input of the neurano $x_{out} = x_{in} + f(x_{in})$. These connections allow the information to float through the network without losing it. Additionally, **batch normalization** allows to elivate this problem. Batch normalization is very important concept since it allows to elivate the **covariate shift** phenomena. Due to the the transformation that happening between the different layers, there can be a distrotion of the distribution. To resolve this one applies **batch normalization**. This is a procedure that provides a normalization for each of the layers $x <- \frac{x-\mu_x}{\sqrt(Var(x))}$, where $\mu_x = \frac{1}{m}\sum_m x_i$ and the $var_x = \frac{1}{m}\sum_m (x_i-\mu_x)^2$. Introudce two parameters $\gamma, \sigma$. The final output is $y<-\gamma x + \beta$. Each of the output is thus changed, by not with change of the parameters of the model. We force theoutput to be some specific one.

Additionally one can use dataset augumentation (e.g add noise to the input samples as one way to improve the generalization). Early stopping and parameter tying may also serve as regularization approaches. 

To further improve the perofromance one can use **Dropout**. This regularization technique defines a Beroullie probability distribution at the output of the layers it is included in. During the training procedure this acts as randomly exluding some of the output neurons making the network less complex during the training. At each forward pass different neurons are dropped. During the test time, each of the corresponding weights associated with the layers where dropout is applied is multipled with the corresponding probability given to Bernoullie. In fact, one can view Dropout as a form of bagging of many simpler networks. The average effect is that the performance can be improved by systainable margin. To provide good performance and preserve larger dependancies, often RNN based methods are augumented with attention mechanism. Attention mechanism can be understand as "smart" averagingin of the inputs given an "identifier". 


Batch size refers to the number of samples before making an update. There is a dependancy between the batch size and the learning rate. Usually large batch size is related to learning rates of moderate value. Smaller batch sizes are related with smaller values for the learning rate. When there are few samples the gradient estimates are very nosiy and in high dimensional space (where we cannot most of the time gurantee Lipschitz properties preserved) the gradient will bounce locally far more. That is why one need to take smaller steps. 

CNN are other type of neural networks that implement a so called convolution layer. Convolution layer is combination of cross-correlations relating multiple input tensor with the outputs. Cross-correlation slides over the input tensor with a kernel-filter and records the corresponding outputs to provide a form of summarization of the input tensor. CNN are performing very well accross most known visual objects (they are universal approach for image recognition), also they are compact in representation. The information can be preserved in finite number of neurons. Although they are not convex usually a CNN can converge to the good optima. They are resource heavy and usually one is approaching with transfer learnign approach for performing good results. 

Transformers are another type of network that implement attention mechanism. More specifically they are implementing the "self-attention" mechanism. This is a specific kind of mechanism identified with three components: key, query and value. The key is used to access a specific value in the given input sequence. The query is used to access a specific value iinside a sequence. The value represents a vector being modified by the corresponding key and query. All this vectors are learnnable. Usually they are stacked into combination of several key, query, value, triplets. They form so called head. Each head attends on specific part of the input and creates the so called multi-head mechanism. Since the transformer implements a feedforward architecture it cannot implement a sequential dependance. To preserve a location within a sequence a special module usualyl is run that implements a series of sines and cosines added to the vector representation of the seuqneces given at the input.
The Transformer arcihteucture is composed of encoder and decoder. The encoder ecnodes the input, the decoder encodes the current output. Additionally the decoder implements a mechanism for the decoder to attend specific parts of the input. This is epseicially important for language translation task. 

GNN are type of neural network that generalize to working with graphs as arbitrary reprsentational structure of images, text, trees and sequences as special cases. GNNs recieve a graph as input. At each layer they implement a propagation steps along the edges of the graph and the corresponding parameters that should be learned. The graph is present and propagated at each layer.

Let's assume that we are given a graph with adjacency matrix A and associated normalzied Laplacian $\lambda_t$. Let $H_0$ represnet some inital state of layer 0 of the GNN. Let W be a set of trainable parameters of the GNN. The layer of $t$ of a GNN consists of two matrix multiplications followed by a nonlinearity: $H_t = g(\lambda_tH_{t-1}W_t)$. Multiplication of $H_{t-1}\lambda_t$ performs a pooling of the relevant messages coming from neighobouring nodes. Multiplication of $H_{t-1}W_t$ extract features that are relevant for the prediction task. The CNNs can be seen of GNNs. 

Mixture Density Networks are type of networks where given the input data one tries to learn conditional probability density. The input of the network are the datapoints, however the outputs are parameters of a gaussian distribution $\mu_i, \sigma_i$.

RBMs is a type of deep-learning method that aims to discover the underlying regularities of the observed data . A Boltzmann machine can be represented as a fully connected network. The restricted Boltzmann machine additionally has the restriction of connections between neurons in the same layer. Usually, the parameters of the network are learned by minimizing contrastive divergence . Stacking of multiple RBMs creates so-called Deep Belief Networks (DBNs). The standard back-propagation algorithm can be used to fine-tune the parameters of the network in a supervised fashion. Using DBNs one can generate new features like a different representation of the data. Those features can be used as input to any multi-label classifier. It is expected that those features are close representatives of the labels being predicted. The hyperparameters of this method are the same as in BPNN, with an additional two: the number of hidden layers and the output multi-label classifier. The novel representation of the input data provided by DBNs can lead to improved performance, on the cost of increased time and space complexity for training the method. 

Energy-based learning is a general framework for perofmring stracutred predictions with neural networks. Like kernel-based structure predictions it assocoiate to each input-output pair a score. The preictied output is the one that yields the highes score (lowest) energy. For more on energy-based learning see the tutorial on LeCunn on "Energy based learning" from 2006.
