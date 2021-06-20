---
layout: post
title: Representation Learning
tags: [representation learning, Principal Component Analyisis (PCA), Kernel PCA, Dual PCA, Supervised PCA, Isomap, FDA, Multi-dimensional scaling (MDS), Local Linear Embedding (LLE), Laplacian Eigenmaps, Maximal Variance Unfolding (MVU), Nystorms approximations (NA), Stochastic Neigbourhood Embeddings (SNE), t-SNE, Canonical Component Analysis (CCA), Independent Component Analysis (ICA), Autoencoders, Variational Autoencoder (VAE), Reucrrent VAE, Beta VAE, PID VAE, Infomax, Hilber Schmid Statistic (HSIC), Fisher Matrix, Natural Gradient, Maximal Mean Discrepancy (MMD), KL divergence]
---
Representation learning is one of the most important aspects when dealing with a machine learning problem. A proper representation of a problem is one of the most crucial aspects for the successful application of machine learning techniques on the given problem. Good problem representation has several advantages. It allows for using simple methods. It results in faster training time and better generalization performance; the total time and memory to store the model are smaller. It is especially important in applications where the resources are limited and should be saved for crucial tasks. Finally, good representation can boost interpretability, distilling knowledge, making insights and improving the decision process in a data-driven manner. Additionally, a compact problem representation can make the process of debugging the models much easier and potentially mitigating existing biases.
The process of feature selection also can be seen as one form of learning representation. In a sense, we learn which features are relevant in the given context and which are not.

In general, there are three ways how to construct a good representation for a given problem. The first approach is by encoding domain knowledge about specific relevant features of the problem (e.g measurements of gene expressions, some physical phenomena etc.) and constructing features (e.g. first derivative in analysis of motion of a solid body). The second is applying various prespecified transformation of the data (e.g. reducing the dimensionality, feature importance etc.), intending to preserve the relevant information in the data. From the latter, we further extend the meaning of the word "applying transformation" to "learning transformations from the data" (most often concerning some loss function). Due to the tremendous progress and impact of the latter in today's machine learning, we consider it as a third way of constructing representation. Therefore, the third way is learning good representations (e.g. by self-supervised training, contrastive learning, or feature extraction with training on an auxiliary task).

Many different concepts for representation learning exist e.g. distributed representations (characteristic for language representation) sparse representation, dimensionality reduction, independent component decomposition, canonical components extraction, pooling of shared information etc. In the following, we are restricting ourselves predominantly to the topic of dimensionality reduction. We split the methods into two parts. The first part refers to classical dimensionality reduction techniques. The latter part is describing deep learning approaches for dimensionality reduction. More specifically, we will focus on autoencoders, recurrent autoencoders and regularized variational autoencoders (e.g. variational autoencoder) as well as some other forms of feature extraction like self-supervised learning. In lines of discussing dimensionality reduction techniques, we will mention one technique for visualization of high dimensional data i.e. tSNE.  

A list of the described methods is given in the following:

1) PCA,

2) Dual PCA (DPCA),

3) Kernel PCA (KPCA),

4) Supervised PCA (SPCA),

5) Fishier Discriminat Analysis (FDA),

6) MultiDimensional Scaling (MDS),

7) Isomap,

8) Local Linear Embedding (LLE),

9) Laplacian Eigenmaps

10) Maximal Variance Unfolding (MVU),

11) Nystrom approximation,

12) t-SNE (Stochastic neighborhood embedding),

13) Canonical Component Analysis (CCA),

14) Independent Component Analysis (ICA),

15) Variational autoencoder

16) $\beta$ VAE

17) Recurrent VAE (RVAE)

18) PID-VAE

To define some of these methods we define the following concepts:

1) Information

2) Entropy

3) KL divergence

4) HSIC statistic

5) Spectral Clustering

6) Cut and ratiocut

7) Maximal mean discrepancy (MMD)

8) Infomax principle

9) Fisher matrix

10) Natural gradient

11) Laplacian of a graph


## Dimensionality reduction

One instance of representation learning is the dimensionality reduction aspect. As a task, it belongs to the unsupervised learning paradigm. However, there exist methods that can introduce information from the labels to produce better representations.

In general, the methods for representation learning can be grouped into three groups:

1) methods that aim to preserve maximal variation in the data with reduced dimensionality;

2) methods that try to add sparsity in the representation. These techniques increase the representation dimensions making large parts of the data being sparse.

3) methods based on statistical properties (like independence testing as in Independent Component Analysis)

In this post, we predominantly consider methods within the first type of representations. There are two main directions to provide the goal of preserving the maximal variations of the data. The "classical" approach and "modern" deep learning approaches. Although this distinction is arbitrary,  as we shall see later, many of the "traditional" approaches can be cast in terms of kernel PCA (either with the predefined or learned kernel). On the other side, the DL approaches (e.g regularized autoencoders with nonlinear activations) can be seen as an equivalence function class as that of kernel PCA. Effectively, it allows drawing some parallels between the traditional approaches as some special cases of the deep learning methods like the autoencoder framework. The latter is one prominent representative of representation learning approaches from the DL paradigm. Take note that the traditional approaches and autoencoder based approaches are not the same. A disclaimer, deep neural networks can be seen as automatic feature extractors and are automatically "learning" representations.

Before explaining the traditional approaches for dimensionality reduction, we will introduce some details and refreshers for the programming machinery that hides all the tedious low-level abstraction. The first group of techniques predominantly relies on linear algebra techniques. These approaches usually rely on some form of a similarity measure. Utilizing different matrix factorization ( e.g: SVD, QR, LU etc.) on the top of the similarity matrix, kernel or distance matrices, many techniques are developed. It is interesting to observe that these approaches can combine both the descriptive and target attributes. Thus the methods can be both supervised and unsupervised. Most of these approaches are using SVD decompositions. That is why we first consider some basic definitions from linear algebra, some motivation and two simple implementations of methods for extracting eigenvectors and eigenvalues of matrix implemented without using functions from e.g LAPACK.

It is very important to take note that dimensionality reduction is only possible in case when the kernel K has a rank(K) much less than the dimensionality of the data. Otherwise does not make sense to examine the problem. Geometrically, this means that the data live in some submanifold of the currently observed one.

**Note** Interesting resources for the implementation of QR decomposition of a matrix from scratch in Python [https://www.quantstart.com/articles/QR-Decomposition-with-Python-and-NumPy/](https://www.quantstart.com/articles/QR-Decomposition-with-Python-and-NumPy/).

### Singular Value Decomposition

Singular Value Decomposition (SVD) is a matrix decomposition method. It decomposes the matrix into three matrices. Two of them are orthogonal, while one of them is diagonal. The values of the diagonal matrix are called singular values.
$ X = SDV^{T},   X \in C^{nxm}, S \in C^{nxn}, V \in C^{nxm}  D* \in C^{mxm}$, where $C$ is the set of complex numbers in the most general case. We are interested in situations where X is part of the real number set of numbers because the data usually come in that form. Geometrically one can interpret the eigenvectors as rotation axis that transform a unit circle into an ellipse with a radius determined by the corresponding eigenvalues for the corresponding eigenvector.

The calculation of SVD in most programming languages is using the LAPACK library. It is a Fortran library for linear algebra. The core and main advantages of this library are that it allows for efficient running on shared-memory vectors and parallel processing. LAPACK organizes the algorithms to use block matrix operations. These block operations can be organized to account for the hierarchical organization of the memory in the machines, therefore, producing optimal performance. For more read: [http://www.netlib.org/lapack/](http://www.netlib.org/lapack/) and [https://en.wikipedia.org/wiki/Eigenvalue_algorithm](https://en.wikipedia.org/wiki/Eigenvalue_algorithm).

Various algorithms can be used for performing the SVD decomposition:

**1)** Golub–Reinsch SVD

**2)** High Relative Accuracy Bidiagonal SVD

**3)** Divide and Conquer Bidiagonal SVD

**4)** Biorthogonalization SVD

**5)** Jacobi Rotation SVD

More details can be found: [http://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf](http://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf)

If the matrix is Hermitian (complex matrix X is equal to its conjugate transpose), there are more efficient methods that can be used to calculate the decomposition. Similar optimizations can be made for different constraints on the matrix form.

**Note**: Numpy implementation of SVD (a wrapper around Fortran's LAPACK) additionally performs sorting operation to make the eigenvalues from descending to ascending order.

____________________________________________________
##### Few notes on LAPACK
LAPACK is written in Fortran 90 and provides routines for solving systems of simultaneous linear equations, least-squares solutions of linear systems of equations, eigenvalue problems, and singular value problems. The associated matrix factorizations (LU, Cholesky, QR, SVD, Schur, generalized Schur) are also provided. Dense and banded matrices are handled, but not general sparse matrices. In all areas, similar functionality is provided for real and complex matrices, in both single and double precision.

____________________________________________________
### Eigenvalue decomposition and the (Inverse) Power methods
SVD decomposition can be seen as a generalization of the eigenvalue decomposition of a matrix. The **eigenvalue** decomposition of a matrix holds for **square matrices**, while SVD does not have that assumption. Furthermore, if the matrix is positive-definite its eigenvalues are all real numbers. **Orthogonal matrix** is a matrix where each of the columns vectors are orthogonal vectors.  **Orthogonal vectors** have their dot product equal to 0, meaning they are normal vectors. **Orthonormal vectors** are orthogonal vectors with norm 1. **Norm** is a function that maps a vector from dimension $R^{d}$ to $R$.


We implement the **power method** for calculating the eigenvector with the highest eigenvalue. A very nice explanation can be found here: [https://www.youtube.com/watch?v=OzeDqsVoTFc](https://www.youtube.com/watch?v=OzeDqsVoTFc).


--------------------------------------------------------------
**INPUT**: A matrix of $X \in R^{nxn}$, convergence_threshold


**Step 1.** initialize u = numpy.ones(n, 1)

**Step 2.** iterate until convergence

  **Step 2.1.** update $u_{k+1}$ = $\frac{Au_k}{\|\|Au_k\|\|_{2}}$

**Step 3.** max_eigenvector = u

**Step 4.** max_lambda = $\|\|Au\|\|_{2}$

**OUTPUT** max_eigenvector, max_lambda

-----------------------------------------------------------
To calculate the remaining eigenvectors we can use power method on $A^{h}= A - \lambda_{1}w_{1}w_{1}^{T}$ and so on. This procedure is known as deflation.


```python
def power_method(A, eps):
    m = A.shape[0]
    v = np.ones(m)
    cnt = 0
    while True:
        AU = A@v
        u = np.divide(AU, np.linalg.norm(AU))
        if np.linalg.norm(u-v) < eps:
            return u, np.linalg.norm(AU)
        else:
            v = u
            cnt += 1

```

Sometimes we are faced to produce the eigenvector corresponding to the smallest eigenvalue. The **power method** returns just the eigenvector corresponding to the largest eigenvalue. To achieve our goal of finding the eigenvector corresponding to the smallest eigenvalue we use the **inverse power method**. The pseudo-code is given in the following:


--------------------------------------------------------------
**INPUT**: A matrix of $X \in R^{nxn}$, convergence_threshold


**Step 1:** initialize u = numpy.ones(n, 1)

**Step 2:** calculate $A^{-1}$

**Step 3:** iterate until convergence

**Step 3.1:** update $u_{k+1}$ = $\frac{A^{-1}*u_k}{\|\|A^{-1}u_{k}\|\|_{2}}$

**Step 4:** min_eigenvector = u

**Step 5:** min_lambda = $\|\|Au\|\|_{2}$ # Take a note that we are using the original matrix A isntead of the inverse

**OUTPUT** min_eigenvector, min_lambda

-----------------------------------------------------------


```python
import numpy as  np
def inverse_power_method(A, eps):
    m = A.shape[0]
    v = np.ones(m)
    cnt = 0
    A_orig = A
    A = np.linalg.inv(A)
    def calculate_lambda(u, A):
        return np.linalg.norm(A@u)
    while True:
        AU = A@v
        u = np.divide(AU, np.linalg.norm(AU))
        if np.linalg.norm(u-v) < eps:
            return u, calculate_lambda(u, A_orig)
        else:
            v = u
            cnt += 1



X = np.array([[10, 20, 40], [50, 17, 244], [1, 2, 10]])
eps = 0.0001


u_largest, lamb_largest = power_method(X, eps)
u_smallest, lamb_smallest = inverse_power_method(X, eps)
print("The eigenvalues are {} {}".format(lamb_smallest, lamb_largest))
print("The eigenvectors are {}".format(np.array([u_smallest, u_largest ]).T))
print("\n")
print("Output from sklearn \n", np.linalg.eig(X))
```

    The eigenvalues are 4.0746666767039335 55.107754667882745
    The eigenvectors are [[-0.97086049  0.44120115]
     [-0.12332233  0.89604137]
     [ 0.20547873  0.04951163]]


    Output from sklearn
     (array([ 55.10431606, -22.17906212,   4.07474607]), array([[ 0.44122479,  0.49229127, -0.97085933],
           [ 0.89602962, -0.86956769, -0.12332992],
           [ 0.04951375,  0.03874706,  0.20547966]]))


## Principle Component Analysis (PCA)

Principle component analysis is a dimensionality reduction technique. To be applied as a dimensionality reduction technique, it assumes that the data lie on a linear manifold.
It provides a transformation function applicable for out of sample estimates.

The main idea is to find vectors that can transform the data in a way that the preserved variance in the data will be maximized. We refer to these vectors as principal components. The first principle component maximizes the variance of the overall data, the second principle components maximize the projection of the data on the first component and so on.

Let's assume that we are given $X \in R^{d}$ and vector w.
Following a simple rules from basic probability course we know that given a random variable x with mean $mean(x)$ and variance $var(x)$, if we multiply its mean by a constant c, then the mean transforms as $mean(cx)=c\*mean(x)$, while the variance (sample covariance matrix) transforms as $VAR(cx)=c^{2}\*VAR(x)$. Similalry, for X being mutlivariate we have, $mean(wx)=w\*mean(x)$, while for the variance we have $VAR(wX)=w^{T}\*VAR(x)\*w$.

Since the definition of the first principle component given as maximization problem, following the previously described rule, we can write the problem as $argmax_{w} w^{T}VAR(X)w$.
The latter is an ill-constrained problem, from an optimization perspective because it does not poses constrain on the w. However, we are just interested in finding the direction of the maximal variation of the data. Therefore, any arbitrary constraint, for example on the magnitude on the vector $w$, on the vector $w$ will suffice.

Thus the PCA problem can be defined as follows:

\begin{equation}
    argmax_{w} w^{T}VAR(X)w
\end{equation}
\begin{equation}
    w^{T}w = I
\end{equation}

To solve it we can adopt any optimization strategy e.g. utilizing the Lagrangian method.

\begin{equation}
    L(w, \lambda) = w^{T}VAR(X)w + \lambda(I-w^{T}w)
\end{equation}
\begin{equation}
    \frac{\delta{L(w, \lambda)}}{\delta{w}} = VAR(X)w - \lambda w = 0
\end{equation}
\begin{equation}
    VAR(X)w = \lambda w
\end{equation}

It is a standard eigendecomposition problem. The optimal solution for $w$ is the eigenvector corresponding to the largest eigenvalue, at the same time, it preserves the direction of maximal variation in the data. One can prove this by replacing the last equation with the optimization function and replacing the constraint $w^{T}w=I$. The mean can be denoted as $mean(X)=\mu_{x}$. The $k$ represents the number of dimensions to preserve the $X$.


### The algorithm for PCA is as follows:
--------------------------------------------------------------
**INPUT**: A matrix of $X \in R^{dxn}$, integer $k<d$, dimensions to preserve


**Step 1:** X = $\sum_{i=1}^{n}({x_{i}}-\mu_{x})$

**Step 2:** $C = \frac{1}{N}XX^{T}$

**Step 3:** U, $\lambda$ $V^{T}$ = $SVD(X)$ or $EIG(C)$, prefer former for stability

**Dimensionality Reduction** $Y_{reconstruction}=U_{:k}^{T}X$

**Anomaly Detection** $anomaly score(x_{i}) = \|\|U_{k:}^{T}x_{i}\|\|_{2}^2$

**Noise Removal** $X_{noise reduced} = U_{:k}Y_{reconstruction}$ + Undo Centering

**OUTPUT** $Y_{reconstruction}$ or $anomaly score(x_{i})$ or $X_{noise reduced}$

-----------------------------------------------------------

PCA has many different applications. Despite dimensionality reduction, it can be used for noise removal, anomaly detection, face recognition using eigenfaces etc. A strong point of this "linear" version of PCA is its ability to produce mapping for the unknown test sample and to reconstruct original samples. A weak point of PCA is that it is not robust to outliers in the data. The outliers significantly contribute to the variance;  PCA assumes that the data lie in a linear manifold.

The observation of why the SVD decomposition of X is sufficient resides in the fact that $U$ is eigenvectors of $XX^{T}$,  $V$ is eigenvectors of $X^{T}X$ and the eigenvalues are the square roots of $XX^{T}$. One prefers SVD decomposition before eigenvalue decomposition because the former is numerically more stable to compute.



### Dual PCA

If one is familiar with kernels, one can immediately recognize that $X^{T}X$ is a linear kernel calculated on data $X \in R{dxn}$. This allows us to define a whole family of approaches that are closely related to PCA or are its generalization e.g dual, kernel and supervised PCA.
Dual PCA is beneficial in the case where $d>n$. The advantage resides in the fact that we perform singular value decomposition on a smaller matrix. Utilizing the fact that it is cheaper to calculate the eigenvectors of $X^{T}X$ in the described step we aim to rewrite the PCA algorithm.

We start by replacing $U$ with an expression that depends on $X$, $\lambda$ and $V$.


\begin{equation}
X = U \lambda V^{T} /V
\end{equation}
\begin{equation}
XV = U \lambda /\lambda^{-1}
\end{equation}
\begin{equation}
XV\lambda^{-1} = U
\end{equation}

To project data in p-dimensional space (we assume that we are working with the truncated matricies):
\begin{equation}
U^{T}/ X = U \lambda V^{T}
\end{equation}
\begin{equation}
Y = U^{T}X = \lambda V^{T}
\end{equation}

To reconsturct training data:
\begin{equation}
X = UY
\end{equation}

replacing the above equations:
\begin{equation}
X_{rec} = XV \lambda^{-1}\lambda V^{T} = XVV^{T}
\end{equation}

for one point $x$ out of sample projection:
\begin{equation}
y = U^{T}X = \lambda^{-1}V^{T}X^{T}x_{new}
\end{equation}

for one point $x$ out of sample reconsturction:
\begin{equation}
x_{rec} = UY = XV\lambda^{-1}\lambda^{-1}V^{T}X^{T}x_{new}
\end{equation}
note $\lambda$ is a matrix of eigenvalues of the sample coveriance matrix. As such it is a diagonal matrix (real-valued) that has an inverse.

### Kernel PCA and relation to auto-encoders

Kernel PCA is another method for dimenisionality reduction. It addresses the limmitation of PCA that assumes that the data exist on a linear manifold (or a subset). To do this it adhers to the "kernel trick". See details in the kernel section. In short, the kernels are functions, that satisfy the symmetry and the positve-semidefinitness properties (Mercer condtions) and their value corresponds to the dot-product of the mapping of the input arguments in some, usually higher order dimension.

\begin{equation}
K(X, X) = \phi{(X)}^T \phi{(X)}
\end{equation}

The main benefit is that we are not required to know the exact form of $\phi{(X)}$, that in the most general case can be a mapping to inifite dimensional space (e.g Gaussian kernel). Thus, there does not exist any inverse mapping (or reverse image). The kernel can be seen as similarity (recall cosine similariy) between two functions of our input in arbitrary dimensional space. In such high dimensional space the "curse of dimensionality" becomes "bless of dimensionality" since the poitns are easly separable (e.g a linear classifier can be very effective since almost everything is linear there).

To derive the Kernel PCA method we can refer to the Dual PCA and whenver we have X we will replace it with $\phi{(X)}$.


\begin{equation}
\phi{(X)} = U \lambda V^{T} /V
\end{equation}
\begin{equation}
\phi{(X)}V = U \lambda /\lambda^{-1}
\end{equation}
\begin{equation}
\phi{(X)}V\lambda^{-1} = U
\end{equation}

To project data in p-dimensional space (we assume that we are working with the truncated matricies):
\begin{equation}
U^{T}/ \phi{(X)} = U \lambda V^{T}
\end{equation}
\begin{equation}
Y = U^{T}\phi{(X)} = \lambda V^{T}
\end{equation}

To reconsturct training data:
\begin{equation}
\phi{(X)} = UY
\end{equation}

replacing the above equations:
\begin{equation}
\phi{(X)}_{rec} = \phi{(X)}V \lambda^{-1}\lambda V^{T} = \phi{(X)}VV^{T}
\end{equation}

for one point $x$ out of sample projection:
\begin{equation}
y = U^{T}\phi{(X)} = \lambda^{-1}V^{T}\phi{(X)}^{T}\phi{(x_{new})}
\end{equation}

for one point $x$ out of sample reconsturction:
\begin{equation}
x_{rec} = UY = \phi{(X)}V\lambda^{-1}\lambda^{-1}V^{T}\phi{(X)}^{T}\phi{(x_{new})}
\end{equation}

Recalling that in most general form we do not know what is the mapping $\phi{(X)}$, it is pretty obvious that not all steps from the Dual PCA are possible. We can project a new point to the $p-$dimensional space and can provide out of sample reconstruction. However, we cannot project back any training point back to the orignal space, nor can recounstruct out of sample point back, because they explicitly involve the mapping $\phi{(X)}$ not their dot-product. Again as in the case of Dual PCA, we need to calculate the right eigenvectors and the eigenvalues of the kernel matrix. Additional catch one should take in care is that it needs to centralize the kernel data in the kernel space (an operation that boils down to summation and substraction of kernels). The summation of two kernels is again a kernel function so we do not have any problems there.


###### Autoencoders

One interesting thing that can be also observed with such formulation of the problems, we will come later back to is the definition of an algorithm like PCA (and its variants) as an optimization problem in the following form:
\begin{equation}
Y = U^{T}X
\end{equation}
\begin{equation}
X = UY = UU^{T}X
\end{equation}
\begin{equation}
min_{u} ||X - UU^{T}X||_{2}^{2}
\end{equation}
or more in the nonlinear case where the data are transformed
\begin{equation}
min_{u} ||\phi{(X)} - UU^{T}\phi{(X)}||_{2}^{2}
\end{equation}

NOTE: PCA and autoenocders do not result in the same solution, one can more think of this as very similar methods, to better understand the autoencoders.

The last equation allows to define the arbitrary transformations U such that it closly resambles the Kernel PCA algorithm, however the transformation U is not always the same. Moreover, the last equation we can rewrite it in plain English as:
\begin{equation}
min_{u} ||\phi{(X)} - decode(encode(\phi{(X)}))||_{2}^{2}
\end{equation}

Thus at the final end this results in an encode-decoder structure, or we refer to as autoencoder. One can imagine
that the encoder is a neural network architecture as well as the decoder. With their joint optimization (e.g via backproapagation) one can solve this problem and learn a very powerful representations. The autoencoder is a powerful mechanism for dimensionality reduction and learning representations.

In kernel terms it corresponds to learning of arbitrary kernels instead of prespcifiying them as in the case of kernel PCA.

Very often adding additional regularization terms, either in terms of KL divergence between the reconstruction and arbitrary distribution, or $L_{1}$ or $L_{2}$ norms of the weights or playing with the size of the inner representation or adding a bit Gaussian or Laplacian noise to the inputs, one can create powerful representations. We will later recall some of this stractures, especially Variational Autoencoder (and its sequential companion Recurrent Variational auto-encoder), and the $\beta$-autoencoder, which are especially interesting and one of major building blocks of the research in the time of writting.

There are furthermore: suffcient dimenisonality reduction, where the goal is to find $P(y|x) = P(y|u^{T}x)$. Or metric learning where the goal is to find the semi-positive definitiness matrix A such that it is optimized some criteria involving the Mahalanobis distance.


# Supervised representation learning techniques
While there exist various unsupervised learning techniques, there are also tehniques that can exploit a supervised information while aimiing to produce better representation. From all of the supervised techniques we will discuss:

1) **Supervised PCA**

2) **Fisher Discirminat Analysis**

## Supervised PCA
To introduce Supervised PCA we first need to introudce the HSIC statistics or (Hilbert Schmit Independence Criteria).

**Hilber-Schmidt norm and HSCI statitsc**

To build some intuition for **HSIC** we start with an example. Imagine that we are given two univeariate random variables with same mean and different variance. This means that their first moment is the same, while the second moment of the distribtuion is different. If we append those random variables with their squares and again compare them by their first and second moment, of the new bi-variate random variables,  we can see that they are different on both the first and the second moments of the distributions. This is one part of the intuiton for HSIC.

Second important part of HSCI is the definition of independence of two random variables. **Two random variables are independent if any bounded continious function of those random variables are independent.** This is important assumption since combining with the first one intuition we can say "Imagine that there exist a mapping that caputre all of the bounded functions (similalry as in the case with the example when all moments are preserved, e.g the infinite RBF kernel), then measuring the correlation between the mappings will result in measure of dependence. To capture the **dependence** in one number we calculate the **norm** of the matrix as one number desribing the matrix. If this number is large there is dependence between the variables, otherwise they are independent". This is not entierly true in mathematical sence, since we define cross-covariance operator instead of correlation matrix (the RBF kernel corresponds to infinite mapping), however it is sufficient for intuition. HSIC norm is similar to the Frobenious norm but it does not apply to vectors instead to the **mappings**.

**MMD (maxiamal mean discrepancy)**.
Third important considiration for HSIC is related to the maximal mean disrepancey as a distance measure between probabilties functions of two random variables.  It is given as:
\begin{equation}
MMD(X, Y)=||\frac{1}{n}\phi(X) - \frac{1}{m}\phi(Y)||_{2}^{2} = (\frac{1}{n}\phi(X) - \frac{1}{n}\phi(Y))^{T}(\frac{1}{n}\phi(X) - \frac{1}{m}\phi(Y))= \frac{1}{n^2}\sum_{i,j}(\phi(X_i)^{T}\phi(X_j)) + \frac{1}{m^2}\sum_{i,j}(\phi(Y_i)^{T}\phi(Y_j))-\frac{2}{mn}\sum_{i,j}(\phi(X_i)^{T}\phi(Y_j))
\end{equation}

where X and Y are random variables obtained from n observations drawn from $p(x,y)$. This is in fact a metric distance between two distributions that satisfy all the properties of a metric. It is used in the same context as KL divergence but the later does not have the metric properties, hence it is divergence.


Combining this 3 preqrequsists one can derive the **Hilber-Schmidt Independnce Criteria (HSIC)** as testing for independence between two random variables:
\begin{equation}
p(x, y) = p(x)p(y)
\end{equation}
Using MMD we can derive the expression for HSIC:
\begin{equation}
||p(x, y) - p(x)p(y)||_2^2
\end{equation}
\begin{equation}
HSIC(X, Y) = \frac{1}{(n-2)^2}Tr(KHBH),
\end{equation}
where $K,H,B \in R^{nxn}$ $K_{i,j}=k(x_i, x_j), B_{i,j}=b(y_i, y_j)$ and $H=I-\frac{1}{n}ee^T$ and k and b are positive semi-definite kernels. H is a centring matrix. One can center just the right or just the left kernel it does not matter. HSIC is a measure of the dependence of the random variables.

The intuiton behind the HSIC is that we can know the correlation between two random variables if we measure the norm of the covariance matrix. This norm is the Hilber-Schmidt norm. Furthermore, it is important that we pick kernel such as  **RBF** that allows comparison accross all of the moments of the distributions. Small values for the norm indicate independce between the variables since HSCI is a measure of dependence. Large values for the norm indicate dependence between the variables.

#### Supervised PCA (SPCA)
Input $\{(x_i, y_i)\}$ where $x \in R^{d}$ and $y \in R^{q}$, and the HSIC formula. We define:

**GOAL:** The goal is to find a mapping $U^TX$, such that $U^TX$ has the maximal **dependency** to Y.

1) Make a linear kernl on $U^TX$ which is $K=X^TUU^TX$ we make

2) Make a kernel B over Y

3) Write HSIC(K, B) = $\frac{1}{(n-2)^2}Tr(X^TUU^TXHBH)$, just U is uknown and we maximaze HSIC with regard to U.

4) Objective:
\begin{equation}
argmax_{u}Tr(X^TUU^TXHBH)
\end{equation}
\begin{equation}
argmax_{u}Tr(U^TXHBHX^TU)
\end{equation}
add constrain of form
\begin{equation}
U^TU=I
\end{equation}
it is very easy to optmize this problem. Similar like in PCA, write the Lagrangian dual form, optimize and the values for U are the eigenvectors of the matrix $eig(XHBHX^T)$ (XH = x-$\mu_{x}, -> XHIHX^T$ is the covarinace matrix). Interestingly if one set $B=I$, it results into oridnary PCA prbolem as a special case of supervised PCA.


The previous algorithm for is written in its linear form. However, we can use the kernel trick and rewrite it in a kernelzied form as follows. Replace
\begin{equation}
U = \phi(X)\beta
\end{equation}

\begin{equation}
argmax_{u}Tr(\beta^T\phi(X)^T\phi(X)HBH\phi(X)^T\phi(X)\beta)
\end{equation}
subject to \begin{equation}
\beta^T\phi(X)^T\phi(X)\beta=I
\end{equation}

\begin{equation}
argmax_{u}Tr(\beta^TK(X, X)HBHK(X, X)\beta)
\end{equation}
subject to \begin{equation}
\beta^TK(X, X)\beta=I
\end{equation}

The solution of this problem is generalized eigen decomposition. So to calculate the $\beta$'s one should perform eigendecomposition on the cross-covariance operator $HBHK$. After the calculation of the parameters all operations as followed in the KPCA can be applied.

## Fisher Discriminat Analysis (FDA)
Another supervised approach for dimensionality reduction recides in the method of Fisher Discriminant Analysis (FDA). This method tries to find a projection of the data such that the distance of the means of the lower dimensional projection of the data is maximized, while the within variation of the projection is minimized.

Recall that: $\mu(cx)=c\mu(x)$ and $VAR(cx)=c^2VAR(x)$, $Tr(a)=a$ and $||*||$ is a scalar.

The previous descripiton of the FDA approach can be written as:
\begin{equation}
||w^T\mu_{x1} - w^T\mu_{x2}||_2^2 = (w^T\mu_{x1} - w^T\mu_{x2})^T(w^T\mu_{x1} - w^T\mu_{x2}) = \mu_{x1}^Tww^T\mu_{x1} - 2\mu_{x2}^TTww^T\mu_{x1}  + \mu_{x2}^Tww^T\mu_{x2}
\end{equation},
where $\mu_{x1}$ and $\mu_{x2}$ denote the means of class 1 and class 2.
\begin{equation}
\sigma_{b} = Tr(||w^T\mu_{x1} - w^T\mu_{x2}||_2^2) = Tr((w^T\mu_{x1} - w^T\mu_{x2})^T(w^T\mu_{x1} - w^T\mu_{x2})) = Tr(\mu_{x1}^Tww^T\mu_{x1} - 2\mu_{x2}^TTww^T\mu_{x1}  + \mu_{x2}^Tww^T\mu_{x2}) =
Tr(w^T\mu_{x1}\mu_{x1}^Tw - 2w^T\mu_{x1}\mu_{x2}^TTw  + w^T\mu_{x2}\mu_{x2}^Tw) = Tr(w^T(\mu_{x1}-\mu_{x2})(\mu_{x1}-\mu_{x2})^Tw) = Tr(w^T\sigma_{between}w)
\end{equation},

$\sigma_{between}$ is a rank 1 matrix in the k=2 class dimensionl case, or k-1 in k dimensional case.For the covariance matrix we have:
\begin{equation}
\sigma_{total}=w^T\sigma_{x1}w + w^T\sigma_{x2}w  = w^T(\sigma_{x1} + \sigma_{x2})w  = w^T\sigma_{within}w
\end{equation}

Following this formulation we define an optmization problem of the following form:
\begin{equation}
argmax_{w}\sigma_{b}/\sigma_{total}
\end{equation}

This is ill-defined optimization problem since it miss on the constrain. We need to define a constrain. Utilizing Ryglihg-Cauchy optimization procedure we can define the problem in the following form:
\begin{equation}
argmax_{w}\frac{\sigma_{b}}{\sigma_{total}}
\end{equation},
subject to
\begin{equation}
w^T\sigma_{within}w = I
\end{equation}
We can write the Lagrangian from here and obtain: $\sigma_{within}^{-1}\sigma_{b}w = \lambda w$. The solution for $w$ is again an eigenvector decomposition of the general eigenvalue problem, which we already saw how can be computed. The idea is that now when building the feature vectors the information for the target is also included because the mean and the sample covarinace matricies are calculated per class. The overall discussion can be applied and for K classes. It is important there to know that the between class covariance matrix is calculated as a difference of the total covariance matrix and the sum of the witihn covariance matricies for each of the k classes.

# Again Unsupervised
## Multi-dimensional scaling (MDS)

MDS is an approach for lower dimensional embedding. It requires just a pairwise distance computation between the points. It tries to find the lower dimensional representation such that the distances in the original space are preserved.

\begin{equation}
cost= \sum_{i<j} ||d_x(i, j)-d_y(i, j)||_2^2
\end{equation}
where d(i, j) is the pairwise distances in the original space $ X \in R^d$, while $Y \in R^p$ where $p<d$.

The algorithm goes as follows:

Input D(X) pairwise distances of X

**Step 1** Random initialization of points from $Y \in R^p$

**Step 2** until convergence do

**Step 2.1** Calculate distances between the target Y

**Step 2.2** update y with a gradient step calculated from the cost function

One can also obtain the soludion for MDS using eigenvalue decomposition where: $Y = \lambda^{-0.5}V^T$. This is identical to the solution of dual PCA where instead of signaluar values of the diagonal we have their square root. The square root of eigenvalues correspond to the singular values. V is the eingevectors of $X^TX$. This method is also linear. We need double centring of the X^TX.

Additional versions can involve normalization like Sammons mapping. This kind of mapping can preserve the structure in higher dimensioal space which highly depdnts on the implemented pairwise distance measures. It converges to PCA if the data is in linear manifold.


## Isomap

Isomap is MDS where insetead of Euclidean distance we are using Geodesic distance. This method assumes that the data lie on a manifold. First we construct a graph of all of the points. The distance between point is defined as the minimal critical distance between the points in the graph.
We refer to this distance as Geodesic distance since it takes into account the structure of the graph. This method works by projecting the geodestic distance into a lower dimensionl space using the geodesic distance.

-------------------------
Input: X, k


**Step 1** Construct a k-nearest neighbour graph on n data points $X \in R^{dxn}$

**Step 2** Compute shortest path between all points as estimation $D^{g}(X) = geodesic_distance(X)$

**Step 3** $K = -0.5HKH$, H is centring matrix $H = I - \frac{1}{n}ee^T$

**Step 4** V, $\lambda$ = eig(K), V is eigenvectors of K, $\lambda$ are eigenvalues of K

**Step 5** Y = $\lambda^{\frac{1}{2}}V$ in p-dimensions

**Step 6** $Y_{new} = U^TG(X_{new})$ NO!!

As it can be seen by Step 6 this method is usefull for the representation of the training data. It is not able to provide out of sample estimates since it requires construction of the graph of all points to find the neighbours of the $X_{new}$.

One important detail is that K may not be positve-semidefinite matrix. Thus there is no gurantee that one can decompose the K matrix to positive eigenvalues. To solve this one needs to map the K matrix to a cone of a semi-positive matrix e.g via applying $abs(K(X))$.

It is very important for the graph to be connected so the distances can be calculated correctly.

## Local-linear embedding (LLE)
LLE is another approach for dimensionality reduction. It assumes that the data locally lie on a subspace. It tires to capture the locallity properties of the manifold given the data and then reconsturcts the same locallities in the lower-dimensional space.

It assumes that each point is linearly related with $k$-neighbouring data points. As such it allows to calculate the linear dependency of a particular point to the others. Afterwards, tries to find a subset of points of smaller size that adhere to the same local linear properties as the original subspaces, patches, in the original space.
The goal in both cases is to minimize the corresponding functions in sequential order.

\begin{equation}
J_{original}=\sum_{i}||x_{i}-\sum_{j}^{k}w_{i,j}x_{i}||_2^2
\end{equation}

\begin{equation}
J_{embedding}=\sum_{i}||y_{i}-\sum_{j}^{k}w_{i,j}y_{i}||_2^2
\end{equation}
where $x\in R^d$ and $y \in R^q$ $d>q$.

After the $w$ weights are obtined with simple linear regression fits, then the solution for $y$ can be done using an iterative gradient descent method. However, as a second effective solution one can rewrite the two cost functions and come to an elegant solution for the calculation of the optimial representations as the eigenvectors corresponding to the smallest eigenvalues of a specifically computed matrix which derivation and overall solution of the problem is given in the following.

_________________________
Input X, k

**Step 1** Construct a k-nearest neighbour graph

**Step 2** Compute $W_{ij}$ via minimization $\sum_{i}||x_{i}-\sum_{j}^{k}w_{i,j}x_{i}||_2^2$ for all points i. to be a well defined problem we further assume $\sum_{i,j} w_{i,j}=1$

**Step 3** Compute $Y_i$ via minimization of $\sum_{i}||y_{i}-\sum_{j}^{k}w_{i,j}y_{i}||_2^2$

We define the following matricies $V_i$ as all neighbours of point $i$:
\begin{equation}
V_i = [x_{i1}, x_{i2} ... x_{il}] \in R^{dxk}
\end{equation}

\begin{equation}
e_i = [1;1 ... 1] \in R^{kx1}
\end{equation}

\begin{equation}
w_i = [w_{i1}; ... w_{ik};] \in R^{kx1}
\end{equation}

\begin{equation}
x_i = x_ie_i^Tw_i
\end{equation}

It is very important for the graph to be connected so the distances can be calculated correctly. Small k results in disconnected graph. Large k results in linear method since we no longer have locallity. Same as ISOMAP. Appropriate choice of k is mandatory.

Using the above definition of the matricies we rewrite them in the following way:

\begin{equation}
min_{w_i}||x_ie_i^Tw_i-V_{i}w_{i}||_2^2 = min_{w_i}w_i^T(x_ie_i^T - V_i)^T(x_ie_i^T - V_i)w_i = min_{w_i}w_i^TGw_i
\end{equation}
s.t
\begin{equation}
e^Tw_{i}=1
\end{equation}
The solution can be done with writting the Lagrangian:
\begin{equation}
w_{i}=\frac{\lambda}{2}G^{-1}e
\end{equation}
as long as $\sum{w_i}=1$ the value for $\lambda$ does not matter.



\begin{equation}
Y_i = [y_{i1}, y_{i2} ... y_{il}] \in R^{pxn}
\end{equation}


\begin{equation}
I = diag(1) \in R^{nxn}
\end{equation}

\begin{equation}
I_{:i} = [0;0 ... 1 ... 0;0] \in R^{nx1}
\end{equation}

\begin{equation}
W = [w_{1} ... w_{n}] \in R^{nxn}
\end{equation}

\begin{equation}
w_{i:} = [0; 0; ... w_1; w_2; ... w_k; ... 0] \in R^{nx1}
\end{equation}

We aim to rewrite the optimization function with matricies, utilizing the following substitutions.
\begin{equation}
y_{i} = YI_{:i}
\end{equation}

\begin{equation}
\sum_{j} w_{ij} y_{j} = YW_{:i}
\end{equation}

in the second optimization function of the problem and following the fact that the Frobenious norm can be written as $|A|_F^2 = Tr(AA^T)$

\begin{equation}
\min_{Y} \sum_i^n|YI_{:i}-YW_{:i}|^2 <=> \min_{Y} |YW - YW|^2 <=> \min_{Y} |Y(I-W)|^2 <=> \min_{Y} Tr((Y^T(I-W)^T(I-W)Y)
\end{equation}
, since it is ill defined problem we need to define it. That is why we add a constrain of the form $YY^T=I$

The solution of the optmial values for Y are the eigenvectors corresponding to the bottom $p+1$ eigenvalues. The last eigenvalue is always 0 and need to be discarded. This is due to the fact that $I-W$ is a Laplacian of the construted graph. The property of the Laplacian graph suggest that the number of 0 values on its eigendecomposition, corresponds to the number of fully connected subgraphs in the graph.


### Spectral clustering

Spectral clustering is an unsupervised approach used to produce groups of data given a set of $n$ unlabelled observations on which we can define some similairty measure. It creates the maximal connected graph out of them. Since the goal of spectral clustering is to produce subgraphs of the original graph where each point is the most similar with the points in the group it belongs to we need to choose where to cut the original graph.

In graph theory this is know as the $cut(A, A*)=\sum_{i,j}w_{ij}$ problem. Given the graph $G=(V, E)$ where $V=A$U$A*$ and A and A* are disjoint sets of verticies, the goal is to find the optimal set of verticies such that $cut(A, A*)=\sum_{i,j}w_{ij}$ is minimized. The problem with this formulation of the problem is that it can be highly influenced by the distances $w_{ij}$ that are very dissmilar. Those distance will be high and than we will have issues. To that end we opt to minimize the ratiocut. Ratiocut is a similarity measure between two sets of verticies from a graph that accoutns for the large differences in similarities.

It is given as:
\begin{equation}
ratiocut(A, A^{*}) = \frac{cut(A, A^{*})}{|A|} + \frac{cut( A^{*},A)}{|A^{*}|}
\end{equation}

In order to minimize this function we find an equivalence function that we minimize. To this end we first introduce
label for each point given as:
\begin{equation}
f_{i} = \sqrt{\frac{|A^{*}|}{|A|}}, i \in A
\end{equation}
\begin{equation}
f_{i} = -\sqrt{\frac{|A|}{|A^{*}|}}, i \in A^{*}
\end{equation}

The introduction of this label with respect to the being part of the sets allows to write the loss function in temrs of
\begin{equation}
min_w \sum_{ij}w_{ij}(f_i-f_j)^2
\end{equation}

Proof:
\begin{equation}
min_w \sum_{ij}w_{ij}(f_i-f_j)^2 = \sum_{i \in A j \in A^{*}}w_{ij}(\sqrt{\frac{|A^{*}|}{|A|}} + \sqrt{\frac{|A|}{|A^{*}|}})^2 +  \sum_{i \in A^{*} j \in A}w_{ij}(-\sqrt{\frac{|A^{*}|}{|A|}} - \sqrt{\frac{|A|}{|A^{*}|}})^2 =
(\frac{|A^{*}|}{|A|} + \frac{|A|}{|A^{*}|}+2)(\sum_{i \in A j \in A^{*}}w_{ij}+\sum_{ i \in A^{*}, j \in A}w_{ij})=
K(cut(A, A^{*}) + cut(A^{*}, A))=K(\frac{cut(A, A^{*})}{|A|} + \frac{cut(A^{*}, A)}{|A^{*}|}) => ratiocut(A, A^{*})
\end{equation}

Furthermore, with straiightforward representation calculation of $f^TLf = f^T(D-W)f$ one can show that $f^TLf <=> \frac{1}{2}\sum_{ij}w_{ij}(f_i-f_j)^2$, where L is the Laplacian of the connected graph, W is the weight matrix of the graph and D is the diagonal nodes of the degree of each node in the graph. Adding the contraint $f^Tf=I$, one can show that the optimial solution for the f's being the $p+1$ eigenvectors of the matrix L. The last column is discarded since it represents the number of partitions the graph has. D is diagional matrix and the sum of the rows of W equal the corresponding diagonal element in the row. Note that this is minimization problem.

In order to do more clusters, one is prespecifing the number of clusters $p$ it wants and runs corresponding k-means algorithm on the $p+1$ eigenvectors corresponding to the $p+1$ minimal eigenvalues.

### Laplacian Eigenmaps
Laplacian eigenmaps as a dimensionality reduction technique concerned with finding the lowest dimensions. One can proof that they correspond to the optimal values find in the Spectral clustering. The difference between kmeans and spectral clustering is that kmeans is kmeans on original space, while spectral clustering is kmeans on the reconsturcted lower dimensional p-space.



### Maximum Variance Unfloding

All the methods for dimensionality redcution discussed so far are KPCAs with different kernels. Follwing this observation the question that arises is related to: Can we learn kernels from the data?

This can be done via semi-definite programming casting of the problem.
Let's assume $x \in R^{dxn}$ and $y \in R^{pxn}$. The kernels are able to preserve the local properties of the data (e.g nearest neighbour).
\begin{equation}
|x_{i}-x_{j}|^2 =  |\phi(x_{i})-\phi(x_{j})|^2 = K_{ii} + K_{jj} -2K_{ij}
\end{equation}

We are lookging for a kernel such that, if $x_{i}$ and $x_{j}$ are neighbours:$|x_{i}-x_{j}|^2 = K_{ii} + K_{jj} -2K_{ij}$. Since K is kernel in need to be positie-semi definite, K should be also be symmetric and centered. Next, we need to define a cost fucntion to optmize. This is very easy. If we recall that the PCA was performing maximization of the variance, as a natural step is to do maximization of $Tr(K)$ as a measure of variance of the embedding.

Finally, the first part of the problem for Maximum Variance Unfolding can be formulated as:
\begin{equation}
\max_{k}Tr(K)
\end{equation}
subject to the following three constraints:

Centered:
\begin{equation}
\sum_{ij}K_{ij}=0
\end{equation}

Non-negative:
\begin{equation}
K \geq 0
\end{equation}

Preserving the locallity:
\begin{equation}
||x_{i}-x_{j}||^2_2 = K_{ii}^2 + K_{jj}^2 -2K_{ij}^2
\end{equation}


This problem belongs to the cateogry of semi-definite programming and it can be solved using standard approaches from semi-definite progrmming for solution. Once the kernel is found, run kernel PCA on top of it and you will obtain the solution of MVU. One solver is by Helnberg-Kojima-Monterio interior point method.

### Nystorm approximation

One issue with all of the discussed methods is that they do not scale well. Nystorm apprpximation is a technique that makes the methods scalable.

Lets assume we are given a matrix $K=[A | B; B^T | C]$. The claim is that if we know A and B we can reconstruct C.
One can show that $C = B^TA^{-1}B$. If the rank(C)=m, and we choose m rows for the matrix A, we can reconsutct C,  otherwise we cannot. Knowing some partial distances put constraint on where the other points in the space can be located.

To spped up the computations one adhers to calculating of the properties locally and try to appriximate the other points with Nystorm approximation. As long as we choose good values for m (being above the rank(K)) we are good even if we do that at random.

Proof that Nystorm approximation works:
Let K is a kernel function such that $K=X^TX$ and $X \in R^{dxn}$. Let m be an integer representing the number of chosen rows of the matrix K. We will write $X=[R; S]$ where $R \in R^{dxm}$ and $S \in R^{dxn-m}$. Then the matrix
$K=[R^TR | R^TS; S^TR | S^TS] = [A|B; B^T|C]$.

The matrix R can be found as an SVD decomposition of matrix A. $R = \sigma^{0.5} U$, where $\sigma$ and $U$ are the diagonal matrix of eigenvalues and eigenvectors of matrix A. $B=R^TS$, replacing the solution for R we have $B=U\sigma^{0.5}S <=> U^TB=\sigma^{0.5}S <=> \sigma^{-0.5}U^TB=S$. Since $C=S^TS$, replacing for S we have $C=S^TS=B^T\sigma^{-0.5}\sigma^{-0.5}U^TB=B^TU\sigma{-1}U^TB <=>C=B^TA^{-1}B$.
This approximation is exact if rank(K) is at most m.

Under this umbrella we can fit all the fast versions of the methods: FastMDS,  Fast ISOPMAP etc.
Fast MDS: 1) Select $m<<n$ data points; 2) calculate pairwisie distance between the m points; 3) calculate the distance between m and all other n-m points.
There is a paper that shows that all fast (or landmark) works are different reinventions of the Nystorm approximation method.

### Non-negative matrix factorization.

The goal is that given a matrix $A \in R^{mxn}_{+}$, find matricies W and H such that $A=WH$.


There are many algorthims one can use to solve this problem. One for example is using gradient descent:
\begin{equation}
\max_{W, H} ||A-WH||_2^2
\end{equation}

starting from some initial random values (or some careful intialization) one can find both the bases (W) and the factors (H).
Also, one can use the R1D algortihm to find the bases. The intution follows from "Leading singular value of a nonnegative matrix is nonnegative. (Theorem)". Utilizing this notion one can write the problem of fidning W and H as eigenvectors.

### Stochastic neighbour embedding (SNE)

Stochastic neighbour embedding is used for representation of the data in a lower space. The main idea is that it assumes that the local neighbourhood of a datapoint in the original high dimensional $X \in R^{dxn}$ space can be embedd under a Gaussian. This results in conversion of distances to probabilities. The same assumption is made for the lower dimensional representation $Y \in R^{pxn}, p<<d$ . Then it tries to minimize the KL divergence between these two distributions in an iterative way.
The conversion of distances to proabiltities is given with:

\begin{equation}
p_{j|i}=\frac{\frac{e^|x_{i}-x_{j}|^2}{2\sigma_{i}}}{\sum_{k!=i}\frac{e^|x_{i}-x_{j}|^2}{2\sigma_{i}}}
\end{equation}

where each $\sigma_{i}$ is different for each point.

\begin{equation}
q_{j|i}=\frac{\frac{e^|y_{i}-y_{j}|^2}{2\sigma_{i}}}{\sum_{k!=i}\frac{e^|y_{i}-y_{j}|^2}{2\sigma_{i}}}
\end{equation}

the $\sigma_{i}=\frac{1}{\sqrt{2\pi}}$ in Y space is set to constant.

The cost function is than:
\begin{equation}
J=\min_{y_i, y_j}KL(P||Q) = \sum_{i,j}p_{j|i}log\frac{p_{j|i}}{q_{j|i}}
\end{equation}


One of the problem we are faced with such formulation is the "crowding" problem. This problem arises in the case when we want to map data from higher dimension to lower dimension due to the uneven volumes of the both spaces. It usually will result that the points that are on medium and large distance in a higher dimensional space to be mapped very far from one another in the lower dimensional space. To eliminate this problem in SNE, tSNE is introduced.


#### t-SNE
It builds on the top of SNE, but with two differences. Instead of using separate $\gamma_{i}$ for all points separately it uses one $\gamma$ for all points. The second important thing is that the Gaussian distribution in the lower dimensional space is replaced with t-student distribution. The main intuition is that the t-student distribution has thicker tails. This results in preserving the large distances as compared to SNE since it can accommodate greater volume in the lower dimensional space.

\begin{equation}
p_{ij}=\frac{\frac{e^|x_{i}-x_{j}|^2}{2\sigma}}{\sum_{k!=i}\frac{e^|x_{i}-x_{j}|^2}{2\sigma}}
\end{equation}

\begin{equation}
q_{ij}=\frac{\frac{1}{1+|y_{i}-y_{j}|^2}}{\sum_{k!=i}\frac{1}{1+|y_{i}-y_{k}|^2}}
\end{equation}

where each $\sigma$ is the same. We again optimize $KL(P||Q)$, for y.


### Component analysis

Canonical Component Analysis (CCA)
The main idea behind this method is that given $X \in R{n}$ and $Y \in R{m}$ find a representation Z such that it is maximally related to both of them. One can look as X and Y being conditionally independent given Z.

Find the projections $w_x \in R^{m}$ and  $w_y\in R^{n}$. The objective function can be formulates as:
\begin{equation}
argmax_{w_x, w_y}w_xXY^Tw_y
\end{equation}
s.t

\begin{equation}
w_xXX^Tw_x = 1
\end{equation}

\begin{equation}
w_yYY^Tw_y = 1
\end{equation}

We assume that the data is centered.
Compute the cross-covariance matricies:
\begin{equation}
C_{xy} = \frac{1}{N}XY^T
\end{equation}

\begin{equation}
C_{xx} = \frac{1}{N}XX^T
\end{equation}

Now we can write the Lagrangian:
\begin{equation}
L = w_x^TC_{xy}w_y - \frac{1}{2}\alpha (w_x^TC_{xx}w_x - 1)- \frac{1}{2 \beta (w_y^TC_{yy}w_y - 1)
\end{equation}

We calculate the partial derivatives:
\begin{equation}
\frac{\delta L}{\delta w_x^T} = C_{xy}w_y - \alpha C_{xx}w_{x}
\end{equation}

\begin{equation}
\frac{\delta L}{\delta w_y^T} = C_{yx}w_x - \beta C_{yy}w_{y}
\end{equation}


From here follows:
\begin{equation}
w_x^TC_{xy}w_x = \alpha w_x^TC_{xx}w_x
\end{equation}

\begin{equation}
w_y^TC_{yx}w_y = \beta w_y^TC_{yy}w_y
\end{equation}

Combining all of these results in a block matrix form we get:
\begin{equation}
[0| C_{xy}; C_{yx} | 0][w_x; w_y] = \alpha [C_{xx}| 0; 0 | C_{yy}][w_x; w_y]  
\end{equation}

This is a generalized eigenvalue equation solvable with standard eingenvalue solver.
It can be extended on more then 2 variables and we can utlized also Kernel versions of it.
To capture nonlinear dependecies use kernel. Hence we have **kCCA** (kernel Canoical Component Analysis).

\begin{equation}
[0| K_{xy}; K_{yx} | 0][\alpha_x; \alpha_y] = \alpha [K_{x}^2| 0; 0 | K_{y}^2][\alpha_x; \alpha_y]  
\end{equation}
To recover $w_x$ use $w_x=X\alpha_x$ same for $w_y$.
There is also a temporal kernel CCA. When variables are copuled with delay.  

\begin{equation}
argmax_{w_x(\tau), w_y} Corr(\sum{w_x(\tau)^Tx(t-\tau), w_y^Ty(t)})
\end{equation}

tkCCA finds canonical convolution and correlogram.



### Independent Component Analysis (ICA) Bell 1995

ICA solves the "cocktail party problem". This problem refers that there exist multipe sources of signals that are linarly mixed between one another. This assumption of linar mixing is valid since it relfects the baisc principle of superposition. Moreover, it imples an important assumption of independance between the different sources. Formally, the problem is defined as follows:
\begin{equation}
X = AS
\end{equation}
, where $X \in R^{pxn}$  represents the p time points of n-dimensional observartions. A is the mixing matrix and S is the matrix representing the sources.
The goal of ICA is to find an unmixing matrix $W$, such that:
\begin{equation}
S = WX
\end{equation}

We can also refer to the sources as independent components.
There are various view points of the ICA method and various approaches to it. Here a description of the "InfoMax" method for ICA will be presented.

#### InfoMax ICA
Let's assume that there $s$  comes from uknown densitiy $P_s{(s)}$, parametarized with $W$ or it can be written as $P_s{(Wx)}$. Our goal is to estimate this probability $P^*_{s}{(s)}=\prod_{i=1}^N P_{s_i}^{*}(s_i)$ given the obesrvations, under the independence assumption.
Thus we can write our cost functions as minimization of the KL divergence of these two probabilities with respect to the unmixing matrix W.
\begin{equation}
D_{KL} = D_{KL}[P_s(s), P^*_{s}(s)] = \int dsP_s(s)ln\frac{P_s(s)}{\prod_{i=1}^N P_{s_i}^{*}(s_i)} (1)
\end{equation}

To solve this one may think of the infomax principle. From Wikipedia **"Infomax is an optimization principle for artificial neural networks and other information processing systems. It prescribes that a function that maps a set of input values I to a set of output values O should be chosen or learned so as to maximize the average Shannon mutual information between I and O, subject to a set of specified constraints and/or noise processes"** .More specifically, we would like to apply a specific non-linear transformation of our reconsturcted sources (to each of them individually) $u(s^*)$ such that the $P(u)=const.$. This reflects an intuition that the reconstructed sources will be as much unrelated as possible.

Using the nonlinear transformation $u^{*}_i = f^{*}_{i}(e_i^TWx)$ and following the law of conservation of probability (or the rule of change of variables) we can derive the infomax princile, starting from equation (1).
This is a long process and we are going to state just the result:
\begin{equation}
H = -\int du^*P_u(u^*)lnP_u(u^*)
\end{equation}
this can be obesrved as entropy of the flattent reconstructed sources. The intuition is that "If you like a matrix W that transforms something to statistically independent thing you should maximize the entropy. For multivariate variabels with some pdf and if the pdfs are flat, then the entorpy is maximal if and only if the sources are independet".

Following this formulation one can derive the following cost function:
\begin{equation}
E^G=ln(|det(W)|) + \int dx*P_x(x)\sum^N_{l=1} ln f^{*'}_l(\sum_{k=1}^Nw_{lk}x_k))
\end{equation}
The problem with this equaton is that we cannot calculate the integral. One work around is to utilize the ERM principle and replace it with the sum. ERM principle allows to replace the mathematical expectation by an empirical average. This allows us to rewrite the equation in the following form:

\begin{equation}
E^T=ln(|det(W)|) + \frac{1}{p}\sum_{\alpha=1}^p \sum_{l=1}^N f^{*'}_l(\sum_{k=1}^Nw_{lk}x_k^{\alpha}))
\end{equation}

This formulation of the problem can be solved using the gradient ascent algorithm. However, there is a problem with this formulation. It requires calculation of a determinant of the matrix. To go around that problem one may use the Natural gradient instead of the gradient. The natural gradient acts is similar to normal gradient descient with a difference that instead of finding the optimal step for the current update of the parameter in the Euclidiean space, it tries to find optmial updated in a space of distributions. This allows to eliminate the calculation of the gradient of the determinant of the unmixing matrix and provide a solution for the problem.

##### A few words on Natural gradient descent:

**Definition:** Natural gradient is defined as:
\begin{equation}
\nabla_{\theta}L(\theta) = F^{-1}\nabla_{\theta}
\end{equation}

Repeat until convergence:
1. Do forward pass on our model and compute loss $L(\theta)$
2. Compute the gradient $\nabla_{\theta}L(\theta)$
3. Compute the Fisher Information Matrix F, or its empirical version (wrt. our training data).
4. Compute the natural gradient $\nabla_{\theta}L(\theta) = F^{-1}\nabla_{\theta}$
5. Update the parameter: $\theta = \theta - \alpha \nabla_{\theta}L(\theta)$  , where α  is the learning rate.

In practice it is difficult to compute F matrix (it is the negative Hessian). Fisher Information Matrix can be seen as a curviture of the negative expected logliklihood of the loss function. [cite the blog from Augustinus Kristiadi]. Also it can be defined as the variance of the log of maximal liklihood estimate as a score for the goodness of the estimate $\nabla_{\theta}log(p(x|\theta))$.

### Variational Autoencoder

VAE is using variational inference to provide esimates of untractable probabiltitites.

To define all the things we need we start with first defining what is information
**Information** is quantified with the $I=-log(p(x))$, where x is some event. For example, the probability of raining snow in July is low, since it is very unlikly. However, if someone tells that the snow will rain that means that that information has very great information within.


The average of information is **entropy**. The mathematical expectation of the information is entorpy $E(x)= \int -p(x)log(p(x)dx$.


Similarly as MMD (maximal mean discrepancy) **KL divergence** measures the dissimilarity between two different distributions. As opposed to MMD the KL divergence is not symmetric. It is always positive.

For easier remambering we can observe the **KL divergence** as difference between entropie of two distributions. This is not true but helps for better remamberinng:
\begin{equation}
KL(p||q) ~= \sum -q(x)log q(x) + \sum p(x)log(p(x))
\end{equation}
The truth is
\begin{equation}
KL(p||q) ~= \sum -p(x)log q(x) + \sum p(x)log(p(x)) = \sum p(x)log p(x)/q(x)
\end{equation}

The properties of the KL divergence are: it is positive and it is not symmetric.

$p(z|x) = p(x,z)/p(x)$

It is difficult to compute p(x)
To that end we use:
1) Monte Carlo methods or
2) Variational inference

With variational inference, we try to approximate p(z|x) with a known q(z).
We want to make q close to p. That means we want to
\begin{equation}
min KL(q(z)||p(z|x)) = -\sum q(z) log \frac{p(z|x)}{q(z)}.
\end{equation}tli
This can further be decomposed and equivalent
\begin{equation}
p(z|x) = \frac{p(x|z)p(z)}{p(x)}= \frac{p(x,z)}{p(x)} = -\sum q(z)log \frac{\frac{p(x,z)}{p(x)}}{q(z)} = -\sum q(z) log \frac{p(x,z)}{q(z)} * \frac{1}{p(x)} = - \sum q(z)[log \frac{p(x, z)}{q(z)} -log p(x)] = -\sum_z q(z)log \frac{p(x,z)}{q(z)} + \sum_z q(z)log p(x) = -\sum_z q(z) log \frac{p(x,z)}{q(z)} + log p(x) * \sum_z p(z) -> 1 <=> log p(x) = KL (q(z) || p(z|x)) +  \sum q(z) log \frac{p(x,z)}{q(z)}
\end{equation}


$log p(x) = const.$ it is fixed number if x is given. That means it does not depends on q(z).
Since the objective is to minizie the  KL divergnece. that means that we want to maximize the lower bound $L=\sum q(z) log \frac{p(x,z)}{q(z)}$. This is called a variational lower bound. We can maximize this variational bound instead of minimizing the KL divergence. However, this bound is not tight and that will have additional reflections in the result since the KL divergence will not exactly be minimized.


\begin{equation}
max L = \sum q(z) \frac{p(x,z)}{q(z)}  = \sum q(z)[log(p(x|z) + log\frac{p(z)}{q(z)}] = \sum q(z)log(p(x|z)) + \sum q(z)\frac{p(z)}{q(z)} = \sum q(z)log(p(x|z)) - KL(q(z)||p(z)) = E_{q(z)}logp(x|z) - KL(q(z)||p(z))
\end{equation}

In fact if we interpret VAE as a graphical model: one can use $z->x$ and map it to probability p(z|x), while at the same time it can make assumpuion of exsiting recursive relationship $x->z$ to represent the mapping $q(x|z)$. Then the later can be seen as encoder, and the former as decoder. This practically means that we choose some distribution for $z$ that we want to reconstruct. With minizmiation of the above quantity one can actually try to learn how to map the corresponding chosen distribution.

In case of Gaussian E_{q(z)}logp(x|z) results in $E_{q(z)}log(exp(-|x-x^*|^2)$ which is equivalent to minimization of the reconstruction error thus:
\begin{equation}
\max E_{q(z)}logp(x|z) - KL(q(z)||p(z))  <=> \max -E_{q(z)}logp(x|z) - KL(q(z)||p(z)) <=> \min recon\_error +  KL(q(z)||p(z))
\end{equation}

It is important to note that in the bottleneck we are aiming to reconstruct not the code of the distribution but the code for its parameters $\mu$ and $\sigma$. Then we sample from this distribution and pass the sample through the decode. This is reered to as reparametarization trick. And we can use this model as generative model in that sence. We are using the reparametarization trick to do this. Read the original paper "VAE 2013" for more details.


-------
Algorithm

1) initilize paramteres $\theta$ and $\phi$

2) while convergence do:

2.1) X^M sample a minibatch from X

2.2) sample $\epsilon$ from a distribution (e.g. Gaussian)

2.3) g <- $\nabla_{\theta, \phi} L(\theta, \phi; X^M, \epsilon)$

2.4) $\theta, \phi$ <- update parameters using g

3) return $\theta$ and $\phi$


There are various difference ascepts of the VAE framework. Adding a regularizing parameter infront of the KL term, allows to implicitly contorl for the amount of independece enforced on the output. Thus one can either use simmulated annealing, or add it as large constant (e.g beta VAE). Thus one can end up in representations with different richness. For example, with large beta's one allows for emphasised independance between the latent factors enfored with the isotropic covarince of the latent factor embedding that encodes it. This is has interesting observation in causality learning. Another interesting extension is the recurrent variational autoencoder. It is extension of the VAE framework but at the same time it takes into account the previos state encoded by a hidden layer in an LSTM for example. Another interesting extension is regularizing the value for beta using a closed-loop system when the value for the regularization parameter using the KL divergence as input.
