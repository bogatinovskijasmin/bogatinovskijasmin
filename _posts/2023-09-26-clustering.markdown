---
title: "Markdown Common Elements"
layout: post
date: 2016-02-24 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- markdown
- elements
star: true
category: blog
author: johndoe
description: Markdown summary with different options
---


## Clustering

Clustering is a task from the unsupervised learning paradigm in machine learning. The goal of clustering is finding groups of samples sharing similar characteristics. The properties of the groups are such that the samples within the groups are similar between each other, while the samples between the groups are different. One main challenge in clustering is defining "what does it mean for samples to be similar?". As a topic arises from the need to categorize the similarity between a set of heterogeneous objects (based on some notion of similarity). 

In essence, clustering is an ill-defined problem. When given data without labels the notion of similarity is vague. Let's think of the following example: let's assume we have a dataset with 4 examples. The examples have two features: shape and colour; 1) red square 2) red circle 3) blue square 4) blue circle. If we want to find two natural groups in this data it is not clear what they are (if we do not have a clear goal that we want to separate the red from the blue or squares from the circles). The two groups can be either group 1 with samples 1 and 4; group 2 with samples 2 and 3 or the two groups can be composed of group 1 with samples 1 and 2, and group 2 with samples 3 and 4. Thus clustering is an ill-defined problem. 

![image info](../images/clustering/red_truck.png)



More formally, we can define clustering as follows. A clustering function is any function $f$ that takes a set $S$ of $n$ points with pairwise distances between them and returns a partition of $S$. There is no constraint on the points constructing the set $S$. The only information we have available for the points are the pairwise distances between them.

Clustering is one very interesting topic of research as well as application. One very interesting axiomatic theoretical work on clustering is the work of [Kleinberg 2002](http://alexhwilliams.info/itsneuronalblog/papers/clustering/Kleinberg_2002.pdf). In this work, Kleinberg makes a set of intuitive assumptions about the properties a cluster group should satisfy in respect to the other groups and as well the group concerning itself. On top of them, a theorem and proof are introduced that all the properties cannot be satisfied at the same time. This further justifies the difficulty of the problem.

___
Basic axioms about the groups (Kleinberg 2002) should satisfy are:

1) **scale-invariant;** The results from the clustering should not change when the data change their scale.

2) **consistency:** If we change the distances (shrinking and stretching) between the given points, such that we increase the between cluster distance or decrease the within-cluster distance, the results of the clustering should not change;

3) **rich:** Given is a set of data points for which we do not know anything about their distance. An ideal clustering function would be flexible enough to produce all possible partition/clusterings of this set. This means that all partitions of the set $S$ are achievable. 

However, one can prove that none of the three properties can hold at the same time (but there are many cases where 2 out of 3 can hold). For more details, refer to the paper. 

A simple visual illustration for the contradictory obtained is given on the following image. Let's assume that all the three axioms hold. Then due to the richness axiom there exist two distance measure $d_1$ and $d_2$ that can realize any possible partition on $S$ (top and bottom left). We can define a third distance measure $d_3$ that scales $d_2$ so that the minimum distance between points in $d_3$ space is larger than the maximum distance in $d_1$ space. This leads to a contradiction.  The clustering should remain unchanged after the T1 and T2 transformation.

![image_proof](../images/clustering/impossability_proof.png)

___

When applying some of the common approaches for clustering, several usual questions need to be considered:

1) The number of clusters; It is not clear what is the exact number of clusters existing in a dataset. There are heuristics like the **"elbow-method"** that can be used to find this number. However, these methods should be taken with caution (due to the ill-definition of the clustering problem). Some methods do not require the number of clusters predefined. 

2) Having a good **similarity measure**; It is not clear what does a similar and different object look alike. Before applying clustering to a given set of points, one should carefully examine the data and define the similarity appropriately. 

3) It is difficult to determine which points are **outliers** and clusters for themselves, especially in high dimensional space. In high dimensional space, the data points are usually far apart between one another and it can be hard to distunugish between group of points. 

4) It is difficult to differentiate among overlapping clusters;

A common strategy when solving clustering problem is to apply several clustering methods in an ensemble like clustering or sometimes referred to as **collaborative clustering**. In such a way one may end up in a set of clusters with higher confidence that indeed represent some phenomena in the data. 
___
The number of clustering methods is quite large. Generally, the approaches are grouped according to the underlying paradigm they adopt. Most frequently they are separated into: agglomerative, spectral,
information-theoretic, centroid-based, methods from combinatorial optimization and probabilistic generative models. In this post we will consider:

1) **K means/k medians**; is a goto method for clustering;

2) **Hierarchical clustering** (**Agglomerative clustering** with different linkages (distances between set of points) ward, single, complete etc. or **Divisive clustering** e.g tree-like methods such as Predictive Clustering Trees); 

3) **Spectral clustering**; Based on graph theory.

4) **Gaussian Mixture models** (and other Bayesian techniques); This method allows for the term **soft clustering** assigning a point to multiple clusters with a specific confidence. Having this uncertainty can be very neat in some cases.

Additionally in this post, we are going to introduce:

1) **Jensen inequality**

2) **Expectation maximization principle**

3) **Coordinate descent**

4) **Soft clustering**

5) **Consensus clustering**

6) **Ensemble of clusters**

7) **Linkage**

8) **Biclustering**

9) **Density estimation**

10) **Graph, (edges, nodes, Laplacian, adjecency, spectral gap, Fiedler value, graph cut, ratio cut)**

