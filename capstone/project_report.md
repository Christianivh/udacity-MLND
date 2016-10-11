# CIFAR 10 Image Recognition using CNN Machine Learning and Caffe
- Machine Learning Engineer Nanodegree Capstone Project
- Neo Xing, 2016/10

## I. Definition

### Project Overview
- overview/ motivation/ applications
- Image classification, [figure to show general applications]
- Deep learning, NN, [general application]
- CNN
  - [figure to show structure]
  - [table to show advantages and weakness of CNN]

### Problem Statement
- definition
  - CIFAR 10 image classification challenge using CNN and Caffe, seeking best accuracy and performance
  - [figure to show CIFAR image sets]
- strategy
  - convolutional neural network
  - Caffe features
- solution fine tuning CNN with caffe framework
  - [default structures]
  - transfer learning
  - other optimization, data augmentation, pre trained data

### Metrics
- metrics
  - accuracy/ error rate
  - overfitting loss function, learning curve
  - performance

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
- image sets
  - [table of CIFAR10 data/metadata overview]
  - [table of sample images]
- challenges, [figures to show typical challenges in image classification]
  - smaller size compared to CIFAR 100, easy to overfit, [some are hard for human recognition]
  - illumination, deformation, background

### Exploratory Visualization
- statistical plot of features, [figure PCA feature exploration]
- t-SNE patterns, [figure]

### Algorithms and Techniques
- vanilla NN using sklearn
- caffe
- optimization

### Benchmark
- [table of best performance]


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
- caffe preprocess, read into data base
- central, normalize
- split and initialize

### Implementation
- NN with sklearn is straight forward
- structure of CNN, layers, filters

### Refinement
- data augmentation
- regularization
- pre trained data

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
- chosen of hyper parameters
  - [model analysis]
  - [learning curve]
  - [loss function]
  - visualization of [activation] [weights]
- validate the robustness of this model and its solution, [testing curves or eval sets]


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
- [tSNE]
- [sample prediction]

### Discussion
- layer design, parameters control, overfitting
- long training circle
- future work, [evaluation on general datasets]
