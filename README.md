# Spam Filter
Spam FIlter using Support Vector Machine, K-NN and Naive Bayes

## Assignment
Write a spam filter using discrimitative and generative classifiers. Use the [Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase) which already represents spam/ham messages through a bag-of-words representations through a dictionary of 48 highly discriminative words and 6 characters. The first 54 features correspond to word/symbols frequencies; ignore features 55-57; feature 58 is the class label (1 spam/0 ham).

1. Perform **SVM** classification using **linear**, **polynomial of degree 2**, and **RBF** kernels over the TF/IDF representation. Can you transform the kernels to make use of angular information only (i.e., no length)? Are they still positive definite kernels?
2. Classify the same data also through a **Naive Bayes** classifier for continuous inputs, modelling each feature with a Gaussian distribution, resulting in the following model:
![equation](https://latex.codecogs.com/svg.image?p(y%20=%20k)%20=%20%5Calpha_%7Bk%7D)
![equation](https://latex.codecogs.com/svg.image?p(y%20=%20k)%20=%20%5Cprod_%7Bi=1%7D%5E%7BD%7D%5Cbigg%5B(2%5Cpi%5Csigma%5E%7B2%7D_%7Bki%7D)%5E%7B-1/2%7Dexp%5Cbigg%5C%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E%7B2%7D_%7Bki%7D%7D(x_i%20-%20%5Cmu_%7Bki%7D)%5E%7B2%7D%5Cbigg%5C%7D%5Cbigg%5D)
where α_k is the frequency of class k, and μ_ki, σ^2_ki are the means and variances of feature i given that the data is in class k.

3. Perform k-NN clasification with k=5

Provide the code, the models on the training set, and the respective performances in 10 way cross validation.
Explain the differences between the three models.

## Provided Solution
