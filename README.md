1 Simple Linear Regression

Regression models are used to predict a dependent variable using other known (denoted by
Y) or independent variables (denoted by X). Linear regression is one such regression model
in which prediction is made using one independent variable(X) to predict a dependent
variable(Y). The relationship between X & Y is assumed to be linear.

In Linear regression, we try to find the best fitting line to minimize the errors in prediction.
This line is called the “line of best fit” i.e. the line of regression on which the errors will be
minimal. We try to minimize the distance between the observed value and the predicted
value. So, our general equation for Linear regression comes out to be:

Y = b 0 + b 1 X 1

Where,

Y = Dependent variable

b 0 = Y-intercept

b 1 = slope of the line

x 1 = independent variable

1.1 Experiment

We are going to use a housing dataset [1] to illustrate the application and workings of Simple
Linear Regression. We will use this dataset and train a model to predict housing prices using
linear regression. The data has features as below:

1.2 Feature Selection

Above we built an sns plot in which the diagonal plots show the distribution of a single
variable and the other plots in the upper and lower triangle show the relationship between
two variables. From the above we can tell that for our target variable (Y) the distributions
that are most linear are:

sqft_living
sqft_above
Next, we will create a correlation plot. Correlation helps us understand the relation between
two variables. It ranges from -1 to +1. If the value of the relationship is zero, there is no
correlation. As the value of correlation changes from zero to a positive or negative one, the
linear relationship grows stronger. We can observe from the above plot that the best
correlation for the price variable is from:

sqft_living
grade
sqft_above
As sqft_living is having the best correlation with price we choose this variable for our
analysis.

1. 3 Training the model

We will then store the price variables(Y) and sqft_living(X) and split this data into training
and test in the ratio of 0.75 and .25 respectively. The train dataset is used for training our
simple linear regression model.

Now we will train to fit our model on the train dataset such and test the model to predict
using the Test Dataset. We will then compare the Actual values to the predicted values:

We can also find the intercept and coefficient values from the trained model:

From this we can build our equation with intercept value = -41565.741905973875 and
coefficient value = 280.28160476. So, our linear regression equation looks like:

Y = 280.28X - 41565.

We will now perform visualization our results:

train dataset using actual data and predicted data:
test dataset using actual data and predicted data:
1. 4 Validation

We will use mean absolute error for validation. MAE calculates the average error between
the actual and the predicted values. MAE is:

This means that on average our predictions we off by 173359.02 which means that our model
performs poorly.

References

[1] Housing Price dataset from https://www.kaggle.com/datasets/shivachandel/kc-house-data

[2] https://medium.com/analytics-vidhya/simple-linear-regression-using-python-98ddd7e6b

2 Multiple Linear Regression
Multiple linear Regression is based upon linear regression but it takes more than one
independent variable to predict a dependent variable. It fits a linear equation based upon two
or more dependent variables[X] to predict the independent variable [Y] and forms a line of
best regression.

Thus the equation is:

Y = b 0 + b 1 X 1 + b 2 X2 + b 3 X 3 + ... + bnXn

Where,

Y = Dependent variable

b 0 = Y-intercept

b 1 , b 2 , b3,..., bn = slope of the line

x 1 , x 2 , x 3 ,..., xn = independent variable

Example: Predicting if a customer will buy a car based on his salary, city, gender, etc

2 .1 Experiment

We are going to use a housing dataset [1] to illustrate the application and workings of
Multiple Linear Regression. We will use this dataset and train a model to predict housing
prices using linear regression. The data has features as below:

2 .2 Feature Selection

Above we built an sns plot in which the diagonal plots show the distribution of a single
variable and the other plots in the upper and lower triangle show the relationship between
two variables. From the above we can tell that for our target variable (Y) the distributions
that are most linear are:

sqft_living
sqft_above
Next, we will create a correlation plot. Correlation helps us understand the relation between
two variables. It ranges from -1 to +1. If the value of the relationship is zero, there is no
correlation. As the value of correlation changes from zero to a positive or negative one, the
linear relationship grows stronger. We can observe from the above plot that the best
correlation for the price variable is from:

sqft_living
grade
sqft_above
2. 3 Training the model

We will perform label encoding for all the features of our dataset. We will then store the
price variables(Y) and all the other independent features(X) and split this data into training
and test in the ratio of 0.75 and .25 respectively. The train dataset is used for training our
simple linear regression model.

Now we will train to fit our model on the train dataset and test the model to predict using the
Test Dataset. Predicted values are :

We can also find the intercept and coefficient values from the trained model:

From this we can build our equation with intercept value = 11836332.32264203 and
coefficient value = -3.39679112e+04, 4.09513566e+04, 1.08176103e+02....

2. 4 Validation

We will use mean absolute error for validation. MAE calculates the average error between
the actual and the predicted values. MAE is:

This means that on average our predictions we off by 131275.09 which means that our model
performs poorly but better that simple linear regression

References

[1] Housing Price dataset from https://www.kaggle.com/datasets/shivachandel/kc-house-data

[ 2 ]
https://medium.com/machine-learning-with-python/multiple-linear-regression-implementation-in-pyth
on-2de9b303fc0c

3 Logistic Regression
Logistic regression is a supervised machine learning algorithm classification that is used to
predict categorical data. In logistic regression, the target variable or dependent variable(Y) is
a binary variable and has either 1(true) or 0(false) as values. So essentially the model
predicts P(Y=1) as a function of X. For example, will a certain customer buy a product or
not?

Logistic regression uses a sigmoid function which is a mathematical function that takes any
real number and returns the value as either 0 or 1.
Sigmoid function [ 2 ]:

Sigmoid graph

The sigmoid function forms an S shaped graph, so when x becomes infinity the value of P(Y)
is 1, and when X reaches negative infinity the value of P(Y) is 0. The model sets a threshold
as to above which value the model predicts if the event Y will happen or not. Example the
thresh hold might be p(Y) = 0.5, any value above it will return true for event P happening and
false for it not happening.

3 .1 Experiment

We are going to use the dataset from the National Institute of Diabetes and Digestive and
Kidney Diseases[1] to train a model to predict whether a patient has diabetes or not. The data
has features as below:

3 .2 Feature Selection

We will create a correlation plot. Correlation helps us understand the relation between two
variables. It ranges from -1 to +1. If the value of the relationship is zero, there is no
correlation. As the value of correlation changes from zero to a positive or negative one, the
linear relationship grows stronger. We can observe from the above plot that the best
correlation for the outcome variable is from:

glucose
Insulin
As glucose is having the best correlation with outcome we choose this variable for our
analysis.

3. 3 Training the model

We will then store the outcome variable(Y) and Insulin(X), and split this data into training
and test in the ratio of 0.75 and .25 respectively. The train dataset is used for training our
logistic regression model.

We will use the standard scaler in scikit-learn to normalize the independent variable as
logistic regression uses gradient descent and if some features have higher magnitude and
some have lower magnitude the convergence becomes difficult. Scaling helps make
convergence happen faster.

Now we will train to fit our model on the train dataset and test the model to predict using the
Test Dataset.

3. 4 Validation

The classification report gives us a brief summary of our model.

Here,

Precision: It is the score of accuracy that a label has been predicted correctly.

Precision = TP/(TP + FP)
Recall: It is the true positive rate.

Recall = TP/(TP + FN)

F1 score: It is the harmonic mean between precision and recall. It has the best value at 1 and
the worst at 0.

Weighted Average: It is the average accuracy of this model and is calculated as the average
of the f1 score for both labels. It is 0.77 in our logistic regression model for the given data.

Using a confusion matrix to compare the results is a form of visualized comparison:

From the above figure, we can tell that there are 36 false negative and 16 false positive labels
that have been marked incorrect.

We will create a scatter plot that will show us how the model performs:

ROC Curve plots specificicity to sensitivity of the model. More the area under the curve,
better is our model. For our model the AUC is 0.82. In best case scenario it should be equal
to 1.

References

[1] https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

[ 2 ] https://www.educative.io/answers/what-is-sigmoid-and-its-role-in-logistic-regression

[3] https://medium.com/codex/machine-learning-logistic-regression-with-python-5ed4ded9d

4 Decision Tree
A decision tree classifier is a supervised machine learning algorithm that is used for
classification. It can with continuous as well as categorical data as well as a mix of both.

A decision tree splits the data on certain conditions into a binary tree with a root node at the
top and branches in between and leaf nodes at the bottom[2].

A decision tree classifier splits the data into two on the concept of maximum information
gain achieved by a feature. This is done iteratively from the root and at each level until pure
children nodes are achieved for each branch.

Information gain is calculated as the measure of how much entropy is reduced by splitting
the data on a given feature.

Entropy is the measure of randomness or impurity in the
dataset.

Thus the goal of information gain is to minimize entropy.

Gini index is also used to measure the impurity of node I, with n different classes of
probability P.

Among Gini index and Entropy, Gini Index is more efficient to be calculated.

4 .1 Experiment

We are going to use a car sale dataset to build a decision tree classifier model which will
predict if a customer will buy a car or not. The data has features as below:

4 .2 Feature Selection

We will create a correlation plot. Correlation helps us understand the relation between two
variables. It ranges from -1 to +1. If the value of the relationship is zero, there is no
correlation. As the value of correlation changes from zero to a positive or negative one, the
linear relationship grows stronger. We can observe from the above plot that the best
correlation for the outcome variable is from:

Age
AnnualSalary
As Age and AnnualSalary have a greater correlation with Purchased we choose these
variables for our analysis. Additionally, we will also use the feature gender.

We will perform label encoding for the feature gender and set male as 1 and female as 0.

4. 3 Training the model

We will then store the Purchased variable(Y) and Age,AnnualSalary & Gender(X), and split
this data into training and test in the ratio of 0.75 and .25 respectively. The train dataset is
used for training our Decision tree classifier model.

Now we will train to fit our model on the train dataset and test the model to predict using the
Test Dataset.

4. 4 Validation

The classification report gives us a brief summary of our model.

Here,

Precision: It is the score of accuracy that a label has been predicted correctly.

Precision = TP/(TP + FP)
Recall: It is the true positive rate.

Recall = TP/(TP + FN)

F1 score: It is the harmonic mean between precision and recall. It has the best value at 1 and
the worst at 0.

Weighted Average: It is the average accuracy of this model and is calculated as the average
of the f1 score for both labels. It is 0. 90 in our logistic regression model for the given data.

A classification report is not the best measure to find if the model is good or not. A better

way is to use a confusion matrix:

From the above figure, we can tell that there are 16 false negative and 4 false positive labels
that have been marked incorrect.

References

[1] https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset

[2] https://medium.com/codex/decision-trees-in-python-98ca587f


1. Introduction
Support Vector Machine (SVM) algorithm is a supervised machine learning algorithm used for
regression classification analysis. In SVM, if we have a labelled dataset, data is divided into classes
by drawing a hyper plane between them. SVM finds the maximum distance hyperplane from the
datapoints, each side of the hyperplane forms 2 separate classes. In two-dimensional data, this
hyperplane forms a line dividing the data into 2 different classes.

When we have more complicated data in which we cannot draw a separating line as in fig 1.2, we transform the data
and add one more dimension say z-axis (Fig 1.3). We can observe that the points around the origin in xy plane will be
clearly separated by z-axis.

For the datapoints in z plane we can consider the points in a circle with the equation w = x^2 + y^2 and
manipulate these points as a function of distance from the z axis. If we transform them back to XY
plane, we will get a circular boundary as in figure 1.4.
These transforms are known as kernels.

2. Parameters
a. C Parameters
It controls the distance between the data points and the hyperplane.

large C: distance is small, it can overfit
small C: distance is large, it can underfit
b. Kernel
as stated before, SVM uses kernel trick to transform nonlinear data. There are 4 types of kernel which we can use:
Radial Basis Function Kernel K(x,xi) = exp(-gamma * sum((x – xi^2))
Good local performance
Polynomial Kernel K(x,xi) = 1 + sum(x * xi)^d
Global kernels
Sigmoid Kernel K(x,xi) = tanh(αxi.xj +β)
Neural Net
Linear Kernel K(x, xi) = sum(x * xi)
When sample is separable in low dimension space
c. Gamma Parameter
It defines how distant values from hyperplane must be considered in rbf, polynomial or sigmoid kernel.

Low Gamma: far away points are considered
High Gamma: only close points are considered
3. Dataset
We are using asteroseismology data which has the distinct features of oscillations of 6008 Kepler stars and using it to
predict whether it is a Red Giant or a Helium burning star.
Feature:
POP: Target Variable, if 0: Red giant star & 1: Helium burning star
Dnu: Mean large frequency separation of modes with the same
degree and consecutive order
Numax: Frequency of maximum oscillation power
Epsilon: Location of the l=0 mode

4. Experiment
From the asteroseismology dataset we will use the features to predict whether the star is a Red Burning Giant or Helium
Burning Star.
We will read the dataset from csv and store it into a data frame. Then we will perform our initial
analysis of the data. We can see that all the features are integer or float so will not require any labeling
or one hot encoding them.
Then we will look for null values. We can tell that the data set does not have any
null values and will not require imputation.

We will perform bivariate analysis using pair plot. Pairplot helps us plot
pairwise bivariate distributions in a dataset which helps us summaries the data
visually. The x & y axis have all the features plotting against the target variable
POP and the diagonal has the distribution of each element against the target
variable POP.

Then we will check for the count of each
classification in the target variable POP to check if
the data is not highly imbalanced. Our data is not
highly imbalanced.
We will then plot the correlation for each variable in the dataset.
This helps us find highly correlated features which we can
remove from our model training as they will not affect our
training by much.

We will then split our dataset into test(30%) and training(70%)

We will then fit our Training data into an SVM classifier and predict the result.
We will compare this result with the actual values from the dataset by using a
confusion matrix.

We are getting an accuracy of 92.03%. This accuracy is without any
normalization and parameter tuning.
We can get better result

We will then normalize our data and check if our
accuracy increases.
Normalizing the data increases our accuracy to 94.02%.
We can increase this further by parameter optimization.
We will now perform parameter optimization by using grid search. For this will run SVM model prediction on different
values of C, Gamma and Kernels.
C: 0.1, 1, 10, 100
Gamma: 1, 0.1, 0.01, 0.
Kernel: rbf, polynomial, sigmoid, linear.
Using GridSearchCV we will run SVM for all the above parameters and select the best ones that fit.

We find out that the best parameters are C = 10, gamma = 1 and kernel being rbf.

Now, finally we will train our model using the best parameter that we found out and build
our confusion matrix and test for the accuracy score.

We are now getting the accuracy of 96.01% which is greater than the previous conditions.
This tells us how using normalization and parameters optimization we can increase our accuracy.

4. References
1 .https://medium.com/machine-learning-101/chapter- 2 - svm-support-vector-machine-theory-f0812effc
2 .https://medium.com/@cdabakoglu/what-is-support-vector-machine-svm-fd0e9e39514f

https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/469/4578#/article
Naïve Bayes Classifier Experiment Report
1. Introduction
Naïve Bayes Classifier is based on Bayes theorem, it is a series of simple probabilistic classifiers which use Bayes
theorem. It is a classification algorithm. Classification algorithm divides the data into separate categories or classes.
The algorithm is called “Naïve” because it assumes that the features in the data have strong independence i.e. they are
unrelated to other features in other classes. For e.g. In case of attrition in a cellphone operator, we assume that the
customer not getting good service and the cost of the cellular plan affecting him to decide to switch are both
independently contribute to the customer leaving.

Conditional probability: Chances of an event occurring given that another event has already
occurred.
Bayes Theorem: Conditional probability of an event, given that another event has already
occurred is equal to the probability of second event multiplied by probability of the first
event.
Bayes rule can further be extended for more than 2 events as:

Conditional Distribution: As events are assumed conditionally independent the distribution can be expressed as:

Since the characteristic variables are constant, z is dependent only on the features. It increases the stability of the
model.

Classifier construction: We commonly select the maximum posterior probability decision criterion to build our
classifier. We assume y has K categories, our equation will be:

2. Dataset
We will now use a dataset which contains the data of a social networking website users who have purchased a product
by clicking on the add or not. It has the following features:
User ID: Unique id of the user
Gender: Gender of the user
Age: Age of the user
EstimatedSalary: Estimated salary of the user

Purchased: 1 if purchased after seeing the advertisement else 0

3. Experiment
We will use the social networking website dataset to build a classification model using naïve bayes classifier to
predict whether a user will purchase the product after clicking on the advertisement. This model can be used to
target users and thereby reduce the costs associated with marketing.
We will read the dataset from csv and store it into a data frame. Then we will
perform our initial analysis of the data. We can see that all the features are integer
or float so will not require any labeling or one hot encoding them.

Then we will look for null values. We can tell that the data set
does not have any null values and will not require imputation
We will then plot histograms of our
dataset.
We will perform bivariate analysis using pair plot. Pairplot helps us
plot pairwise bivariate distributions in a dataset which helps us summaries
the data visually. The x & y axis have all the features plotting against the target variable Purchased and the diagonal
has the distribution of each element against the target variable Purchased.

We will also perform correlation analysis. The correlation coefficient gives
us the direction and degree of the relationship. Age and Purchased
correlation are the same direction and have a good correlation. Age and User ID attributes are in inverse direction and
have negligible correlation. Correlation is done to see in which direction the dependent variable will change when the
independent variable changes.

We will then plot Age and Estimated salary against the target variable Purchased to see the distribution.

We can tell that both the features affect the user’s decision to purchase a product.

We will store our features in X and target variable in Y and then split our dataset into test( 25 %) and training(7 5 %).
The we will perform feature scaling and then fit our training data to the Naive Bayes classifier.
The we will run prediction on our test data and plot it into a confusion matrix:

Visually from the confusion matrix we can tell that our model accuracy is pretty good.

Additionally, we can map our training set classification:

As well as our prediction classification:

From sklearn.metrics we can find the accuracy of our model as below:

4. References
https://towardsdatascience.com/introduction-to-naive-bayes-classifier-f5c202c97f
https://medium.com/analytics-vidhya/everything-you-need-to-know-about-na%C3%AFve-bayes-9a97cff1cba

Neural Network Backpropagation Experiment Report
1. Introduction
Neural networks are algorithms inspired from how the human brain functions. It works on processing the data in a way
similar to how neurons process our sensory observations in our brain. It takes in data and recognizes patterns, draws
out references and gives out an output.
They are also called Artificial Neural Networks (ANN) as they perform functions like human brain neurons but are not
natural. They are made to artificially mimic the functions of neuron. An ANN is made up of a large number of neurons
which work together to solve a problem.
ANN learn by making observations like humans. The are configured by making them learn for various problems like
classification, pattern recognition, Image recognition, etc. by using examples.

Layers: ANN consists of 3 types of layers usually:
Input unit: This layer takes raw input from the data.
Hidden unit: All the processing happens in the hidden unit using the raw data from
the input unit. The functioning depends on the input unit and the weights on the
connection from the input unit.
Output unit: It functions depending on the problem statement, for eg for
classification it will display the different classes. Its functioning depends on the
hidden unit and the weights on the connection from the hidden unit.

In a simple neural network the hidden layers are free to create their own representation of data. The weights between
the input and hidden layer determine when the hidden layer will be active.

Neuron: each hidden layer is made up of neurons. They are similar to neurons in Human and are also called nodes or
units. The neuron receives an input, learns and computes from it and sends an output. Every input node has a weight(w)
associated with it based on its importance. Hidden node applies a function(f) to the weighted sum of the inputs.

The above image compares a biological neuron with a computation neuron. x 1 and x 2 are inputs with weights w 1 & w 2
associated with the inputs. There is one more input 1 with a weight n associated with it.

Activation function:
The above neuron computes the output Y. The function used here, f, is a non linear function called activation function.
The activation function is used to introduce non-linearity to the neuron output. This conversion is necessary as in reality
as real data is not usually linear.

Each activation function takes a single input and performs mathematical operations. Some of the commonly used
activation functions are:
a. Sigmoid: It transforms the real valued input into between 0 & 1.
σ(x) = 1 / (1 + exp(−x) )

b. Softmax: It is an activation function which transforms the outputs into probabilities which sum up to 1.It
basically takes a real vector and transforms it into values between 0 & 1 such that the total sum is 1.
Probability (A) + Probability (B) = 1
c. Tanh: It transforms a real values within a range of -1 & 1

d. ReLu: Rectified Linear Unit takes in a real input and transforms it to a threshold
at zero.
f(x) = max(0, x)
Every neuron has two methods of propagation:

Forward Propagation:
In this, the weights are randomly assigned.
Lets assume weights to be w1,w2,w3.
Input: x2,x3 say 35,67 hours of study
Target: [1,0] where 1 is pass and 0 is fail
The output(V) from the node have activation f calculated as:
V = f ( 1 *w1 + x 2 w2 + x3w3)
= f ( 1 *w1 + 35 *w2 + 67 *w3)
Output from other nodes in hidden layer are also similarly calculated. Two
nodes with these calculations then feed to the output layer which helps us to calculate output of one node from two
different hidden nodes.
Lets hypothetically make an assumption that two nodes in output are 0 .4 & 0. 6.

These values are far off from the target of 0 or 1. Therefore, the network formed in the above image is false. To correct
this error we implement back propogation.

Backward Propagation:

After forward propagation node output is found out to be incorrect, errors are measured, and these errors are sent back
to the hidden layer using back propagation to calculate gradients.
Then we adjust all weights using optimization techniques like Gradient descend to adjust the weights aiming to reduce
the error in output unit/layer.
Formula’s needed for back propagation algorithm:

For the partial
derivatives

2.
For the final layer's
error term
3.
For the hidden
layers' error terms
4.
For combining the
partial derivatives
for each input-
output pair
5.
For updating the
weights
General Algorithm:
Step 1: Backward phase calculation:
Step a: For every input-output pair , store the calculated values in for each weight while
connecting node i in layer k- 1 another node j in layer k proceeding from output layer to the layer 1.
Step b: calculate the error from the final layer δ 1 m by the second equation
Step c: backpropagate the error terms in the hidden layer δ jk^ , from final hidden layer k = m- 1 and before, repeatedly
using third equation.
Step d: Calculate partial derivatives of each error Ed w.r.t wkij by the help of first equation.

Step 2: Combining individual gradient:

For every input-output pair, compute total gradient for all sets of input output pairs
by using fourth equation.

Step 3: Updating the weights: Update the weights using the learning rate α and total gradient using the
fifth equation.

From the example taken in forward propagation which gave us incorrect answer, we will now apply backward
propagation. The weights will now be adjusted to minimize the error. As shown in the below image the output will now
reduce to [0.2,-0.2] from [0.6,-0.4] previously found. This is closer to [1,0] and now our error has been reduced.

We will repeat this till forward and back propagation our output nodes come to [1,0].

2. Dataset
This is a dataset of electrical impulse measurement of freshly excised tissues samples of breast collected by NEB-
Instituto de Engenharia Biomédica, Porto, Portugal. It has the following features:
I 0 Impedivity (ohm) at zero frequency
PA500 phase angle at 500 KHz
HFS high-frequency slope of phase angle
DA impedance distance between spectral ends
AREA area under spectrum
A/DA area normalized by DA
MAX IP maximum of the spectrum
DR distance between I0 and real part of the maximum frequency point
P length of the spectral curve Class(Classes:car(carcinoma), fad (fibro-adenoma), mas (mastopathy), gla (glandular),
con (connective), adi (adipose))

3. Experiment
We will use the breast tissue dataset to classify the data into different classes using Neural net with the help of back
propagation. The target variable here is class.

For the initial analysis we will look
up for null or missing values in the
dataset. We will move on to the
analysis of the data. The data has 106
instances of electrical impedance
measurements of freshly excised
breast tissues. There are 9 features
and 1 target class data column.
There are a total of 6 classes.

From the pairplot we can analyse
That our target variable P & IO are
In a direct linear relationship.

From the correlation plot we can tell that IO is
highly correlated with our target variable. We can
also observe that DR & DA are highly correlated
and hence one of them can be removed from our
analysis as removing one of them will decrease
complexity without much affect to the result.
Hence, the most optimum features we can use are
IO, DA, A/DA, MAX IP, DR.

For more observations we can also
use the describe function for data
frame.
We will then normalize our dataset to
stabilize the gradient descend which
will allow us to use larger learning
rate and also help the model to
converge faster.

We will then labialize out target variable different classes from 'car','fad','mas','gla','con','adi' to 0,1,2,3,4,5 respectively.
Then we shall remove the ‘case#’ variable split our data into test and training in the ration of 25% & 75% respectively.

Next, we will build 6 vectors for our 6 classes in the target columns and create a vector ‘target’ for the results of the
values in train data frame.
Next we need to implement the ANN model for which we choose
Input layer neuron: number of features for training = 9
Hidden layer neuron: 12 using the formula in the image
Output layer neuron: number of classes = 6

We will assign random weights w1 & w2 initially using random function.

Next we will run forward propagation using the current random
weights applying the sigmoid activation function discussed above.
Sigmoid plot is shown in the figure.

We will then train the model by using an alpha rate of 0.2 and epoch
limit of 1000 and adding back propagation to the results from the forward
propagation initialized values.
We can see that our error after applying back propagation is 0.

Plotting the error:

Next, we will run our prediction on the dataset and check for its accuracy

We can see that we have an accuracy of 68.75% buy using backpropogation algorithm on ANN with learning rate of
0 .2 and 11 neurons on the hidden layer. Our simple ANN model can be represented as in the figure above with 9 Input
nodes, 11 hidden nodes and 6 output nodes

4. References:
https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%
0propagation,to%20the%20neural%20network%27s%20weights
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-
feedforward-neural-netw
https://towardsdatascience.com/a-step-by-step-guide-to-building-a-multiclass-classifier-for-breast-tissue-
classification-5b685d765e
https://purnasaigudikandula.medium.com/a-beginner-intro-to-neural-networks-543267bda3c
Gaussian Mixture Model Experiment Report
1. Introduction
Gaussian Mixture Model is a type of clustering algorithm. It is a probabilistic model used on normally distributed
clusters of data within a dataset. It doesn’t require knowing the which cluster a data point belongs to for training, it
learns this on its own and hence it is classified as an unsupervised clustering algorithm.
GMM is similar to k-means clustering, we can also say that K-means is a part of GMM.

Mixture model: mixture model is a combination of two different models. GMM is a mixture model of two Gaussian
Distributions with their weights. The sum of weights
One dimensional GMM: pi is the weight mixing coefficient of the model
Here the sum of pi should be 1 as probability cannot
be more than 1.

Multi-dimensional GMM:

When there are two or more Gaussian’s in data, each of
the gaussians have their own mean parameter and
covariance. It can be seen in the figure.
In the figure on the left we can see that if we fit one

gaussian, the distribution isn’t proper around the cluster.
We need two gaussians for the cluster to be dense
around the mean.

We can also overlap gaussians in a GMM. The
numbers in the graph on left are the weights. Image in
the right shows density in a 3D graph.
Expectation Maximization: This is done to find maximum likelihood of a hidden variable.
Algorithm:
Step1: Initialize the variables and compute the log likelihood.
Step2: We find the current estimator parameter by evaluating the posterior distribution of Z
Step3: Using this distribution of Z we evaluate the complete likelihood.
Step4: Maximizing the Q function from above:
Using previous guess, we computed the expected complete log likelihood
EM for GMM:
We assume that the hidden variables and latent variables (present in data) are equal. We need to find is the covariance,
mixing coefficient and mean.

We calculated the log likelihood of expected value above.
For M step we need to differentiate the
equation based on the guessed parameter:

Here the responsibility becomes
constant:
We will continue similarily for other parameters:
Initialize v^0 , calculate l^0 = log p(X|v^0 )
vm:
μkm =
-
-
We look for convergence and stop if lm - l(m-1) < ε
AIC: Akaike’s Information Criteria consists of log likelihood, we set it at maximum to find a good model and the
number of parameters. We consider the number of parameters ad ignore a better log likelihood as increasing the
number of parameter will increase the accuracy but will tend to over fit the data.
We need this number of parameter to use it to know how many gaussians we can fit in ou

AICi = -2LogLi + 2Pi
BIC: Bayesian Information Criteria is similar to AIC but instead of number of parameter, it has weights. We consider
the best fit for number of cluster in a BIC curve as to the point after which the graph is relatively flat.

BICi = - 2LogLi + Pi.logn
2. Dataset:
We will use a dry bean dataset which has observations from images of 7 different types of beans. This Multivariate
dataset has been created by Murat KOKLU, Faculty of Technology, Selcuk University, TURKEY.
Attributes of the dataset:
Area (A): The area of a bean zone and the number of pixels within its boundaries.
Perimeter (P): Bean circumference is defined as the length of its border.
Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
Minor axis length (l): Longest line that can be drawn from the bean while standing perpendicular to the main axis.
Aspect ratio (K): Defines the relationship between L and l.
Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
10.Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
11.Roundness (R): Calculated with the following formula: (4piA)/(P^2)
12.Compactness (CO): Measures the roundness of an object: Ed/L
13.ShapeFactor1 (SF1)
14.ShapeFactor2 (SF2)
15.ShapeFactor3 (SF3)
16.ShapeFactor4 (SF4)
17.Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)
3. Experiment:
We will use the dry beans dataset to cluster the data into different classes of dry beans using Gaussian mixture model.
We will load the csv data into a dataframe and then print some rows of the tables.
For the initial analysis we will lookup for null or missing values in the dataset.
We will move on to the analysis of the data. The data has 13611 instances observation of dry beans. There are 17
features and 1 target class data column i.e. Class. There are a total of 7 classes.
Building pair plot to observe relationship of features:
From the pairplot we can analyse that our target variable most of the seed dimension fields are in a linear relationship.

We will drop the ‘Bean ID’ column. The we will build a
correlation matrix.
Highly correlated features(we can ignore one from the
pair):
Area & Perimeter
Area & EquivDiameter
Perimeter & MajorAxisLenght
Perimeter & EquivDiameter
Eccentrivity & AspectRation
removing one of them from each pair will decrease
complexity without much affect to the result.
We will then normalize our dataset to bring
the numeric columns to a common scale:

Note: We may or may not standardize the data for GMM model as the optimization parameter can learn and fit
variance.
We then convert the normalized data to a data frame and perform PCA to reduce
dimensionality. We will select the components which give us maximum explanation of the
variance. Here 2 PCA explains enough data so we select it.
For finding the best fit cluster size we will use Silhouette score and BIC test:
From Silhouette: The highest peak is the best cluster size, i.e. 4
From BIC plot the best cluster size is the lowest, so the best number of cluster is 7.
We will initialize a GMM model with the data and the parameter n = 4 and n = 7 for 4 & 7 clusters respectively. As
GMM does not on its own find the number of clusters, we have to always use other tests like silhouette, AIC/BIC to
learn the best number of clusters.
We can now visualize our GMM clusters on the data:

K=4 (clusters) K= 7 (clusters)
4. References:
https://medium.com/swlh/ml-gmm-em-algorithm-647cf373cd5a
https://data.world/makeovermonday/2021w14/workspace/file?filename=Dry_Bean_Dataset.txt
https://ryanwingate.com/intro-to-machine-learning/unsupervised/gaussian-mixture-model-examples/



Hidden Markov Models Experiment Report

1. Introduction
Hidden Markov model(HMM) is a discrete sequential data modelling method which works on the assumption that the observed
events depend on a trait which are not directly observable. This is where the hidden part of the model comes from.
As for the Markov part, it comes from how we model the hidden states. We consider the Markov property from probability theory
which is also known as the memoryless, i.e. the next hidden states only depends on the current state and is not affected by any
other states.
HMM’s are applied in speech recognition, activity recognition from video, gesture tracking, gene tracing.

HMM is used as a tool to represent probability distributions of sequential data. In HMM we take a stochastic process which has
an observation Xt at a time t, here we have an unknown hidden state Zt. We assume that Zt satisfies the Markov property and Zt
at time t is hence only dependent on the previous state Zt-1 at time t-1. This is the first order of Markov model. Hence, nth Markov
model would depend on the previous n states.
Below is an image of a Bayesian network representing the first order HMM. Shaded states are the hidden states.

Hence, Joint distribution of a sequence of states and order for first order HMM:

2. Dependence
HMM is characterized by the below five elements:
a. Number of states(N): The number of states that a hidden Markov model has is also related to the problem being modeled.
We will encode the state Zt at time t as a K x 1 vector of binary numbers.
b. Number of Distinct observations(Ω): Observations are the number of disctinct observations in the experiment. We will
encode Xt at a time t as Ω x 1 vector of binary numbers, where the only non-zero element is the only lth element.
c. State transition model(A): This is a N x N matrix with elements Aij containing the probability of transitioning from Zt-1,i to
Zt,j in one time steps. It is written as:

Row sum of each row in the State transition matrix sums to 1, hence it is a stochastic matrix.
Above is a 3 state stochastic matrix for which
If in one jump a state can be reached from another state the value of Aij will be > 0
else it will be zero. Hence, the state transition model for the above diagram will be
Conditional probability can be written as
We can take its log and write it as :
d. Observation model(B): It is a matrix that describes the emission probabilities of the observable process(with elements Xt,k.
Elements of this matrix are given as Bk,j at state Zt.j, and the matrix B is calculates as B = Ω× K.

Elements of B i.e. Bk,j can be written as:
Conditional probabilities can be written as:
We can rake its log and write it as:
e. Initial state distribution(π): Initial distribution of the states(π) is used for modeling where πi = P(Z1i=1| π) =

These 5 parameters can be used to specify the HMM model as, λ = (A, B, π).
3. Three classes of Problems that can be solved using HMM
a. Known a set of observations Ω and 3 model parameters 𝝅, A and B, we can find out the probability of occurrence of X. We
can find this by using forward filtering.
b. Known a set of observations Ω and 3 model parameters 𝝅, A and B, we can find out the optimal set of hidden states Z that
result in X. We can find this by using Viterbi algorithm.
c. Known a set of observations Ω we can find out the optimal parameters 𝝅, A and B using Baum-Welch algorithm.
4. Algorithms
a. Forward Backward Algorithm: Forward-backward algorithm is a dynamic programming algorithm which uses belief
propagation. It computes the filtered and smoothed marginals that are used to perform inference, sequence classification,
MAP estimation, anomaly detection and clustering.
b. Viterbi Algorithm: To compute the set of most probable sequence of hidden states(problem b) we will use the Viterbi
algorithm. It uses the trellis diagram of HMM to compute the shortest path. Trellis diagram connects each state of the next
time step in the model. Next, we use the forward-backward algorithm but using max-product instead of sum-product
algorithm.
c. Baum-Welch Algorithm: Baum-Welch Algorithm is a dynamic programming algorithm which uses Expectation
Maximization algorithm to tune parameters for HMM. It is used to find optimized values of A, B & 𝝅 such that the model
data overlaps with the actual data.
5. Limitation:
HMM model considers only the last state. If we want to consider the last 2 states the training time grows exponentially
considering that we have to build a set of cartesian multiplication of all states. Hence, our time complexity will become N^2 for
training. For this we can use RNN as it can use continuous states.

6. Dataset:
We will use the silver dataset from Kaggle. It has the data of daily silver price changes from January 2000 to September 2022.
Feature:
Date: Date of data instance
Open: Opening price of Silver per ounce in US Dollar
High: Highest price of Silver per ounce in US Dollar
Low: Lowest price of Silver per ounce in US Dollar
Close: Closing price of Silver per ounce in US Dollar
Volume: Ounces of Silver sold
Currency: Currency of transaction

7. Experiment
We will analyze historical silver prices using hmmlearn, downloaded from: https://www.kaggle.com/datasets/psycon/daily-silver-
price-historical-data.

We will import the necessary libraries and also import hmmlearn. We will calculate daily change in the silver prices as it is a better
way of modelling stock like data which changes daily. We will consider data after 2008 , ignoring the 2008 financial crisis.

We will plot the silver price and change in silver price from 2008.

Then, we will use a Gaussian model and fit the daily silver price change with 3 hidden states. We select 3 hidden states as we
expect 3 different variety of data, namely high, medium low.
Fitting will result in three unique hidden states

We see that for this our data, the model starts mostly with state 1.

Next we look at our Gaussian Mean:

For state 1, the mean in 0. 0 27 for state 2 our mean is -0.013, for state 3 our mean in -0.013. State 0 & 2 have similar mean; hence
our model might not be good at representing the data.
Next we will look at the Gaussian covariance:

As our data is one dimensional, our covariance matrix will be scalar(not vector).
For state 1 the covariance is 0.026, for state 0 the covariance is 0.12 7 and for state 2 the covariance 0.795. From this we can tell
that for state 1 the volatility is low and for state 2 the volatility is high.

Next we will plot our models prediction :

We can tell that states 1,0,2 represent low, medium and high volatility.
From our plot we can see, the period of high volatility due to recession during 2011 - 2012 and the high volatility during 2020- 2021
due to covid crisis.
We can also tell that the cost of silver increases during these highly volatile periods as investors purchase commodities instead of
equity when the markets collapse.

References:
https://www.kaggle.com/datasets/psycon/daily-silver-price-historical-data
https://scholar.harvard.edu/files/adegirmenci/files/hmm_adegirmenci_2014.pdf
https://medium.com/@natsunoyuki/hidden-markov-models-with-python-c026f778dfa
https://medium.com/analytics-vidhya/baum-welch-algorithm-for-training-a-hidden-markov-model-part- 2 - of-the-hmm-series-
d0e393b4fb
Random Forest Classifier Models Experiment Report
1. Introduction
Random Forest algorithm is a type of supervised machine learning algorithm. It has
two variations: Classifier and Regressor.
Random forest Classifier uses ensemble learning i.e. many algorithms are run or
one algorithm is run multiple times to create several decision trees from random
subsets of the data. It uses the average from these decision tree predictions to
predict the final class of the test object. This selection from several decision trees
is known as voting.
As Random Forest algorithm uses several random decision trees, it is known as
Random Forest algorithm.

2. Algorithm
Random Forest algorithm works in the following steps:
Starts by selecting random samples from the data
It constructions decision trees for all the samples and gets prediction results from all the samples.
It performs voting for each predicted results.
It ends by selecting the most voted prediction result.

If we use a high depth decision tree we will get low bias and high variance, which means that our model will be overfitted and
predictions of unknown data will not be accurate. To solve this problem we use Random forest by combining many decision trees
instead of one decision tree.

Gini Index:
While splitting the data into decision trees, the algorithm uses the Gini index to get a pure split. A pure split is a split in which one
node has one type of categorical data and another has a different.
Here, P+ is the probability of a positive class and
P_ is the probability of a negative class.

We, choose the decision tree feature to split which gives us the minimum Gini index(impurity) for the root.

Feature Selection: We can perform an initial run of the model to get the
important features that we should use for the main model.

3. Experiment
We will use the breast tissue dataset to classify the data into different classes using Random forest classifier. The target variable
here is class.

For the initial analysis we will look
up for null or missing values in the
dataset. We will move on to the
analysis of the data. The data has 106
instances of electrical impedance
measurements of freshly excised
breast tissues. There are 9 features
and 1 target class data column.
There are a total of 6 classes.

From the pairplot we can analyse
That our target variable P & IO are
In a direct linear relationship.

From the correlation plot we can tell that IO is
highly correlated with our target variable. We can
also observe that DR & DA are highly correlated
and hence one of them can be removed from our
analysis as removing one of them will decrease
complexity without much affect to the result.
Hence, the most optimum features we can use are
IO, DA, A/DA, MAX IP, DR.

For more observations we can also use the
describe function for data frame.
Then we shall remove the ‘case#’ variable split our data into test and training in the ration of 2 0 % & 80 % respectively, while also
deleting the ‘class’ column, our target data, from the test data.

We will then perform Hyper-Parameter Optimization using GridSearchCV to find the best parameters to use in our model.

Next we will train our model with the training data and use feature_importances_ from
sklearn to find the best features for our data and plot the importance of each feature.

We can tell that the feature ‘P’ is the most important and the feature HFS is the
least. We will remove ‘HFS’ from out dataset.

Using ‘HFS’ removed we will again train our model with the best parameters given by grid search.

We got an accuracy score of 0.7 73 , this does not make much sense so we will plot the results in a confusion matrix.

Our confusion matrix looks good and acceptable, there are not many false
positive and false negatives.

From sklearn.metrics we can find the accuracy of our model as below:

4. References
1. https://medium.com/analytics-vidhya/random-forest-classification-and-its-mathematical-
implementation-1895a7bb743e
2. https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial
3. https://www.analyticsvidhya.com/blog/2021/10/an-introduction-to-random-forest-
algorithm-for-beginners/
Autoencoder Experiment Report
1. Introduction
Autoencoder is a type of Artificial neural network which has the ability to learn the input representation. It can encode and
compress data efficiently and then learn from the process how to reverse it and construct the input from the compressed
data.
There are three main parts in an autoencoder: encoder, decoder and latent block(or bottle neck. The encoder extracts the
best features, the latent space is used for storage of these features, and the decoder does the reverse of encoder and
reconstructs an image from using the features.
One application of Autoencoder is denoising data. Data can lose quality over transmission on the internet. Autoencoders can
be used to get back the lost quality.
2. Types of Autoencoder:
Denoising Autoencoder: in this encoder we create a copy of the
input data with some noise added. This avoids the
generalization of the input and makes the models learn features
from the input data and train to recover the original data.
It minimizes the loss function between the output and the noisy
input. It is easy to setup. The only drawback is that his model
does not memorize the training as the input and output keeps
changing
2. Sparse Autoencoder: in this encoder, we have sparsity penalty in the
training criterion. There are two ways to add sparsity: L1 regularization and
KL-divergence. Adding sparsity helps to activate only fewer but insightful
nodes, which helps the model learn from latent representations instead of
redundant nodes. This also prevents overfitting.
Deep Autoencoder: It works by creating two identical deep belief networks: one for
encoding and one for decoding. Unsupervised layer by layer pretraining is used for this
model. Each is a Restricted Boltzmann Machine which is the base of deep belief
network, a binary transform is performed after each layer. Encoding is fast using Deep
autoencoder and they are used in topic modeling across a set of documents.
Contractive Autoencoder: It is just like sparse autoencoder, but here the regularizer used is the calculated by the
Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input. This model is robust, and
less sensitive to variations.
Undercomplete Autoencoder: They have a small number of hidden layers which helpt to capture the most important
features in the data. They do not need any regularization as they maximize probability.
Convolutional Autoencoder: Convolutional Autoencoder uses a
mix sum of signals to reconstruct the input data rather than taking
them one at a time like other models. It uses convolution neural
network, which is better at retaining connected information
between the data.
3. Experiment
We will use the ‘Labeled Faces in the Wild’ dataset. It contains the photographs of faces used for face recognition. We will
use this dataset to build a model for image resolution enhancement.
First, we will download the data set using wget and extract it.
We will reduce the size of our images as per the limitations of our device and to reduce the training time. I have set it at
80x80.
Then we will split the date into training and test. We will keep aside the test dataset and used it for model evaluation.
We then reduce the resolution of the images to 4 0% so as to add noise in the training data, which we will later enhance
using our model. We will use this low resolution image dataset as training data for our model.
We will then define our encoding model.
Our encoder consists of 2 Convolution neural networks followed by a max pool layer and then again a Convolution neural
network. We have used the activation function as relu for all layers.

We will then define our decoding model.

Our decoder consists of 1 Convolution neural network layer followed by an up-sampling layer and then 2 Convolution
neural networks. We have used the activation function as relu for all layers.

Our model summary is:

I have set the model parameters as epochs = 5 and batch size = 256 depending on the limitations of my machine.

Now we can use the trained model on our test dataset:

4. References
https://www.analyticsvidhya.com/blog/2020/02/what-is-autoencoder-enhance-image-resolution/
https://medium.com/ai%C2%B3-theory-practice-business/understanding-autoencoders-part-i-116ed2272d
https://blog.francium.tech/enhance-images-with-autoencoders-58afa4fd638f
https://medium.com/@syoya/what-happens-in-sparse-autencoder-b9a5a69da5c
http://vis-www.cs.umass.edu/lfw/lfw.tgz
AdaBoost Experiment Report
1. Introduction
Adaptive Boosting is a machine learning algorithm used in ensemble method. It iteratively combines and learns from the
mistakes of weak classifiers and promote them into strong ones. It can use any classifier but the most commonly used is
Decision tree with one 1. This is called a decision stump.
2. Algorithm
Steps for the algorithm:
A decision stump is created which is a weak classifier on the training data based on the weights and are given equal
weights for the initial stump.
Equation for assigning weights:
N: number of datapoints
Decision stumps are created for each variable. Each decision stump is analyzed to see how well it classifies the data to
the target class. This analysis is done by Gini index of each stump.
Here, P+ is the probability of a positive class and
P_ is the probability of a negative class.
We increase the weights of the stumps which classified incorrectly, this in turns make them classify correctly in the
next decision stump. More accurate accuracy stumps are given more weights as well.
We iterate step 2 and 3 until all the data has been correctly classified.
ADA Boosting steps with mathematics:
3. Experiment
We are using asteroseismology data which has the distinct features of oscillations of 6008 Kepler stars and using it to
predict whether it is a Red Giant or a Helium burning star using ADA bosting.
Feature:
POP: Target Variable, if 0: Red giant star & 1: Helium burning star
Dnu: Mean large frequency separation of modes with the same degree and
consecutive order
Numax: Frequency of maximum oscillation power
Epsilon: Location of the l=0 mode
We will read the dataset from csv and store it into a data frame. Then we will perform our initial analysis of the data. We can see
that all the features are integer or float so will not require any labeling or one hot encoding them.
Then we will look for null values. We can tell that the data set does not have any null values and will not
require imputation.

We will perform bivariate analysis using pair plot. Pairplot helps us plot
pairwise bivariate distributions in a dataset which helps us summaries the data
visually. The x & y axis have all the features plotting against the target variable
POP and the diagonal has the distribution of each element against the target
variable POP.

Then we will check for the count of each
classification in the target variable POP to check if
the data is not highly imbalanced. Our data is not
highly imbalanced.
We will then plot the correlation for each variable in the dataset.
This helps us find highly correlated features which we can remove
from our model training as they will not affect our training by much.

We will then split our dataset into test(30%) and training(70%)

We train the ADA boost model from our training set with n_estimators = 100

Now we will use our model to predict our test data and evaluate it using a confusion matrix:

And check our accuracy scores:

Our ROC/AUC curve fits as: (^)
We can see that we have got a good accuracy score although it is a little overfit and our AUC is close to one so we can say that
our model is good for use

4. References
https://blog.paperspace.com/adaboost-optimizer/
https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/
https://towardsdatascience.com/adaboost-for-dummies-breaking-down-the-math-and-its-equations-into-simple-terms-
87f439757dcf
https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c
This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports
