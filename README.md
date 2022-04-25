# prStocks
Stock Market Prediction  using Machine Learning  and  Setimental  Analysis(Deep learning Algorithem )
Steps :  1  Upload   the    code 
         2 :  Train the  data  over various  multiple  models  and   real  time  data  sets 
         3:  Test  the   data  set   over the  cases  and  use  the  for the  result
         4 : Matplot  lib    use to   print  the   graph  of the  result  outcome 

Lstm:
Long Short Term Memory networks — usually just called LSTMs — are a special kind of RNN, capable of learning long-term dependencies. Refined and popularized by many people in following work. They work tremendously well on a large variety of sequence modelling problems, and are now widely used. LSTMs are explicitly designed to avoid the
long-term dependency problem. Remembering information for long periods of time is their default behavior.

Sequential:
Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer. Each layer has weights that correspond to the layer that follows it.

Dense:
The dense layer is a neural network layer that is connected deeply,which means each neuron in the dense layer receives input from all neurons of its previous layer. The dense layer is found to be the most commonly used layer in the models.In the background, the dense layer performs a matrix-vector multiplication. The values used in the matrix are actually parameters that can be trained and updated with the help of backpropagation.

Sentimental  Analysis:
Twitter   Sentiment   Analysis   means,   using   advanced   text   mining   techniques   to   investigate  t he   sentiment   of   the   text   (here,   tweet)  w ithin   the   sort   of   positive,   negative,   and  n eutral.   it's   also   called   Opinion  M ining,  i s   primarily   for   analyzing   conversations,   opinions,   and   sharing   of  v iews   (all  w ithin   the   sort  o f  t weets)  f or   deciding   business   strategy,   political  a nalysis,   and   also   for   assessing   public   actions.  S entiment   analyses   are   often   want   to   identify   trends   within   the   content   of  t weets,   which  a re   then   analyzed  b y   machine  l earning   algorithms.   Sentiment   analysis   is   a  c rucial   tool   within   the   eld   of  s ocial   media   marketing   because   it'll  d iscuss   how   it   will   be   accustomed   to   predict   the   behavior   of   a   user's  o nline  p ersona.   Sentiment  a nalysis   is   employed   to   investigate   the   sentiment   of   a   given   post   or   investigate   any  g iven  t opic.In   fact,   it's   one   of   the  f oremost   popular tools in social media marketing.   
Text   understanding   could   be   a   signi cant   problem   to   resolve.   One  a pproach   may   well  b e   to  r ank   the   importance   of   sentences   within   the  t ext  t hen   generate   a   summary  f or  t he   text   supported  b y   the   important   numbers. 
Twitter is a popular social networking website where members create and interact with messages known as   
“tweets”. This serves as a means for individuals to express their thoughts or feelings about di erent subjects.   Various di erent parties such as consumers and marketers have done sentiment analysis on such tweets to  gather insights into products or to conduct market analysis. Furthermore, with the recent advancements in   machine learning algorithms, I was able to improve the accuracy of our sentiment analysis predictions. In   this report, I will attempt to conduct sentiment analysis on “tweets” using various 	 di erent machine   learning algorithms.attempted to classify the polarity of the tweet where it is either positive or negative. If   the tweet has both positive and negative elements, the more dominant sentiment should be picked as the   nal label.     
I used the dataset from Kaggle 	  which was crawled and labeled positive/negative. The data provided comes   with emoticons, usernames and hashtags which are required to be processed and  converted into a standard   form. I also need to extract useful features from the text such as unigrams and bigrams which is a form of   representation of the “tweet”   

Naive Bayes is a simple model which can be used for text classification. In this model, the class cˆ is assigned to a tweet t, where
 P(c|t) n
P(c|t) ∝ P(c)YP(fi|c)
i=1
In the formula above, fi represents the i-th feature of total n features. P(c) and P(fi|c) can be obtained through maximum likelihood estimates.
6
Maximum Entropy : 
Maximum Entropy Classifier model is based on the Principle of Maximum Entropy. The main idea behind it is to choose the most uniform probabilistic model that maximizes the entropy, with given constraints. Unlike Naive Bayes, it does not assume that features are conditionally independent of each other. So, we can add features like bigrams without worrying about feature overlap. In a binary classification problem like the one we are addressing, it is the same as using Logistic Regression to find a distribution over the classes. The model is represented by
 
Here, c is the class, d is the tweet and λ is the weight vector. The weight vector is found by numerical optimization of the lambdas so as to maximize the conditional probability.
3.4.3	Decision Tree
Decision trees  =>are a classifier model in which each node of the tree represents a test on the attribute of the data set, and its children represent the outcomes. The leaf nodes represents the final classes of the data points. It is a supervised classifier model which uses data with known labels to form the decision tree and then the model is applied on the test data. For each node in the tree the best test condition or decision has to be taken. We use the GINI factor to decide the best split. For a given node t, GINI(t) = 1 − Pj[p(j|t)]2, where p(j|t) is the relative frequency of class j at node t, and  number of records at child i, n = number of records at node p)indicates the quality of the split. We choose a split that minimizes the GINI factor.
3.4.4	Random Forest
Random Forest  => is an ensemble learning algorithm for classification and regression. Random Forest generates a multitude of decision trees classifies based on the aggregated decision of those trees. For a set of tweets x1,x2,...xn and their respective sentiment labels y1,y2,...n bagging repeatedly selects a random sample (Xb, Yb) with replacement. Each classification tree fb is trained using a different random sample (Xb, Yb) where b ranges from 1...B. Finally, a majority vote is taken of predictions of these B trees.
3.4.5	XGBoost
Xgboost is a form of gradient boosting algorithm which produces a prediction model that is an ensemble of weak prediction decision trees. We use the ensemble of K models by adding their outputs in the following manner
 
where F is the space of trees, xi is the input and yˆi is the final output. We attempt to minimize the following loss function
 
where Ω is the regularisation term.
3.4.6	SVM
SVM, also known as support vector machines, is a non-probabilistic binary linear classifier. For a training set of points (xi,yi) where x is the feature vector and y is the class, we want to find the
7
maximum-margin hyperplane that divides the points with yi = 1 and yi = −1. The equation of the hyperplane is as follow
w · x − b = 0
We want to maximize the margin, denoted by γ, as follows
maxγ,s.t.∀i,γ ≤ yi(w · xi + b) w,γ
in order to separate the points well.
3.4.7	Multi-Layer Perceptron
MLP or Multilayer perceptron is a class of feed-forward neural networks, which has atleast three layers of neurons. Each neuron uses a non-linear activation function, and learns with supervision using backpropagation algorithm. It performs well in complex classification problems such as sentiment analysis by learning non-linear models.
3.4.8	Convolutional Neural Networks
Convolutional Neural Networks or CNNs are a type of neural networks which involve layers called convolution layers which can interpret spacial data. A convolution layers has a number of filters or kernels which it learns to extract specific types of features from the data. The kernel is a 2D window which is slided over the input data performing the convolution operation. We use temporal convolution in our experiments which is suitable for analyzing sequential data like tweets.
3.4.9	Recurrent Neural Networks
Recurrent Neural Network are a network of neuron-like nodes, each with a directed (one-way) connection to every other node. In RNN, hidden state denoted by ht acts as memory of the network and learns contextual information which is important for classification of natural language. The output at each step is calculated based on the memory ht at time t and current input xt. The main feature of an RNN is its hidden state, which captures sequential dependence in information. We used Long Term Short Memory (LSTM) networks in our experiments which is a special kind of RNN capable of remembering information over a long period of time.
4	Experiments
We perform experiments using various different classifiers. Unless otherwise specified, we use 10% of the training dataset for validation of our models to check against overfitting i.e. we use 720000 tweets for training and 80000 tweets for validation. For Naive Bayes, Maximum Entropy, Decision Tree, Random Forest, XGBoost, SVM and Multi-Layer Perceptron we use sparse vector representation of tweets. For Recurrent Neural Networks and Convolutional Neural Networks we use the dense vector representation.


Baseline:
For a baseline, we use a simple positive and negative word counting method to assign sentiment to a given tweet. We use the Opinion Dataset of positive and negative words to classify tweets. In cases when the number of positive and negative words are equal, we assign positive sentiment. Using this baseline model, we achieve a classification accuracy of 63.48% on Kaggle public leaderboard.
4.2	Naive Bayes
We used MultinomialNB from sklearn.naive_bayes package of scikit-learn for Naive Bayes classification. We used Laplace smoothed version of Naive Bayes with the smoothing parameter α set to its default value of 1. We used sparse vector representation for classification and ran experiments using both presence and frequency feature types. We found that presence features outperform frequency features because Naive Bayes is essentially built to work better on integer features rather
8
than floats. We also observed that addition of bigram features improves the accuracy. We obtain a best validation accuracy of 79.68% using Naive Bayes with presence of unigrams and bigrams. A comparison of accuracies obtained on the validation set using different features is shown in table
5.
Maximum Entropy: 
The nltk library provides several text analysis tools. We use the MaxentClassifier to perform sentiment analysis on the given tweets. Unigrams, bigrams and a combination of both were given as input features to the classifier. The Improved Iterative Scaling algorithm for training provided better results than Generalised Iterative Scaling. Feature combination of unigrams and bigrams, gave better accuracy of 80.98% compared to just unigrams (79.34%) and just bigrams (79.2%).
For a binary classification problem, Logistic Regression is essentially the same as Maximum Entropy. So, we implemented a sequential Logistic Regression model using keras, with sigmoid activation function, binary cross-entropy loss and Adam’s optimizer achieving better performance than nltk. Using frequency and presence features we get almost the same accuracies, but the performance is slightly better when we use unigrams and bigrams together. The best accuracy achieved was 81.52%. A comparison of accuracies obtained on the validation set using different features is shown in table 5.
4.4	Decision Tree
We use the DecisionTreeClassifier from sklearn.tree package provided by scikit-learn to build our model. GINI is used to evaluate the split at every node and the best split is chosen always. The model performed slightly better using the presence feature compared to frequency. Also using unigrams with or without bigrams didn’t make any significant improvements. The best accuracy achieved using decision trees was 68.1%. A comparison of accuracies obtained on the validation set using different features is shown in table 5.
Random Forest :
We implemented random forest algorithm by using RandomForestClassifier from sklearn.ensemble provided by scikit-learn. We experimented using 10 estimators (trees) using both presence and frequency features. presence features performed better than frequency though the improvement was not substantial. A comparison of accuracies obtained on the validation set using different features is shown in table 5.
4.6	XGBoost
We also attempted tackling the problem with XGboost classifier. We set max tree depth to 25 where it refers to the maximum depth of a tree and is used to control over-fitting as a high value might result in the model learning relations that are tied to the training data. Since XGboost is an algorithm that utilises an ensemble of weaker trees, it is important to tune the number of estimators that is used. We realised that setting this value to 400 gave the best result. The best result was 0.78.72 which came from the configuration of presence with Unigrams + Bigrams.
SVM : 
We utilise the SVM classifier available in sklearn. We set the C term to be 0.1. C term is the penalty parameter of the error term. In other words, this influences the misclassification on the objective function. We run SVM with both Unigram as well Unigram + Bigram. We also run the configurations with frequency and presence. The best result was 81.55 which came the configuration of frequency and Unigram + Bigram.
4.8	Multi-Layer Perceptron
We used keras with TensorFlow backend to implement the Multi-Layer Perceptron model. We used a 1-hidden layer neural network with 500 hidden units. The output from the neural network

Image:


