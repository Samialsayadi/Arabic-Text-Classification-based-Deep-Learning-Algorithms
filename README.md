# Arabic Text Classification based on Machine learning Deep Learning Algorithms
In arabic text classification, texts have to be transformed into numeric representations suitable for the learning algorithms. in this studies, we used TF-IDF with Machine learning algorithms and Embedding words with Deep learnings. In the present study, text classification is comparative study based on a Machine learning methods and Deep learning Methods.  The current work has proved the ability to classify collections of Arabic or English text documents successfully. It showed approximately 98% sav- ings in vector space and 2% performance improvement compared to the best recorded results on Arabic dataset Aljazeera News.


# Dataset
Aljazeera news 5 classes dataset (Alj-News5). Alj-News5 is another different dataset that contains 1500 documents for news articles. Each document is labeled with one of the following five classes {‘Art’, ‘Economic’, ‘Politics’, ‘Science’, and ‘Sport’}. Alj-News5 was used by other researchers . A comparison of classification results of the current approach against other research works on Alj-News5 will be pre- sented in the experiments.

# Step 1: Text Classification based on Machine Learning 

Text classification is a smart classification of text into categories. And, using machine learning to automate these tasks, just makes the whole process super-fast and efficient. Artificial Intelligence and Machine learning are arguably the most beneficial technologies to have gained momentum in recent times. in this study, we used 8 algorithms with competitive results. 

# 1. Naive Bayes Classifier
Naïve Bayes text classification has been used in industry and academia for a long time (introduced by Thomas Bayes between 1701-1761).  Naive Bayes Classifier (NBC) is generative model which is widely used in Information Retrieval.  We start with the most basic version of NBC which developed by using term-frequency (Bag of Word) fetaure extraction technique by counting number of words in documents

```
     precision    recall  f1-score   support

         Art     0.9630    0.9811    0.9720        53
    Economic     0.9310    0.9818    0.9558        55
    Politics     0.9833    0.9077    0.9440        65
     Science     1.0000    1.0000    1.0000        58
       Sport     0.9857    1.0000    0.9928        69

    accuracy                         0.9733       300
   macro avg     0.9726    0.9741    0.9729       300
weighted avg     0.9739    0.9733    0.9731       300
```
# 2. Rocchio classification
The first version of Rocchio algorithm is introduced by rocchio in 1971 to use relevance feedback in querying full-text databases. 
Rocchio's algorithm builds a prototype vector for each class which is an average vector over all training document vectors that belongs to a certain class. Then, it will assign each test document to a class with maximum similarity that between test document and each of the prototype vectors.
When in nearest centroid classifier, we used for text as input data for classification with tf-idf vectors, this classifier is known as the Rocchio classifier.
```
 precision    recall  f1-score   support

         Art     0.8704    0.9216    0.8952        51
    Economic     0.6724    1.0000    0.8041        39
    Politics     1.0000    0.6897    0.8163        87
     Science     1.0000    1.0000    1.0000        58
       Sport     0.9286    1.0000    0.9630        65

    accuracy                         0.8967       300
   macro avg     0.8943    0.9222    0.8957       300
weighted avg     0.9199    0.8967    0.8954       300
```

# 3.Boosting and Bagging
Boosting
Boosting is a Ensemble learning meta-algorithm for primarily reducing variance in supervised learning. It is basically a family of machine learning algorithms that convert weak learners to strong ones. Boosting is based on the question posed by Michael Kearns and Leslie Valiant (1988, 1989) Can a set of weak learners create a single strong learner? A weak learner is defined to be a Classification that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.

```
     precision    recall  f1-score   support

         Art     0.9630    0.9286    0.9455        56
    Economic     0.8793    0.8947    0.8870        57
    Politics     0.9333    0.8750    0.9032        64
     Science     0.9310    1.0000    0.9643        54
       Sport     0.9571    0.9710    0.9640        69

    accuracy                         0.9333       300
   macro avg     0.9328    0.9339    0.9328       300
weighted avg     0.9337    0.9333    0.9330       300
```

# 4. Bagging

```
              precision    recall  f1-score   support

         Art     0.8704    0.9592    0.9126        49
    Economic     0.8276    0.9412    0.8807        51
    Politics     0.9333    0.7671    0.8421        73
     Science     1.0000    0.9831    0.9915        59
       Sport     0.9571    0.9853    0.9710        68

    accuracy                         0.9200       300
   macro avg     0.9177    0.9272    0.9196       300
weighted avg     0.9236    0.9200    0.9188       300
```
# 5. K-nearest Neighbor
The k-nearest neighbors algorithm (kNN) is a non-parametric technique used for classification. This method is used in Natural-language processing (NLP) as a text classification technique in many researches in the past decades.
```
       precision    recall  f1-score   support

         Art     0.8889    0.9231    0.9057        52
    Economic     0.8276    0.9231    0.8727        52
    Politics     0.9000    0.7826    0.8372        69
     Science     1.0000    0.9831    0.9915        59
       Sport     0.9571    0.9853    0.9710        68

    accuracy                         0.9167       300
   macro avg     0.9147    0.9194    0.9156       300
weighted avg     0.9181    0.9167    0.9159       300
```
# 6. Support Vector Machine (SVM)
Support vector machines is an algorithm that determines the best decision boundary between vectors that belong to a given group (or category) and vectors that do not belong to it. we applied SVM based TF-IDF vectors and obtain the high result compare with other machine learning algorithms.
```
              precision    recall  f1-score   support

         Art     0.9630    1.0000    0.9811        52
    Economic     0.9655    0.9825    0.9739        57
    Politics     0.9667    0.9355    0.9508        62
     Science     1.0000    1.0000    1.0000        58
       Sport     1.0000    0.9859    0.9929        71

    accuracy                         0.9800       300
   macro avg     0.9790    0.9808    0.9798       300
weighted avg     0.9801    0.9800    0.9799       300
```
# 7. Decision Tree
One of earlier classification algorithm for text and data mining is decision tree. Decision tree classifiers (DTC's) are used successfully in many diverse areas of classification. The structure of this technique includes a hierarchical decomposition of the data space (only train dataset). Decision tree as classification task was introduced by D. Morgan and developed by JR. Quinlan. The main idea is creating trees based on the attributes of the data points, but the challenge is determining which attribute should be in parent level and which one should be in child level. To solve this problem, De Mantaras introduced statistical modeling for feature selection in tree.

```
      precision    recall  f1-score   support

         Art     0.6852    0.6852    0.6852        54
    Economic     0.7586    0.7857    0.7719        56
    Politics     0.8167    0.6712    0.7368        73
     Science     0.7414    0.9348    0.8269        46
       Sport     0.9286    0.9155    0.9220        71

    accuracy                         0.7933       300
   macro avg     0.7861    0.7985    0.7886       300
weighted avg     0.7971    0.7933    0.7917       300
```
# 8. Conditional Random Field (CRF)
Conditional random fields (CRFs) are a class of statistical modeling methods often applied in pattern recognition and machine learning and used for structured prediction. Whereas a classifier predicts a label for a single sample without considering "neighbouring" samples, a CRF can take context into account. To do so, the predictions are modelled as a graphical model, which represents the presence of dependencies between the predictions. What kind of graph is used depends on the application. For example, in natural language processing, "linear chain" CRFs are popular, for which each prediction is dependent only on its immediate neighbours. 
```
    precision    recall  f1-score   support

         Art     0.7407    0.6667    0.7018        60
    Economic     0.7759    0.7500    0.7627        60
    Politics     0.8167    0.7778    0.7967        63
     Science     0.7586    0.9565    0.8462        46
       Sport     0.9286    0.9155    0.9220        71

    accuracy                         0.8100       300
   macro avg     0.8041    0.8133    0.8059       300
weighted avg     0.8109    0.8100    0.8082       300
```
# Step 2: Text Classification based on Deap Learning 
Deep Neural Networks architectures are designed to learn through multiple connection of layers where each single layer only receives connection from previous and provides connections only to the next layer in hidden part. The input is a connection of feature space (As discussed in Section Feature_extraction with first hidden layer. 
In this Comparative study, we used the commen Deep learning methods (CNN, RNN, DNN, RCNN)to classify arabic text. and we have trained algorithms by only 15 Epoch. The result showed DNN algorithm was the high result compare other algorithms' results.  

# convolutional neural network 
```
        precision    recall  f1-score   support

           0       1.00      0.92      0.96        62
           1       0.93      0.89      0.91        61
           2       0.81      0.93      0.87        60
           3       1.00      0.98      0.99        59
           4       1.00      1.00      1.00        58

    accuracy                           0.94       300
   macro avg       0.95      0.94      0.95       300
weighted avg       0.95      0.94      0.94       300
```
# Recurrent neural networks
```
           precision    recall  f1-score   support

           0       0.93      0.81      0.86        67
           1       0.68      0.81      0.74        57
           2       0.70      0.90      0.79        58
           3       0.72      0.52      0.60        60
           4       1.00      0.98      0.99        58

    accuracy                           0.80       300
   macro avg       0.81      0.80      0.80       300
weighted avg       0.81      0.80      0.80       300
```
# Deep neural networks
Deep Neural Networks (DNN), we used input layer  tf-ifd, while the output layer houses neurons equal to the number of classes for multi-class classification and only one neuron for binary classification. we have multi-class DNNs where each learning model is generated randomly (number of nodes in each layer as well as the number of layers are randomly assigned). Our implementation of Deep Neural Network (DNN) is basically a discriminatively trained model that uses standard back-propagation algorithm and sigmoid or ReLU as activation functions. The output layer for multi-class classification should use Softmax.


```
         precision    recall  f1-score   support

             Art      1.00      0.95      0.98        62
        Economic      0.98      0.98      0.98        58
        Politics      0.94      0.97      0.95        65
         Science      0.96      0.98      0.97        56
           Sport      1.00      1.00      1.00        59

        accuracy                           0.98       300
       macro avg       0.98      0.98      0.98       300
    weighted avg       0.98      0.98      0.98       300
```
# Recurrent convolutional neural network 
