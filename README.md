# Arabic Text Classification based on Machine learning Deep Learning Algorithms
In arabic text classification, texts have to be transformed into numeric representations suitable for the learning algorithms. in this studies, we used TF-IDF with Machine learning algorithms and Embedding words with Deep learnings. In the present study, text classification is comparative study based on a Machine learning methods and Deep learning Methods.  The current work has proved the ability to classify collections of Arabic or English text documents successfully. It showed approximately 98% sav- ings in vector space and 2% performance improvement compared to the best recorded results on Arabic dataset Aljazeera News.


# Dataset
Aljazeera news 5 classes dataset (Alj-News5). Alj-News5 is another different dataset that contains 1500 documents for news articles. Each document is labeled with one of the following five classes {‘Art’, ‘Economic’, ‘Politics’, ‘Science’, and ‘Sport’}. Alj-News5 was used by other researchers . A comparison of classification results of the current approach against other research works on Alj-News5 will be pre- sented in the experiments.

# Step 1: Text Classification based on Machine Learning 
# Naive Bayes Classifier
# Rocchio classification
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

# Boosting and Bagging
# Boosting is a Ensemble learning meta-algorithm
# K-nearest Neighbor
# Support Vector Machine (SVM)
# Decision Tree
# Conditional Random Field (CRF)


# Step 2: Text Classification based on Deap Learning 
# convolutional neural network 
# Recurrent neural networks
# Deep neural networks
# Recurrent convolutional neural network 
