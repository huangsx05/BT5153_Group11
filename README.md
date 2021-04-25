# BT5153 Group 11 Final Project


Below information describes codes used in the 3 stage of the BT5153 project.

## Data Scrape: selenium_shopee_scrape.ipynb

This file is used to scrape data related to mask from shopee webset https://shopee.sg.


## Stage 1: `5153-Stage1-Sentiment-Analysis-Final.ipynb`

This file is to separate negative and non-negative reviews.

The nltk pakage SentimentIntensityAnalyzer and a classification model are used to extract all negative comemnts in 1-4 stars reviews.

A wordcloud is generated on all negative reviews to observe the negative aspects, and we observed 3 main categories(delivery, product, service).


## Stage 2:

* 1) `5153-Stage2-BERT-FullSentence-FeatureExtraction-Final.ipynb`  


This file extracts Bert and DistillBert 1D features [cls, avg, max], and 2D features from the last hidden layer of Bert pretrained model. 
The input data are full sentence of reviews which contain negative comments.

* 2) `5153-Stage2-BERT-NegativePart-FeatureExtraction-Final.ipynb`  


This file extract Bert and DistillBert 1D features [cls,avg,max], and 2D features from last hidden layer of bert pretrained model.
The input data are pure negative reviews which exclude the non-negative comments.

* 3) `5153-Stage2-Similarity-Analysis-Final.ipynb`   

This file uses the unsupervised ML method to clustering all the negative comments into 3 categories observed in stage 1,
K-means clustering methods as well as an innovative proposed method of cosine similarity are used to label the predicted category for each review.

* 4) `5153-Stage2-wordcloud_aspect.ipynb`  

The wordclouds with regards to 3 categories are generated to show the keywords of the different aspects.

## Stage 3: 

In this stage, we use different ML models to classify the reviews to three categories.

* 1) Bidirectional LSTM :   `5153-Stage3-model-bert-feature-lstm-final.ipynb`

This model uses Keras sequential model to do the multi-class, multi-label classification.
The output unit is set to 3 with sigmoid activation, the loss function is BinaryCrossentropy, and Adam optimizer is used.
DistillBert 1D features[CLS, avg], 2D features for both full sentences and pure negative comments are run in the file. 1D features with max values are excluded as results are not ideal.

Bert features are excluded in this file as the results do not perform as well as the DistillBert Feature

Comparing the results in the file, we found DistillBert 1D average feature and 2D features for pure negative comments show better results, and following models will use these features as input.

* 2) `Base Model-knn: 5153-stage3-knn-final.ipynb`

Three binary classifiction model are built for three categories (delivery, product, service), and grid search is set to find the best K value for each model.
This model uses the DistillBert 1D average features for the pure negative comments as features.

* 3) `Base Model- NB: 5153-stage3-NB-final.ipynb`

Three GaussianNB model is used for the binary classification for each category.
This model uses the DistillBert 1D average features for the pure negative comments as features.

* 4) CNN & RNN Model :  `5153-Stage3-cnn-rnn-with-bert-2dvect.ipynb`  
                      `5153-Stage3-cnn-rnn-with-negative-bert-2dvect.ipynb`

The models use DistillBert 2D features as input. Both models are multi-class multi-label models. The output layer and loss function are same as LSTM model.

CNN Model : Three different size filters are used in the Conv1D layer, followed by max pooling and concatenation, then connencted to output layer.
RNN Model: A bidirectional RNN model is used in the analysis. 

* 5) Bert Fine Tune model: `5153-Stage3-model-bert-feature-lstm-final.ipynb`

reference source: https://github.com/charles9n/bert-sklearn

With reference to the above url, we use the scikit-learn wrapper to fine tune the Bert Model for the binary classification model for delivery, product and service, respectively.

## Extended Study: `5153_Extended Studies.ipynb`

This notebook contains the codes of the Extended Studies section, which focuses on analyzing the relationship between covid-relevant data and customers’ sentiments, as well as how aspect-based sentiment scores change over time.
	
