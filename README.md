# Data Analyse Projects on Kaggle Datasets
They are projects I did with my friends in university for a data science and machine learning course.
## Table of content
1. [Titannic Analysis](#titanic-analysis-who-will-survive): Use random forest to predict the live or dead for the passengers in the [Titanic dataset(train.csv)](https://www.kaggle.com/hesh97/titanicdataset-traincsv) on Kaggle using R and Python. And use [AUC/ROC curve](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/) to evaluate the result.
2. Deployed Application Programming Interface (API) and Web scrapping method to extract stock data and sales data from TuShare (Financial data platform) and eBay respectively for data visualization and presentation and used seaborn to visualize the data for stocks.
3. Use linear regression to analyze the correlation of wine quility and the acidity, sugar, so2 percentage for the [wine quality](https://www.kaggle.com/danielpanizzo/wine-quality) dataset on Kaggle.
4. Use Tensorflow to train a seq2seq model for text translation from English to Chinese and evaluate the model by BLUE score.The complete code is avaliable on [Colab](https://colab.research.google.com/drive/1ws4Dk6f-WULnCEbsQL-rwna9tNiUu6tH?usp=sharing)
5. Use sentiment analysis to analyze people's comments for tech gaints from tweets and glassdoor reviews and get insights from it.
## 1.Titanic Analysis: who will survive?
We want to classify the passengers into survive and dead ones according to some characteristics. Here we utilitied two classification ML algorithms, namely logistic regression and random forest. The general process of this project is as follows:
<p align="center">
     <img src="https://user-images.githubusercontent.com/44923423/143433848-d7494e7c-a480-47fc-ba25-9be2b608b386.png" 
     alt="proccess" 
     width="600">
</p>

The result of Random forest and Logistic Regression is show below. You can see logistic regression performs better since it has a higher AUC score.
### How to set the thresholds for logistic regression?
You can change the threshhold to get the maxium AUC score. For example:
<p align="center">
     <img src="https://user-images.githubusercontent.com/44923423/143433572-d2dbb50b-616c-47fa-a661-91ccd872de69.png" 
     alt="proccess" 
     width="500"> 
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/44923423/143434421-f9ecbb07-9044-41a3-86d2-526b5744130c.png" 
     alt="proccess" 
     width="500">
</p>

## 2.Stock data visualization: which stock is better?
<p align="center">
     <img src="https://user-images.githubusercontent.com/44923423/143514215-82c792d9-bde3-4f1d-976c-bdfbde92378e.png" 
     alt="proccess" 
     width="600"> 
</p>

## 3.Text Translation: seq2seq model
We get the dataset which contains more than 20,000 parallel sentences of English and Chinese from [manythings](http://www.manythings.org/anki/) and trained a bilingual model which can translate English into Chinese.

## 4.Use sentiment analysis to analyze people's comments for tech gaints.
We use the comments on glassdoor as well as tweets to get what people think about tech gaints
<p align="center">
     <img src="https://user-images.githubusercontent.com/44923423/143515120-ec236f41-8013-4f7f-bbe4-6b528648bc80.png" 
     alt="proccess" 
     width="300">
</p>
So, basically, there are three main steps. <br>
1. Use topic modeling to find out what they are talking about on tweets<br>
2. Use sentiment analysis to identify the positive or negative sentiment of comments <br>
3. Find out some clues for companies’ success<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/44923423/143515359-641eaa10-85d7-454a-983a-f74a8eeeea1a.png" 
     alt="proccess" 
     width="500">
</p>

### Some insights we got from these data
<p align="center">
     <img src="https://user-images.githubusercontent.com/44923423/143515462-c0e2e6c7-b786-418b-962d-f9093d7f3d30.png" 
     alt="proccess" 
     width="500">
     <img src="https://user-images.githubusercontent.com/44923423/143515528-e83420b4-5092-4be2-9fca-29c4afdeeb5f.png" 
     alt="proccess" 
     width="500">
     <img src="https://user-images.githubusercontent.com/44923423/143515604-83e34395-da1b-48b5-a2bb-8cf1dfd0ea77.png" 
     alt="proccess" 
     width="500">
     <img src="https://user-images.githubusercontent.com/44923423/143515634-d43cc5a3-eb40-4945-bd47-fd5038fd854f.png" 
     alt="proccess" 
     width="500">
</p>

The result for this analysis:
<p align="center">
      <img src="https://user-images.githubusercontent.com/44923423/143515415-56c8d95b-fd80-411a-bb3c-b43228dbe081.png" 
     alt="proccess" 
     width="500">
</p>

## Language:
R, Python
<p align="center">
      <img src="https://user-images.githubusercontent.com/44923423/143517750-1f2872ed-724e-4871-9c27-70dcd69b08ea.png" 
     alt="proccess" 
     width="300">
</p>

## Data source:
Kaggle or [Tushare](https://tushare.pro/) API

## Q&A for ML
### Cross-validation
#### What's cross-validation?
Cross-validation is a vital step in evaluating a model, which can split the data abitraryly and give a more accurate test result. It divide data into folds, and each time use one fold as the test set, fit the model to the other folds. Then predict on the test fold, and compute the matrix of intrest(in sklearn the default is R square)
#### Why use cross-validation?(motivation)
When we evaluate the performance of the model using .score(R square). The R square depends on how we split the data. The data points in the test set may have some peculiarities that mean the R squared computed on it is not representative of the model's ability to generalize to unseen data. To combat this dependence on what is essentially an arbitrary split, we use a technique called cross-validation. This method avoids the problem of your metric of choice being dependent on the train test split.
#### Pros and cons of cross validation?
The advantage of cross-validation is that it maximizes the amount of data that is used to train the model, as during the course of training, the model is not only trained, but also tested on all of the available data. However, a trade-off as using more folds is more computationally expensive. This is because you are fittings and predicting more times. This method avoids the problem of your metric of choice being dependent on the train test split.

#### How to use cross-validation? Can you give me an example?
We begin by splitting the dataset into five groups or folds. Then we hold out the first fold as a test set, fit our model on the remaining four folds, predict on the test set, and compute the metric of interest. Next, we hold out the second fold as our test set, fit on the remaining data, predict on the test set, and compute the metric of interest. Then similarly with the third, fourth, and fifth fold.<br>
![image](https://user-images.githubusercontent.com/44923423/144342213-fcd4d04c-03b0-4382-b3bb-b062213be85a.png)

```Python
# Import the necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
```
The output:
```
[0.81720569 0.82917058 0.90214134 0.80633989 0.94495637]
Average 5-Fold CV Score: 0.8599627722793232
```
### If I give you a Machine Learning project, what will you do steep by steep? And how do you allocate your time for each step? 
I will spend 40% of my time on the EDA. It seems a lot of time. But since EDA is fundamental for the future process, so it’s worthy to do that. Since the data may come from a very different industry, and I need to understand the background. Then try to see the distributions, see the patterns. After that can I know how should I pre-process the data, what kind of analysis should be done. 

Then, the second is the data pre-processing part. I think I will spend 20% of my time on it. I will first look at the null values in the dataset and try to figure out why the value is missing. If the missing row is mainly wrong data, I will just drop it. Else, if it doesn’t matter, I may impute missing values using K-NN. Besides, I need to check the data types for each column. If it’s not correct, need to change them into the correct type. I may also need to split the data, remove duplicates or change the text into lower case. Alongside visualize the data to see the correlations and distributions.

The third part is experiments, using ML models to get insights of the data. As long as we know how our data looks like and what we want to do, it will be easy to find a ML approach to handle that. So I will spend 10% of my time on it. For example, if our task is to prediction the result for new features, when our dataset mainly consists of numerical data, we can use linear regression to do prediction. If it’s categorical, we can use logistic regression for prediction. If our task is classification, we can use K-NN, decision tree, random forest, SVM, naïve bayes, or even neural network. If our task is clustering, we can use K-means, and anomaly detection. 

The fourth part is try to visualize the results and explain them in a reasonable way. And report the insights to the project teams. This part can be hard sometimes as we are not familiar with the background knowledge of the industry. So I will spend 20% of my time on it. I’ve done a project about analysing stock data with students majoring in finance. They helped me a lot about understanding the indexes and the charts.

The last part is packaging and deployment. Like using docker to containerize the data pipeline, designing intelligent model optimisation modules (MLflow), hosting models to scalable cloud infrastructure (AWS / GCP). It’s standard processes and I can ask seniors about that, so it’s relatively easy. I will spend 10% of my time on that.
### Popular classification methods:
- RF: Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. ... For regression tasks, the mean or average prediction of the individual trees is returned.

- LGB: Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks

- SVM: Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. The advantages of support vector machines are: Effective in high dimensional spaces. Still effective in cases where number of dimensions is greater than the number of samples.
### How to handle imbalanced data?
There are generally two ways to handle this. 

First is **resampling** the datasets(under-sampling and over-sampling)<br>
- **Under-sampling**<br>
Under-sampling balances the dataset by reducing the size of the abundant class. This method is used when the quantity of data is sufficient. By keeping all samples in the rare class and randomly selecting an equal number of samples in the abundant class, a balanced new dataset can be retrieved for further modelling.
- **Over-sampling**<br>
On the contrary, oversampling is used when the quantity of data is insufficient. It tries to balance the dataset by increasing the size of rare samples. Rather than getting rid of abundant samples, new rare samples are generated by using e.g. repetition, bootstrapping, or SMOTE (Synthetic Minority Over-Sampling Technique).
Note that there is no absolute advantage of one resampling method over another. The application of these two methods depends on the use case it applies to and the dataset itself. A combination of over-and under-sampling is often successful as well.<br>

Besides that, we can also adjust the ratio of the rare and abundant dataset. To conclude, those methods are all about adjusting the data. 

But actually, a better method is to use all of these data and adjust our model to suit the unbalanced data. We can **design a cost function that is penalizing the wrong classification of the rare class more** than wrong classifications of the abundant class, it is possible to design many models that naturally generalize in favour of the rare class. For example, tweaking an SVM to penalize wrong classifications of the rare class by the same ratio that this class is underrepresented.<br>
### When can regression be used?
Regression analysis is used when you want to predict a continuous dependent variable from a number of independent variables. If the dependent variable is dichotomous, then logistic regression should be used.
### What is regularized regression?
What fitting a linear regression does is minimize a loss function to choose a coefficient ai for each feature variable. If we allow these coefficients or parameters to be super large, we can get overfitting. It isn't so easy to see in two dimensions, but when you have loads and loads of features, that is, if your data sit in a high-dimensional space with large coefficients, it gets easy to predict nearly anything. For this reason, it is common practice to alter the loss function so that it penalizes for large coefficients. This is called regularization. The first type of regularized regression that we'll look at is called regularized regression
#### Ridge regression
The general regression uses OLS(Ordinary least squares) as the loss function, to penalise large coefficience to ovid overfitting, we can add this to OLS.<br>
<img src="https://user-images.githubusercontent.com/44923423/144366665-cbfafa2c-56cc-4391-b9b6-96645fb2971b.png" 
     alt="ridge regression" 
     width="100">
##### How to choose alpha?(hyperparameter tuning)<br>
- A large alpha: rely heavily on large coefficience, and the model will be too simple(underfitting)
- alpha = 0: overfitting
#### Lasso regression
### What’s R square?
R square is the amount of variance between the target variable that is predicted from the feature variables.
### What is a small p value means?
A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis. A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.
### What's the difference between normalize and standarize?
<img src="https://user-images.githubusercontent.com/44923423/146113770-d1eb1e64-b97c-46f5-803f-3f4fda67abae.png" 
     alt="ridge regression" 
     width="100">
Normalization vs. standardization is an eternal question among machine learning newcomers. Let me elaborate on the answer in this section.

Normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.
Standardization, on the other hand, can be helpful in cases where the data follows a Gaussian distribution. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization.
However, at the end of the day, the choice of using normalization or standardization will depend on your problem and the machine learning algorithm you are using. There is no hard and fast rule to tell you when to normalize or standardize your data. You can always start by fitting your model to raw, normalized and standardized data and compare the performance for best results.
