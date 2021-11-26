# Data Analyse Projects on Kaggle Datasets
They are projects I did with my friends in university for a data science and machine learning course.
## Table of content
1. [Titannic Analysis](#titanic-analysis-who-will-survive): Use random forest to predict the live or dead for the passengers in the [Titanic dataset(train.csv)](https://www.kaggle.com/hesh97/titanicdataset-traincsv) on Kaggle using R and Python. And use [AUC/ROC curve](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/) to evaluate the result.
2. Deployed Application Programming Interface (API) and Web scrapping method to extract stock data and sales data from TuShare (Financial data platform) and eBay respectively for data visualization and presentation and used seaborn to visualize the data for stocks.
3. Use linear regression to analyze the correlation of wine quility and the acidity, sugar, so2 percentage for the [wine quality](https://www.kaggle.com/danielpanizzo/wine-quality) dataset on Kaggle.
4. Use Tensorflow to train a seq2seq model for text translation from English to Chinese and evaluate the model by BLUE score.The complete code is avaliable on [Colab](https://colab.research.google.com/drive/1ws4Dk6f-WULnCEbsQL-rwna9tNiUu6tH?usp=sharing)
5. Use sentiment analysis to analyze people's comments for tech gaints.
## 1.Titanic Analysis: who will survive?
We want to classify the passengers into survive and dead ones according to some characteristics. Here we utilitied two classification ML algorithms, namely logistic regression and random forest. The general process of this project is as follows:
![image](https://user-images.githubusercontent.com/44923423/143433848-d7494e7c-a480-47fc-ba25-9be2b608b386.png)

The result of Random forest and Logistic Regression is show below. You can see logistic regression performs better since it has a higher AUC score.
### How to set the thresholds for logistic regression?
You can change the threshhold to get the maxium AUC score. For example:
![image](https://user-images.githubusercontent.com/44923423/143433572-d2dbb50b-616c-47fa-a661-91ccd872de69.png)
![image](https://user-images.githubusercontent.com/44923423/143434421-f9ecbb07-9044-41a3-86d2-526b5744130c.png)
## 2.Stock data visualization: which stock is better?
![image](https://user-images.githubusercontent.com/44923423/143514215-82c792d9-bde3-4f1d-976c-bdfbde92378e.png)
## 3.Text Translation: seq2seq model
We get the dataset which contains more than 20,000 parallel sentences of English and Chinese from [manythings](http://www.manythings.org/anki/) and trained a bilingual model which can translate English into Chinese.

## 4.Use sentiment analysis to analyze people's comments for tech gaints.
We use the comments on glassdoor as well as tweets to get what people think about tech gaints
![image](https://user-images.githubusercontent.com/44923423/143515120-ec236f41-8013-4f7f-bbe4-6b528648bc80.png)

So, basically, there are three main steps. <br>
1. Use topic modeling to find out what they are talking about on tweets<br>
2. Use sentiment analysis to identify the positive or negative sentiment of comments <br>
3. Find out some clues for companies’ success<br>
![image](https://user-images.githubusercontent.com/44923423/143515634-d43cc5a3-eb40-4945-bd47-fd5038fd854f.png)

### Some insights we got from these data
![image](https://user-images.githubusercontent.com/44923423/143515462-c0e2e6c7-b786-418b-962d-f9093d7f3d30.png)
![image](https://user-images.githubusercontent.com/44923423/143515528-e83420b4-5092-4be2-9fca-29c4afdeeb5f.png)
![image](https://user-images.githubusercontent.com/44923423/143515604-83e34395-da1b-48b5-a2bb-8cf1dfd0ea77.png)

![image](https://user-images.githubusercontent.com/44923423/143515359-641eaa10-85d7-454a-983a-f74a8eeeea1a.png)

The result for this analysis:
![image](https://user-images.githubusercontent.com/44923423/143515415-56c8d95b-fd80-411a-bb3c-b43228dbe081.png)

## Language:
R, Python
## Data source:
Kaggle or Tushare API
