# Data Analyse Projects on Kaggle Datasets
They are projects I did with my friends in university for a data science and machine learning course.
## Table of content
1. Use random forest to predict the live or dead for the passengers in the [Titanic dataset(train.csv)](https://www.kaggle.com/hesh97/titanicdataset-traincsv) on Kaggle using R and Python. And use [AUC/ROC curve](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/) to evaluate the result.
2. Deployed Application Programming Interface (API) and Web scrapping method to extract stock data and sales data from TuShare (Financial data platform) and eBay respectively for data visualization and presentation and used seaborn to visualize the data for stocks.
3. Use linear regression to analyze the correlation of wine quility and the acidity, sugar, so2 percentage for the [wine quality](https://www.kaggle.com/danielpanizzo/wine-quality) dataset on Kaggle.
4. Use Tensorflow to train a LSTM model for text translation from English to Chinese and evaluate the model by BLUE score.The complete code is avaliable on [Colab](https://colab.research.google.com/drive/1ws4Dk6f-WULnCEbsQL-rwna9tNiUu6tH?usp=sharing)
5. Use Topic Modeling and sentiment analysis to compare different factors to influence the incoming in China and America.
## Titanic Analysis: who will survive?
We want to classify the passengers into survive and dead ones according to some characteristics. Here we utilitied two classification ML algorithms, namely logistic regression and random forest. The general process of this project is as follows:
![image](https://user-images.githubusercontent.com/44923423/143433848-d7494e7c-a480-47fc-ba25-9be2b608b386.png)

The result of Random forest and Logistic Regression is show below. You can see logistic regression performs better since it has a higher AUC score.
### How to set the thresholds for logistic regression?
You can change the threshhold to get the maxium AUC score. For example:
![image](https://user-images.githubusercontent.com/44923423/143433572-d2dbb50b-616c-47fa-a661-91ccd872de69.png)
![image](https://user-images.githubusercontent.com/44923423/143433615-85ec6b9e-97a3-4c64-b51e-9bb4aa021354.png)
## Language:
Julia, R, Python
## Data source:
Kaggle or Tushare API
