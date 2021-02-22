#import package
library('sampling')
library('dplyr')
library("ggplot2")
library("randomForest")
library("pROC")
library("pryr")

#preprocessing data set
##load data
raw_full_data <- read.csv(file="./train.csv")
##get useful columns
full_data <- subset(raw_full_data,select=c("PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Survived"))
##create new column to represent whether a person is accompanied or not
full_data$Accompany <- full_data$SibSp+full_data$Parch
full_data <- full_data[,-c(5,6)]
full_data$Accompany <- case_when(full_data$Accompany>=1 ~ 1,full_data$Accompany==0 ~ 0)
##fill the missing value in age column
mean_age <- mean(full_data$Age,na.rm=TRUE)
full_data$Age[is.na(full_data$Age)] <- mean_age
##encode sex by male=1 female=0
full_data$Sex <- case_when(full_data$Sex=="male" ~ 1,full_data$Sex=="female" ~ 0)

#set train test data
##set random seed
set.seed(1)
##split data
train_rows <- sample(rownames(full_data),dim(full_data)[1]*2/3)
train_data <- full_data[train_rows,] 
test_data <- anti_join(full_data,train_data,by='PassengerId')
train_data <- train_data[,-1]
test_data <- test_data[,-1]
##fare column standardization
max_train_fare <- max(train_data$Fare)
min_train_fare <- min(train_data$Fare)
train_data$Fare <- (train_data$Fare-min_train_fare)/(max_train_fare-min_train_fare)
max_test_fare <- max(test_data$Fare)
min_test_fare <- min(test_data$Fare)
test_data$Fare <- (test_data$Fare-min_test_fare)/(max_test_fare-min_test_fare)
##output csv files
###write.csv(train_data,"./train_data.csv")
###write.csv(test_data,"./test_data.csv")

#logistic regression part
##train model
t1=Sys.time()
mem1 = mem_change(lr_model <- glm(Survived~Pclass+Sex+Age+Fare+Accompany,data=train_data,family=binomial()))
t2=Sys.time()
summary(lr_model)

##run prediction
pred_logistic <- as.numeric(predict(lr_model,newdata=test_data,type="response")>0.5)
##auc/roc assessment
df_logistic <- data.frame(
	prob = pred_logistic,
	obs = test_data$Survived
)
table(test_data$Survived,pred_logistic,dnn=c("Obs","Pred"))#2.提取Survived和pre_logistic中的频率
logistic_roc <- roc(test_data$Survived,pred_logistic)
plot(logistic_roc, print.auc=TRUE, auc.polygon=TRUE,main='Logistic ROC Curve')

#random forest part
##train model
t3=Sys.time()
rf_model <- randomForest(Survived~Pclass+Sex+Age+Fare+Accompany,data=train_data)
t4=Sys.time()
summary(rf_model)
##run prediction
pred_rf <- as.numeric(predict(rf_model,newdata=test_data,type="response")>0.5)
##auc/roc assessment
df_rf <- data.frame(prob=pred_rf,obs=test_data$Survived)
table(test_data$Survived,pred_rf,dnn=c("Obs","Pred"))
rf_roc <- roc(test_data$Survived,pred_rf)
plot(rf_roc,print.auc=TRUE,auc.polygon=TRUE,main="RF ROC Curve")
print('R\'s performance')
print('logistic regression: time=')
print(t2-t1)
print(mem1)
print('random forest: time=')
print(t4-t3)
print(mem2)