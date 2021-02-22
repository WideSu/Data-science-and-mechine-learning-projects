library('corrplot')
getwd()
setwd("D:/Study/¿ÆÑÐ/Homework/3")
# load data
wine_data <- read.csv('./winequality.csv')

# data preprocessing
wine_data$total_acidity <- wine_data$fixed.acidity+wine_data$volatile.acidity+wine_data$citric.acid
low_quality_wine <- wine_data[which(wine_data$quality <= '4'),]
medium_quality_wine <- rbind(wine_data[which(wine_data$quality == '5'),],wine_data[which(wine_data$quality == '6'),])
high_quality_wine <- rbind(wine_data[which(wine_data$quality == '7'),],wine_data[which(wine_data$quality == '8'),])

# simply check quality distribution
quality <- wine_data$quality
hist(quality)

# for each quality, compare their mean in acidity, sugar, so2, alcohol, pH
## acidity
low_acidity_mean <- mean(low_quality_wine$total_acidity)
medium_acidity_mean <- mean(medium_quality_wine$total_acidity)
high_acidity_mean <- mean(high_quality_wine$total_acidity)
acidity_mean_list <- c(low_acidity_mean, medium_acidity_mean, high_acidity_mean)
acidity_name_list <- c(3.5, 5.5, 7.5)
acidity_df <- data.frame('name' = acidity_name_list, 'mean' = acidity_mean_list)
plot(acidity_df$name, acidity_df$mean, type='p')
## sugar
low_sugar_mean <- mean(low_quality_wine$residual.sugar)
medium_sugar_mean <- mean(medium_quality_wine$residual.sugar)
high_sugar_mean <- mean(high_quality_wine$residual.sugar)
sugar_mean_list <- c(low_sugar_mean, medium_sugar_mean, high_sugar_mean)
sugar_name_list <- c(3.5, 5.5, 7.5)
sugar_df <- data.frame('name' = sugar_name_list, 'mean' = sugar_mean_list)
plot(sugar_df$name, sugar_df$mean, type='p')
## so2
low_so2_mean <- mean(low_quality_wine$total.sulfur.dioxide)
medium_so2_mean <- mean(medium_quality_wine$total.sulfur.dioxide)
high_so2_mean <- mean(high_quality_wine$total.sulfur.dioxide)
so2_mean_list <- c(low_so2_mean, medium_so2_mean, high_so2_mean)
so2_name_list <- c(3.5, 5.5, 7.5)
so2_df <- data.frame('name' = so2_name_list, 'mean' = so2_mean_list)
plot(so2_df$name, so2_df$mean, type='p')
## alcohol
low_alcohol_mean <- mean(low_quality_wine$alcohol)
medium_alcohol_mean <- mean(medium_quality_wine$alcohol)
high_alcohol_mean <- mean(high_quality_wine$alcohol)
alcohol_mean_list <- c(low_alcohol_mean, medium_alcohol_mean, high_alcohol_mean)
alcohol_name_list <- c(3.5, 5.5, 7.5)
alcohol_df <- data.frame('name' = alcohol_name_list, 'mean' = alcohol_mean_list)
plot(alcohol_df$name, alcohol_df$mean, type='p')
## pH
low_ph_mean <- mean(low_quality_wine$pH)
medium_ph_mean <- mean(medium_quality_wine$pH)
high_ph_mean <- mean(high_quality_wine$pH)
ph_mean_list <- c(low_ph_mean, medium_ph_mean, high_ph_mean)
ph_name_list <- c(3.5, 5.5, 7.5)
ph_df <- data.frame('name' = ph_name_list, 'mean' = ph_mean_list)
plot(ph_df$name, ph_df$mean, type='p')

# for each quality, make a pie chart to present acidity percenttage
## low quality
low_fa_mean <- mean(low_quality_wine$fixed.acidity)
low_va_mean <- mean(low_quality_wine$volatile.acidity)
low_ca_mean <- mean(low_quality_wine$citric.acid)
low_a_mean_list <- c(low_fa_mean, low_va_mean, low_ca_mean)
low_a_name_list <- c('fixed','volatile','citric')
low_a_df <- data.frame('mean'=low_a_mean_list)
rownames(low_a_df) <- low_a_name_list
pie(low_a_df$mean, col=gray(seq(.4,1,.3)))
## medium quality
mid_fa_mean <- mean(medium_quality_wine$fixed.acidity)
mid_va_mean <- mean(medium_quality_wine$volatile.acidity)
mid_ca_mean <- mean(medium_quality_wine$citric.acid)
mid_a_mean_list <- c(mid_fa_mean, mid_va_mean, mid_ca_mean)
mid_a_name_list <- c('fixed','volatile','citric')
mid_a_df <- data.frame('mean'=mid_a_mean_list)
rownames(mid_a_df) <- mid_a_name_list
pie(mid_a_df$mean, col=gray(seq(.4,1,.3)))
## high quality
high_fa_mean <- mean(high_quality_wine$fixed.acidity)
high_va_mean <- mean(high_quality_wine$volatile.acidity)
high_ca_mean <- mean(high_quality_wine$citric.acid)
high_a_mean_list <- c(high_fa_mean, high_va_mean, high_ca_mean)
high_a_name_list <- c('fixed','volatile','citric')
high_a_df <- data.frame('mean'=high_a_mean_list)
rownames(high_a_df) <- high_a_name_list
pie(high_a_df$mean, col=gray(seq(.4,1,.3)))

# pie chart to represent acidity, sugar, so2 percentage
## low quality
low_all_list <- c(low_acidity_mean, low_sugar_mean, low_so2_mean)
low_all_name <- c('acidity', 'sugar', 'so2')
low_all_df <- data.frame('mean'=low_all_list)
rownames(low_all_df) <- low_all_name
pie(low_all_df$mean, col=gray(seq(.4,1,.3)))
## medium quality
mid_all_list <- c(medium_acidity_mean, medium_sugar_mean, medium_so2_mean)
mid_all_name <- c('acidity', 'sugar', 'so2')
mid_all_df <- data.frame('mean'=mid_all_list)
rownames(mid_all_df) <- mid_all_name
pie(mid_all_df$mean, col=gray(seq(.4,1,.3)))
## high quality
high_all_list <- c(high_acidity_mean, high_sugar_mean, high_so2_mean)
high_all_name <- c('acidity', 'sugar', 'so2')
high_all_df <- data.frame('mean'=high_all_list)
rownames(high_all_df) <- high_all_name
pie(high_all_df$mean, col=gray(seq(.4,1,.3)))

# three acidity, sugar, and alcohol's heat map with sulphates
corr_table <- data.frame(wine_data$fixed.acidity, wine_data$volatile.acidity, wine_data$citric.acid, wine_data$residual.sugar, wine_data$alcohol, wine_data$sulphates)
colnames(corr_data) <- c('fixed_acidity', 'volatile_acidity', 'citric_acidity', 'sugar', 'alcohol', 'sulphates')
corr_matrix <- cor(corr_table)
corrplot(corr=corr_matrix, method='color')

# sulphates and quality xyplot
sul_quali_df <- data.frame('sulphates'=wine_data$sulphates,'quality'=wine_data$quality)
plot(sul_quali_df$quality, sul_quali_df$sulphates)

# linear regression
reg <- lm(wine_data$quality~wine_data$total_acidity+wine_data$chlorides+wine_data$total.sulfur.dioxide+wine_data$sulphates+wine_data$alcohol)
print(summary(reg))

