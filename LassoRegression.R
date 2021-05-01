# install.packages("Epi")
# install.packages("caret")
# install.packages("glmnet")
# install.packages("dplyr")
library(dplyr)
require(glmnet)
seedNum <- 427
hf <- read.csv("./data/KorAHF_Adm.csv")
testProportion <- 0.3
outcome <- "death1yr"
crossValNum <- 10
crossValMetric = "ROC"
#For parallelization to speed up!(only in Unix. You can use in Mac, too )
library(doMC)
library(parallel)
getDoParWorkers()
registerDoMC(cores = parallel::detectCores())
getDoParWorkers()

##########################################################################################################
####Helper functions####
##Normalization of the elements in the data
minmaxNormalize <- function(x) {return ((x-min(x,na.rm = T)) / (max(x,na.rm = T)-min(x,na.rm = T)))}
zNormalize <- function(x) {return ( (x-mean(x,na.rm = T))/sd(x,na.rm = T))}
##########################################################################################################

##
names(hf) <- tolower(names(hf))
str(hf)
#date into date
hf$final_expire_date <- as.Date(hf$final_expire_date, '%Y/%m/%d')
hf$dx_date <- as.Date(hf$dx_date, '%Y/%m/%d')
hf$death1yr <- ifelse(hf$final_expire_date - hf$dx_date <= 365,TRUE, FALSE)
hf$death1yr <- ifelse(is.na(hf$death1yr),FALSE,hf$death1yr) #outcome of interest is all-cause mortality within one year
outcome <- ifelse(hf$death1yr, 'positive', 'negative')
outcome <- factor(outcome, levels =c('negative','positive'))
table(outcome) #TRUE 1244 FALSE 4381

#remove character and date from feature
for(i in seq(length(hf))){
    if(i==1) charVar <- c()
    if(class(hf[,i])%in%c("character","Date")) charVar <- c(charVar,names(hf)[i])
}

#remove NA values from feature
for(i in seq(length(hf))){
    if(i==1) naVar <- c()
    if(sum(is.na(hf[,i]))>= 100)  naVar <- c(naVar,names(hf)[i])
}


featureVar <- names(hf)[21:length(hf)]
featureVar <- featureVar[!(featureVar %in% charVar)]
featureVar <- featureVar[!(featureVar %in% naVar)]
featureVar <- featureVar[!(featureVar %in% c("death1yr"))]

feature <- hf %>% select(featureVar)
completeInd <- complete.cases(feature)
table(completeInd)

feature <- feature[completeInd,]
outcome <- outcome[completeInd]
#min max normalization
feature <- as.data.frame(lapply(feature, minmaxNormalize))

##split train and test
set.seed(seedNum)
#testInd<-sample.int(n= nrow(wdbc_p), size = floor(testProportion* nrow(wdbc_p)), replace = F)
testInd<-caret::createDataPartition(outcome, p = testProportion, list = F) #To split total data set while preserving the proportion of outcomes in train and test set

featureTrain <- feature[-testInd,]
featureTest  <- feature[testInd,]

outcomeTrain <- outcome[-testInd]
outcomeTest <- outcome[testInd]
####Lasso regression####
fitControl <- caret::trainControl(method = "repeatedcv", number = crossValNum, repeats = 5,
                                  classProbs=TRUE, summaryFunction = caret::twoClassSummary)
set.seed(seedNum)

trainFit <- caret::train(x=featureTrain[,1:100], y=outcomeTrain, 
                         method = "glmnet", 
                         trControl = fitControl, 
                         family = "binomial",
                         #preProcess = c("center","scale"), 
                         metric = crossValMetric,
                         #tuneGrid = expand.grid(.alpha = 1, .lambda = 0.002),
                         tuneLength = 3) #you can increase tune length to try more Ks
#Output of fit
trainFit

#check the importance of the variables
lassoImp <- caret::varImp(trainFit); lassoImp
plot(lassoImp)

####Performance of model####
#applying the trained model to the test set
predictClass <- predict(trainFit,newdata = featureTest)
predictProb <- predict(trainFit,newdata = featureTest,type ="prob")
#Get the confusion matrix to see accuracy value and other parameter values
caret::confusionMatrix(predictClass, outcomeTest,positive = "positive" )
#the accuracy
mean(predictClass == outcomeTest) 

#plot ROC curve
Epi::ROC(form=outcomeTest~predictProb[,2], plot="ROC")
out= Epi::ROC(form=outcomeTest~predictProb[,2], plot="ROC")
out$AUC