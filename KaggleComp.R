eBayTrain = read_csv("eBayiPadTrain.csv")
eBayTest = read_csv("eBayiPadTest.csv")
library(tm)
CorpusDescription = Corpus(VectorSource(c(eBayTrain$description, eBayTest$description)))
CorpusDescription = tm_map(CorpusDescription, content_transformer(tolower), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, PlainTextDocument, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removePunctuation, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removeWords, stopwords("english"), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, stemDocument, lazy=TRUE)
dtm = DocumentTermMatrix(CorpusDescription)
findFreqTerms(dtm, lowfreq = 100)
sparse = removeSparseTerms(dtm, 0.97)
DescriptionWords = as.data.frame(as.matrix(sparse))
colnames(DescriptionWords) = make.names(colnames(DescriptionWords))
DescriptionWordsTrain = head(DescriptionWords, nrow(eBayTrain))
DescriptionWordsTest = tail(DescriptionWords, nrow(eBayTest))
eBayTrain %<>% mutate(condition = as.factor(condition), cellular = as.factor(cellular),
        carrier = as.factor(carrier), color = as.factor(color),
        storage = as.factor(storage), productline = as.factor(productline),
        is_descr = as.factor(description == ""))
eBayTrain %<>%select(-description, -UniqueID)
eBayTrain <- cbind(eBayTrain, DescriptionWordsTrain)
str(eBayTrain)
set.seed(1000)
names(eBayTrain)
names(eBayTrain)[3] <- "condition_is"
split <- sample.split(eBayTrain$sold, SplitRatio = 0.7)
train  <- filter(eBayTrain, split == T)
test <- filter(eBayTrain, split == F)
model_glm1 <- glm(sold ~ ., data = train, family = binomial)
summary(model_glm1)
library(ROCR)
predict_glm <- predict(model_glm1, newdata = test, type = "response" )
ROCRpred = prediction(predict_glm, test$sold)
as.numeric(performance(ROCRpred, "auc")@y.values)
library(rpart)
library(rpart.plot)
model_cart1 <- rpart(sold ~ ., data = train, method = "class")
prp(model_cart1)
predict_cart <- predict(model_cart1, newdata = test, type = "prob")[,2]
ROCRpred = prediction(predict_cart, test$sold)
as.numeric(performance(ROCRpred, "auc")@y.values)
library(caret)
library(e1071)
tr.control = trainControl(method = "cv", number = 10)
cpGrid = expand.grid( .cp = seq(0.01,0.5,0.01))
train(sold ~ ., data = train, method = "rpart", trControl = tr.control, tuneGrid = cpGrid )
model_cart2 <- rpart(sold ~ ., data = train, method = "class", cp = 0.06)
predict_cart <- predict(model_cart2, newdata = test, type = "prob")[,2]
ROCRpred = prediction(predict_cart, test$sold)
as.numeric(performance(ROCRpred, "auc")@y.values)
library(randomForest)
train$sold <- as.factor(train$sold)
set.seed(1000)
model_rf <- randomForest(sold ~ ., data = train)
test$sold <- as.factor(test$sold)
predict_rf  <- predict(model_rf, newdata = test, type = "prob")[,2]
ROCRpred = prediction(predict_rf, test$sold)
as.numeric(performance(ROCRpred, "auc")@y.values)

eBayTest = read_csv("eBayiPadTest.csv")
eBayTest %<>% mutate(condition = as.factor(condition), cellular = as.factor(cellular),
        carrier = as.factor(carrier), color = as.factor(color),
        storage = as.factor(storage), productline = as.factor(productline),
        is_descr = as.factor(description == ""))
eBayTest %<>%select(-description)
eBayTest <- cbind(eBayTest, DescriptionWordsTest)
names(eBayTest)[3] <- "condition_is"
eBayTrain$sold <- as.factor(eBayTrain$sold)

eBayTrain = read_csv("eBayiPadTrain.csv")
eBayTest = read_csv("eBayiPadTest.csv")
eBayTrain %<>% select(-sold)
ebay <- rbind(eBayTrain, eBayTest)
ebeBayTest  <-  slice(ebay, 1862:nrow(ebay))
eBayTest %<>%select(-description)
eBayTest <- cbind(eBayTest, DescriptionWordsTest)
names(eBayTest)[3] <- "condition_is"
eBayTrain = read_csv("eBayiPadTrain.csv")
eBayTrain %<>% mutate(condition = as.factor(condition), cellular = as.factor(cellular),
carrier = as.factor(carrier), color = as.factor(color),
storage = as.factor(storage), productline = as.factor(productline),
is_descr = as.factor(description == ""))
eBayTrain %<>%select(-description, -UniqueID)
eBayTrain <- cbind(eBayTrain, DescriptionWordsTrain)
eBayTrain$sold <- as.factor(eBayTrain$sold)
names(eBayTrain)[3] <- "condition_is"
ay %<>% mutate(condition = as.factor(condition), cellular = as.factor(cellular),
carrier = as.factor(carrier), color = as.factor(color),
storage = as.factor(storage), productline = as.factor(productline),
is_descr = as.factor(description == ""))

set.seed(1000)
model_rf <- randomForest(sold ~ ., data = train, importance = T)
varImpPlot(model_rf)
model_rf <- randomForest(sold ~ .-veri - new, data = train, importance = T)
predict_rf  <- predict(model_rf, newdata = test, type = "prob")[,2]


