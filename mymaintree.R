suppressPackageStartupMessages(library(stopwords))
suppressPackageStartupMessages(library(text2vec))
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(xgboost))


#split_number=1

#setwd(paste("~/data/CS598/MoviewReview/split_", split_number, sep=""))
#getwd()

#####################################
# Load your vocabulary and training data
#####################################
myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,header = TRUE)


train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,preprocessor = tolower, tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)

#####################################
#
# Train a binary classification model
#
#####################################

params <- list(
  eta = 0.3,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.2,
  gamma=1,
  lambda=0.1,
  alpha=0.1
)


set.seed(2685)
xgb.model <- xgboost(data = dtm_train,
                     label = train$sentiment,
                     params = params,
                     nrounds=1000,
                     nthread=4,
                     objective="binary:logistic",
                     early_stopping_rounds=100,
                     verbose = FALSE,
                     eval_metric='auc')


test <- read.table("test.tsv", stringsAsFactors = FALSE,header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)

#####################################
# Compute prediction 
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predited probabilities
#####################################
it_test = itoken(test$review,preprocessor = tolower, tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)
mypred=predict(xgb.model,dtm_test)
output = data.frame(id = test$id, prob = as.vector(mypred))
write.table(output, file = "mysubmission.txt", row.names = FALSE, sep='\t')
