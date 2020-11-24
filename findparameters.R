#! /usr/bin/Rscript
options(width=150)
suppressPackageStartupMessages(library(stopwords))
suppressPackageStartupMessages(library(text2vec))
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library("pROC"))

split_number=1

setwd(paste("~/data/CS598/MoviewReview/split_", split_number, sep=""))
getwd()


myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,header = TRUE)


train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,preprocessor = tolower, tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)

test <- read.table("test.tsv", stringsAsFactors = FALSE,header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,preprocessor = tolower, tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)

test.y <- read.table("test_y.tsv", header = TRUE)



hyper_grid <- expand.grid(
  #eta = seq(0.3,0.5,0.05),
  eta = 0.3,
  #max_depth = seq(2,6,1), 
  max_depth=3,
  nrounds= 700,
  #nrounds= seq(400,1500,100),
  #min_child_weight = 3,
  subsample = seq(0.1,1,0.1),
  colsample_bytree = seq(0.1,1,0.1),
  gamma = c(0, 1, 10, 100),
  lambda = c(0, 0.1, 1, 100),
  alpha = c(0, 0.1, 1, 100),
  #gamma=0,
  #lambda=1,
  #alpha=1,
  auc = 0,          # a place to dump RMSE results
  time= 0          # a place to dump required number of trees
)

maxi=dim(hyper_grid)[1]
mi=0
mauc=0

for(i in seq_len(nrow(hyper_grid))) {
  set.seed(2685)
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i],
    gamma = hyper_grid$gamma[i],
    lambda =hyper_grid$lambda[i],
    alpha = hyper_grid$alpha[i]
  )
  start_time <- Sys.time()
  m <- xgboost(data = dtm_train, 
               label = train$sentiment,
               params = params,
               nrounds=hyper_grid$nrounds[i],
               verbose = 0,
               nthread=8,
               objective="binary:logistic",
               early_stopping_rounds=50,
               eval_metric='auc')
  
  mypred=predict(m,dtm_test)
  end_time <- Sys.time()
  
  output = data.frame(id = test$id, prob = as.vector(mypred))
  pred <- merge(output, test.y, by="id")
  roc_obj <- roc(pred$sentiment, pred$prob,quiet=TRUE)
  hyper_grid$auc[i]=pROC::auc(roc_obj,quiet=TRUE)
  hyper_grid$time[i]=end_time - start_time
  
  if (hyper_grid$auc[i]>mauc){
    mauc=hyper_grid$auc[i]
    mi=i
  }
  print(paste(i,"of",maxi, "Max AUC=",mauc, "Element=",mi, "time=",end_time - start_time))
  print(hyper_grid[i,])
}  
write.table(hyper_grid, file = "tableres.txt", row.names = TRUE, col.names=TRUE,sep = ',')

max(hyper_grid$auc)
which.max(hyper_grid$auc)
hyper_grid[which.max(hyper_grid$auc),]
