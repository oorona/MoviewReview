#! /usr/bin/Rscript
library(stopwords)
library(text2vec)
library(glmnet)
library(pROC)

split_number=2

setwd("~/data/CS598/MoviewReview")

#train = read.table(paste("./split_",split_number,"/train.tsv",sep = ''), stringsAsFactors = FALSE, header = TRUE)

train = read.table("alldata.tsv", stringsAsFactors = FALSE, header = TRUE)
train$review = gsub('<.*?>', ' ', train$review)

#stop_words = c("i", "me", "my", "myself", 
#               "we", "our", "ours", "ourselves", 
#               "you", "your", "yours", 
#               "their", "they", "his", "her", 
#               "she", "he", "a", "an", "and",
#               "is", "was", "are", "were", 
#               "him", "himself", "has", "have", 
#               "it", "its", "the", "us")

stop_words = read.table("stop_words.txt",
                   stringsAsFactors = FALSE,
                   header = FALSE)


stop_words=as.character(stop_words)

it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)

tmp.vocab = create_vocabulary(it_train, 
                              stopwords = stop_words, 
                              ngram = c(1L,2L))

tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                             doc_proportion_max = 0.4,
                             doc_proportion_min = 0.0001)

dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))



set.seed(2685)

tmpfit = glmnet(x = dtm_train, 
                y = train$sentiment, 
                alpha = 1,
                family='binomial')


tmpfit$df

myvocab = colnames(dtm_train)[which(tmpfit$beta[, 64] != 0)]

setwd("~/data/CS598/MoviewReview")

getwd()

myvocab

write.table(myvocab, file = "myvocab.txt", 
            row.names = FALSE, col.names=FALSE)

