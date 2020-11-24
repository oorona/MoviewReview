#! /usr/bin/Rscript
suppressPackageStartupMessages(library("pROC"))
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  split_number = 1 
  script="mymain.R"
} else  {
  script = args[1]
  split_number =args[2]
  vocab=as.integer(args[3])
}


setwd(split_number)
getwd()
source("mymain.R")

test.y <- read.table("test_y.tsv", header = TRUE)
pred <- read.table("mysubmission.txt", header = TRUE)
pred <- merge(pred, test.y, by="id")
roc_obj <- roc(pred$sentiment, pred$prob)
pROC::auc(roc_obj)