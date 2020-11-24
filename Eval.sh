#! /bin/bash

cd ~/data/CS598/MoviewReview

cp ./$2 $3/myvocab.txt
cp ./$1 $3/mymain.R
time ./Eval.R $1 $3 $2
rm $3/myvocab.txt
rm $3/mymain.R
