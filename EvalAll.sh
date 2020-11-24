#! /bin/bash

find ~/data/CS598/MoviewReview/ -name "split_*" |parallel "./Eval.sh $1 $2 {}"
