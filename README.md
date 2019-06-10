# FOLD
This is the First Implementation of FOLD (First Order Learner of Default) algorithm. FOLD is an ILP algorithm that learns a hypothesis from a background knowledge represented as normal logic program, with two disjoint sets of Positive and Negative examples. Learning hypotheses in terms of default theories (i.e., default and exceptions) outperforms Horn based ILP algorithms in terms of classification evaluation measures. It also significantly decreases the number of induced clauses. For more information about the original FOLD algorithm and learning default theories kindly refer to the [FOLD paper](https://arxiv.org/pdf/1707.02693.pdf "FOLD paper").

### SHAP_FOLD
This algorothm replaces the heuristic based search for best clause in ILP, with a technique from datamining known as High-Utility Itemset Mining. The idea is to use [SHAP](https://github.com/slundberg/shap "SHAP") to generate relevant features for each training data. Then our FOLD algorithm learns a set of Non-Monotonic clauses, that would capture the underlying logic of the Statistical Model from which SHAP features were extracted. 

## Install
