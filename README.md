# FOLD
This is the First Implementation of FOLD (First Order Learner of Default) algorithm. FOLD is an ILP algorithm that learns a hypothesis from a background knowledge represented as normal logic program, with two disjoint sets of Positive and Negative examples. Learning hypotheses in terms of default theories (i.e., default and exceptions) outperforms Horn based ILP algorithms in terms of classification evaluation measures. It also significantly decreases the number of induced clauses. For more information about the original FOLD algorithm and learning default theories kindly refer to the [FOLD paper](https://arxiv.org/pdf/1707.02693.pdf "FOLD paper").

### Example (From [FOLD Paper](https://arxiv.org/pdf/1707.02693.pdf "FOLD paper"))
FOLD Algorithm can handle nested levels of exceptions (i.e., Exceptions to defaults, exceptions to exceptions etc.)
Birds and planes normally fly, except penguins and damaged planes that can't. There are however superpenguins that can fly (exceptional penguins)

This example could be found in examples/flies folder. To run the example (after sucessful building from the sources), run the following command:
 <pre> java -jar fold.jar -mode fold flies.pl flies.txt </pre>

```
B:  bird(X) :- penguin(X).
    penguin(X) :- superpenguin(X).
    bird(a).
    bird(b).
    penguin(c).
    penguin(d).
    superpenguin(e).
    superpenguin(f).
    cat(c1).
    plane(g).
    plane(h).
    plane(h).
    plane(k).
    plane(m).
    damaged(k).
    damaged(m).
    -----------
E+: fly(a).
    fly(b).
    fly(e).
    fly(f).
    fly(g).
    fly(h).
E-: Closed World Assumption is implictly used.
   ------------
FOLD Learns the following hypothesis:
    fly(X) :- plane(X), not ab_plane(X).
    fly(X) :- bird(X), not ab_bird(X).
    fly(X) :- superpenguin(X).
    ab_plane(X) :- damaged(X).
    ab_bird(X) :- penguin(X).
```


### SHAP_FOLD
This algorithm replaces the heuristic based search for best clause in ILP, with a technique from datamining known as High-Utility Itemset Mining. The idea is to use [SHAP](https://github.com/slundberg/shap "SHAP") to generate relevant features for each training data. Then our FOLD algorithm learns a set of Non-Monotonic clauses, that would capture the underlying logic of the Statistical Model from which SHAP features were extracted. For more details refer to our [arXiv paper](https://arxiv.org/pdf/1905.11226.pdf). 

### Example
[UCI Car evaluation](https://archive.ics.uci.edu/ml/datasets/car+evaluation) contains examples with the six following attributes: buying price, maintenance, number of doors, persons (capacity), lug_boot size and safety. The following logic program will be learned by SHAP_FOLD algorithm using the insights taken from an XGBoost model. positive(A) indicates the examples for which a target property holds. In this dataset the target property is an acceptable car quality. Each clause states a default theory. For example the first clause says as long as car's safety is high, it has an acceptable quality unless it is an abnormal case, in which it's either too small (only fits 2 person) or the maintanance cost is high.
```
acceptable(A):-safety(A,high),not(ab0(A)).
acceptable(A):-persons(A,4),safety(A,med),not(ab2(A)).
acceptable(A):-lugboot(A,big),safety(A,med),persons(A,more).
acceptable(A):-safety(A,med),lugboot(A,med),persons(A,more),not(ab4(A)).
acceptable(A):-buying(A,med),safety(A,high),not(ab6(A)).
acceptable(A):-persons(A,4),safety(A,high),buying(A,low).
acceptable(A):-safety(A,high),buying(A,low),persons(A,more),not(ab8(A)).

ab0(A):-persons(A,2).
ab0(A):-maint(A,vhigh).
ab2(A):-buying(A,vhigh),lugboot(A,small).
ab2(A):-buying(A,high),maint(A,vhigh).
ab2(A):-maint(A,vhigh),buying(A,vhigh).
ab2(A):-buying(A,vhigh),maint(A,high).
ab2(A):-buying(A,high),lugboot(A,small).
ab4(A):-doors(A,2),maint(A,vhigh).
ab4(A):-doors(A,2),buying(A,vhigh).
ab6(A):-lugboot(A,small),persons(A,2).
ab6(A):-maint(A,high).
ab6(A):-maint(A,vhigh),persons(A,2).
ab6(A):-doors(A,2),persons(A,2).
ab6(A):-doors(A,2),lugboot(A,small).
ab6(A):-doors(A,3),persons(A,2).
ab8(A):-maint(A,vhigh),doors(A,2),lugboot(A,small).
ab8(A):-maint(A,high).
```

## Install 
### Prerequisites
We only support Windows at the moment.
* SHAP <pre> pip install shap </pre>
* XGBoost <pre> pip3 install xgboost </pre>
* [SWI-Prolog](http://www.swi-prolog.org/)  x64-win64, version 7.4.0-rc2
* [JPL Library](https://github.com/SWI-Prolog/packages-jpl) - There is no classpath or Jar file needed, however, the environment variables should be set according to the URL's instructions.
* [ant apache](https://ant.apache.org/) is used to compile java code and produce jar file
### Compile Sources (Create Jar file)
1. Clone this repository
2. Execute the following command: <pre> ant -buildfile build.xml </pre>
3. Upon Successful execution two new folders (i.e., build, dist) will be created.
4. Open the dist folder and copy the fold.jar to your conveneint destination folder. 
5. Copy foil.pl to the same destination folder. Our algorithm uses parts of the foil implementation (Courtesy of John M. Zelle and Raymond J. Mooney) and hence, JPL loads that file upon initialization. 

## Instructions
1. Data preparation
    + Create a single table dataset as ".csv" file.
    + Add a header row so that each column has a name. The class column should be named "label". Order is not important
    + Add a new column named "id" and use MS Excel to assign a unique integer to each data row.
2. Training a Statistical Model
    + Modify the "training.py" according the provided examples to work with your dataset.
    + Remember the dataset_name variable. It will be used later to run FOLD algorithm.
    + Run the Python script as follows:<pre> python training.py </pre>.
    + Upon sucessful execution it will produce 6 files including 2 Prolog files with ".pl" extension. Make sure that all files are put in the same folder that the jar file "UFOLD.java" is located.  
3. Run the FOLD algorithm as follows: <pre> java -jar fold.jar -mode shapfold <dataset_name> </pre>
4. Upon successful execution, the file "hypothesis.txt" will be created. It contains a logic program that would explain the underlying logic of the statistical model trained using "training.py" script.
