import pandas as pd
d = pd.read_csv('student-por.csv', sep=';')
print(len(d))
# binary pass/fail, all grades added togther (G1+G2+G3)
# grade are 0-20 pts each. sum>=35 = pass
d["pass"] = d.apply(lambda row: 1 if (row["G1"]+row["G2"]+row["G3"]) >= 35 else 0, axis=1)
d = d.drop(["G1", "G2", "G3"], axis=1)
d.head()
print(d)
# one-hot encoding
d = pd.get_dummies(d, columns=["sex" , "school" , "address" , "famsize" , "Pstatus" ,
"Mjob" , "Fjob" , "reason" , "guardian" , "schoolsup" , "famsup" , "paid" , 
"activities" , "nursery" , "higher" , "internet" , "romantic"])
print(d)
#shuffling rows
d = d.sample(frac=1)
# split training data (first 500 rows) from testing (remaining rows)
d_train = d[:500]
d_test = d[500:]

# apply attributes and save pass column seperately
d_train_att = d_train.drop(["pass"], axis=1)
d_train_pass = d_train["pass"]

d_test_att = d_test.drop(["pass"], axis=1)
d_test_pass = d_test["pass"]

d_att = d.drop(["pass"], axis=1)
d_pass = d["pass"]

# passing students in whole dataset
import numpy as np
print ("Passing: %d out of %d (%.2f%%)" % (np.sum(d_pass), len(d_pass), 100*float(np.sum(d_pass)) / len(d_pass)))

#start building the decision tree using sickit learn package
# use information gain to decide split time
from sklearn import tree
t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
t = t.fit(d_train_att, d_train_pass)
# tree
tree.export_graphviz(t, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                    feature_names=list(d_train_att), class_names=["fail", "pass"], 
                    filled=True, rounded=True)
print(t.score(d_test_att, d_test_pass))

# verify dataset

from sklearn.model_selection import cross_val_score
scores = cross_val_score(t, d_att, d_pass, cv=5)
# avg score + +/- two standard devs away
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# max depth choices
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))

#graph

depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = scores.mean()
    depth_acc[i,2] = scores.std() * 2
    i += 1

print(depth_acc)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()