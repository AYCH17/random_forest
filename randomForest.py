#%%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from subprocess import call 
# Display in jupyter notebook
from IPython.display import Image

#%%
data = pd.read_csv("./heart.csv")
data.head()
len(data.index)

#%%
data.dropna(inplace=True)
len(data.index)

data.replace(to_replace=['M','F','ATA','NAP','ASY','TA','Normal','ST','LVH','Down','Flat','Up','N','Y'],\
     value=[1,0,0,1,2,3,0,1,2,-1,0,1,0,1],inplace=True)

data_train, data_test = train_test_split(data, test_size=0.2)

# %%
y_train = data_train["HeartDisease"]
X_train = data_train.drop(columns=["HeartDisease"])

y_test = data_test["HeartDisease"]
X_test = data_test.drop(columns=["HeartDisease"])

# %%


clf = RandomForestClassifier(n_estimators=100,  criterion='gini', \
      max_depth=2, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
           max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, \
                oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False, \
                     class_weight=None, ccp_alpha=0.0, max_samples=None)

clf.fit(X_train, y_train)
# %%
y_pred = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

#%%
trees = {}
trees['0'] = clf.estimators_[0]
trees['10'] = clf.estimators_[10]
trees['90'] = clf.estimators_[90]
trees['99'] = clf.estimators_[99]

for i in [0,10,90,99]:

     export_graphviz(trees[f'{i}'], out_file=f'tree{i}.dot', 
                feature_names = X_train.columns,
                class_names = ['Normal','Heart Disease'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    
     call(['dot', '-Tpng', f'tree{i}.dot', '-o', f'tree{i}.png', '-Gdpi=600'],shell=True)
     
     Image(filename = f'tree{i}.png')


# %%

clf2 = RandomForestClassifier(n_estimators=100,  criterion='entropy', \
      max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
           max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, \
                oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False, \
                     class_weight=None, ccp_alpha=0.0, max_samples=None)

clf2.fit(X_train, y_train)

# %%
y_pred2 = clf2.predict(X_test)
acc2 = metrics.accuracy_score(y_test, y_pred2)
print(acc2)

#%%