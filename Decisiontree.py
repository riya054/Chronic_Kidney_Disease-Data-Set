# Splitting the data into dependent and independent variables
X= df.iloc[:,:24]
y= df.iloc[:,24]
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from matplotlib import pyplot as plt
from sklearn import tree
fig=plt.figure(figsize=(28,12))
fig = tree.plot_tree(clf,feature_names=X.columns,  
                    class_names={0:'Notckd',1:'ckd'},
                    filled=True,
                    fontsize=12)

accuracy=accuracy_score(y_pred,y_test)
f1=f1_score(y_pred,y_test)
print(classification_report (y_test , y_pred))
print("\nConfusion matrix:")
confusion_matrix(y_test, y_pred)

#Testing sample input
predicted = clf.predict([[53,90,1.020,2,0,1,1,1,0,70,107,7.2,114,3.7,9.5,29,12100,3.7,1,1,0,0,0,1]])
print(predicted)
ans = np.round_(predicted,decimals = 0, out = None)
if (ans==1):
  print("ckd")
else:
  print("notckd")  
