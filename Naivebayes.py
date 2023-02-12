# Splitting the data into dependent and independent variables
X= df.iloc[:,:24]
X = np.array(X, dtype='float32')
y= df.iloc[:,24]
y = np.array(y, dtype='float32')
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print (classification_report (y_test , y_pred))
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
