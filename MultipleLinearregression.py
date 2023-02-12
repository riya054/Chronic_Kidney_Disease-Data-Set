import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Splitting the data into dependent and independent variables
X= df.iloc[:,:24]
X = np.array(X, dtype='float32')
y= df.iloc[:,24]
y = np.array(y, dtype='float32')
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X, y)
y_pred=regr.predict(X_test)
from sklearn.metrics import r2_score
Accuracy=r2_score(y_test,y_pred)*100
print(Accuracy)


#Testing sample input
predicted = regr.predict([[53,90,1.020,2,0,1,1,1,0,70,107,7.2,114,3.7,9.5,29,12100,3.7,1,1,0,0,0,1]])
print(predicted)
ans = np.round_(predicted,decimals = 0, out = None)
if (ans==1):
  print("ckd")
else:
  print("notckd")
