# Splitting the data into dependent and independent variables
X= df.iloc[:,:24]
X = np.array(X, dtype='float32')
y= df.iloc[:,24]
y = np.array(y, dtype='float32')
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
model = keras.Sequential([
    keras.layers.Dense(3,input_shape=(24,),activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=1, validation_data=(X_test, y_test))
y_pred = (model.predict(X_test) > 0.5)
acc1=history.history['accuracy']

plt.plot(acc1)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

print (classification_report (y_test , y_pred))
print("\nConfusion matrix:")
confusion_matrix(y_test, y_pred)


#Testing sample input
predicted = model.predict([[53,90,1.020,2,0,1,1,1,0,70,107,7.2,114,3.7,9.5,29,12100,3.7,1,1,0,0,0,1]])
print(predicted)
ans = np.round_(predicted,decimals = 0, out = None)
if (ans==1):
  print("ckd")
else:
  print("notckd")
