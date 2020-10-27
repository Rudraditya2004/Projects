# import modules 
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 

# setting the source
dataset = load_iris() 

# training the model
X_train, X_test, y_train, y_test = train_test_split(dataset.data,dataset.target,test_size=0.3,random_state=0) 
model = KNeighborsClassifier(n_neighbors=1) 
model.fit(X_train,y_train) 

# asking the user inputs 
sepal_length = float(input('Enter the sepal length (cm) : ')) 
sepal_width = float(input('Enter the sepal width (cm) : ')) 
petal_length = float(input('Enter the petal length (cm) : ')) 
petal_width = float(input('Enter the petal width (cm) : ')) 

# creating an array of the inputs 
New_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]]) 

# prediction of the name of the flower 
prediction = model.predict(New_data) 

# printing output
print('Predicted Target name : {}'.format(dataset.target_names[prediction]))
