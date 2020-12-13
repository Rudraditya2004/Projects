# importing modules
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

"""
0-> setosa
1-> versicolor
2-> virginica 
"""

# training the model
dataset = load_iris()
main_data = pd.DataFrame(data=dataset.data,columns=["Sepal Length","Sepal Width","Petal Length","Petal Width"])
main_data["Target"] = dataset["target"]

X_train,X_test,y_train,y_test = train_test_split(dataset["data"],dataset["target"],random_state=0)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)
# asking the sepal length, sepal width, petal length, and petal width
Sepal_length = float(input("Enter the sepal length (cm) : "))
Sepal_width = float(input("Enter the sepal width (cm) : "))
Petal_length = float(input("Enter the petal length (cm) : "))
Petal_width = float(input("Enter the petal width (cm) : "))
X_new = np.array([[Sepal_length,Sepal_width,Petal_length,Petal_width]])
prediction = model.predict(X_new)
# printing the predicted species and the accuracy
accuracy = model.score(X_test,y_test,sample_weight=None)*100
print("Predicted Species : {} ({:.2f}%)".format(dataset.target_names[prediction],accuracy))

# description of the data 
print()
print(main_data.describe())

# plotting the sepal length vs sepal width graph
sepal_length = main_data["Sepal Length"]
sepal_width = main_data["Sepal Width"]
plt.scatter(x=sepal_length[range(0,50)],y=sepal_width[range(0,50)],label="Setosa",alpha=0.6)
plt.scatter(x=sepal_length[range(50,100)],y=sepal_width[range(50,100)],label="Setosa",alpha=0.6)
plt.scatter(x=sepal_length[range(100,150)],y=sepal_width[range(100,150)],label="Virginica",alpha=0.6)
plt.title("Sepal Length V/s Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

# plotting the petal length vs petal width graph
petal_length = main_data["Petal Length"]
petal_width = main_data["Petal Width"]
plt.scatter(x=petal_length[range(0,50)],y=petal_width[range(0,50)],label="Setosa",alpha=0.6)
plt.scatter(x=petal_length[range(50,100)],y=petal_width[range(50,100)],label="Setosa",alpha=0.6)
plt.scatter(x=petal_length[range(100,150)],y=petal_width[range(100,150)],label="Virginica",alpha=0.6)
plt.title("Petal Length V/s Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()

# plotting the sepal length vs petal length graph
sepal_length = main_data["Sepal Length"]
petal_length = main_data["Petal Length"]
plt.scatter(x=sepal_length[range(0,50)],y=petal_length[range(0,50)],label="Setosa",alpha=0.6)
plt.scatter(x=sepal_length[range(50,100)],y=petal_length[range(50,100)],label="Setosa",alpha=0.6)
plt.scatter(x=sepal_length[range(100,150)],y=petal_length[range(100,150)],label="Virginica",alpha=0.6)
plt.title("Sepal Length V/s Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.show()

# plotting the sepal width vs petal width graph
sepal_width = main_data["Sepal Width"]
petal_width = main_data["Petal Width"]
plt.scatter(x=sepal_width[range(0,50)],y=petal_width[range(0,50)],label="Setosa",alpha=0.6)
plt.scatter(x=sepal_width[range(50,100)],y=petal_width[range(50,100)],label="Setosa",alpha=0.6)
plt.scatter(x=sepal_width[range(100,150)],y=petal_width[range(100,150)],label="Virginica",alpha=0.6)
plt.title("Sepal Width V/s Petal Width")
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()
plt.show()
