import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing the libraries

data = pd.read_csv('ex1data1.txt',header = 'infer') # read from the dataset

X = data.iloc[:,0] #read first col

y = data.iloc[:,1] # read second col

m = len(y) #length of y ,the training example

data.head() #to view first few rows of the data

plt.scatter(X,y)

plt.xlabel('population of city in 10,000s')
plt.ylabel('profit in 10,000s')

plt.show()

X = X[:,np.newaxis]  # converting rank1 array to rank 2 
y = y[:,np.newaxis]  # same case as before

theta = np.zeros([2,1]) # creating 2 rows and 1 column of zeros

iterations = 1500

alpha = 0.0135

ones = np.ones((m,1))

X = np.hstack((ones,X)) # adding a column of 1s to X

def computeCost(X,y,theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.power(temp,2))/(2*m)
J = computeCost(X,y,theta)
print(J)

'''
this is using gradient descent to optimise 
def gradientDescent(X,y,theta,alpha,iterations):
    for i  in range(iterations):
        temp = np.dot(X,theta) - y
        temp = np.dot(X.T,temp)
        theta = theta - (alpha/m)*temp
        return theta 

theta = gradientDescent(X,y,theta,alpha,iterations)
'''
a = np.dot(X.T,X)
b = np.linalg.inv(a)
c = np.dot(b,X.T)
theta = np.dot(c,y)
print(theta)




J = computeCost(X,y,theta)
print(J)

plt.scatter(X[:,1],y)

plt.xlabel('population of city in 10,000s')

plt.ylabel('profit in 10,000s')

plt.plot(X[:,1],np.dot(X,theta))

plt.show()




