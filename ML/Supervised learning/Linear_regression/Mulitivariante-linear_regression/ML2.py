import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('ex1data2.txt', sep = ',',header = None)
print(data.head())
#print(data.describe())

fig,axes = plt.subplots(figsize= (12,4),nrows=1,ncols=2)
axes[0].scatter(data[0],data[2],color='b')
axes[0].set_xlabel('size (square feet)')
axes[0].set_ylabel('prizes')
axes[0].set_title('house prizes against size of house')
axes[1].scatter(data[1],data[2],color='g')
axes[1].set_xlabel('number of bedrooms')
axes[1].set_ylabel('prizes')
axes[1].set_title('house prize against number of bedrooms')
plt.tight_layout()
plt.show()




x = data.iloc[:,0:2] #reads 
# x1 = data.iloc[:,1:2]
y = data.iloc[:,2]
m = len(y)
#print(m)
#data.head()
#print(data.head())

x = (x - np.mean(x))/np.std(x)
'''colors = list(np.full(2,'k'))
plt.scatter(x,y, c=colors)
plt.ylabel('profit ')
plt.show()

fig = plt.figure()

ax = fig.add_subplot(111,projection = '3d')

ax.scatter3D(x,x1,y,c = y, cmap = 'Blues')

plt.show()
'''
ones = np.ones((m,1))
x = np.hstack((ones, x))
alpha = 0.01
num_iters = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis]
def computeCostMulti(x, y, theta):
    temp = np.dot(x, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)
J = computeCostMulti(x, y, theta)
print(J)

def gradientDescentMulti(x, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        temp = np.dot(x, theta) - y
        temp = np.dot(x.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescentMulti(x, y, theta, alpha, num_iters)
print(theta)
J = computeCostMulti(x, y, theta)
print(J)

