'''

@author: Local-Admin
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    classset = []
    f =open("bezdekIris.data", 'r')
    lines=[line.strip() for line in f.readlines()]
      
    f.close()
    
    lines=[line.split(",") for line in lines if line]
    
    class1=np.array([line[:4] for line in lines if line[-1]=="Iris-setosa"], dtype=np.float)
    
    class2=np.array([line[:4] for line in lines if line[-1]=="Iris-versicolor"], dtype=np.float)
      
    class3=np.array([line[:4] for line in lines if line[-1]=="Iris-virginica"], dtype=np.float)
    
    data = np.array([line[:4] for line in lines], dtype=np.float)
    
    classset += [class1]
    classset += [class2]
    classset += [class3]

    return classset

def drawplot(data,w):
    class1 = np.dot(data[0], w)
    class2 = np.dot(data[1], w)
    class3 = np.dot(data[2], w)
    plt.plot(class1[:,0], class1[:,1], "bs", label="Iris-setosa")
    plt.plot(class2[:,0], class2[:,1], "go", label="Iris-versicolor")
    plt.plot(class3[:,0], class3[:,1], "rp", label="Iris-virginica")
    plt.legend()
    plt.show()

def criterion(S_w,S_b):
    A=np.dot(np.linalg.inv(S_w), S_b)
    J = np.trace(A)
    return J
    
if __name__ == '__main__':
    k = 3
    data = read_data()
    m = np.mean(data[-1],axis = 0)# the global mean
    m_in = [np.mean(data[i],axis = 0) for i in range(k)]
    S_w = sum([np.dot((data[i]-m_in[i]).T, (data[i]-m_in[i])) for i in range(k)])
    S_b = sum([np.outer((m_in[i]-m).T, (m_in[i]-m))*len(data[i]) for i in range(k)])
    A=np.dot(np.linalg.inv(S_w), S_b)
    value,vector = np.linalg.eig(A)
    J = criterion(S_w,S_b)
    v = value[:2]
    L = vector[:,:2]
    drawplot(data,L)
    print J

    