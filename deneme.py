import numpy as np
import random

In_size = 3
Out_size = 4

layers = []
Bias = np.zeros(5)
for A in range(5):
    Weight = np.random.normal(loc=0, scale=np.sqrt(2 / (In_size)))
    Weight = np.random.normal(loc=0, scale=np.sqrt(2 / (In_size)), size=(In_size, Out_size))
    layers.append([Weight, Bias])
    #print(weight)

copy = layers.copy()
newGen = layers[0:2]
newGen2 = layers[3:5]
newNewGen = newGen.extend(newGen2)
parent1 = layers[0][0]
parent2 = layers[1][0]
parentshape = np.shape(parent1) 
row = parentshape[0]
col = parentshape[1]

ofspring = []
"""
for A in range(len(layers)-len(newGen)):
    child = parent1
    for i in range(row):
        for j in range(col):
            if random.uniform(0.0,1.0) <= 0.9:
                print("i am here")
                child[i][j] = parent2[i][j]
    child = [child,np.zeros(5)]
    ofspring.append(child)

newGen.extend(ofspring)
"""

for A in range(len(layers)-len(newGen)):
    child = parent1
    for i in range(row):
        
        child = child
           
    child = [child,np.zeros(5)]
    ofspring.append(child)

rand = random.randint(0,3)


"""
print(layers[:2]) #TAKE FIRST TWO
print (np.shape(layers))
"""
MUTATION_MAX_MAGNITUDE = 1e-1
MUTATION_MIN_MAGNITUDE = 1e-3

random = random.uniform(MUTATION_MIN_MAGNITUDE,MUTATION_MAX_MAGNITUDE)

