import os
import numpy as np


max = 0
lengths = []
for dir in os.listdir("./Rad"):
    for file in os.listdir("./Rad/"+dir):
        l = len(np.load("./Rad/"+dir+"/"+file))
        print(l)
        lengths.append(l)
lengths.sort()
print(lengths)
