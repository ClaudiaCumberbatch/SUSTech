import numpy as np
a = np.array([[6,6,6]])
a = np.append(a, [[1,2,3]], axis=0)
np.random.shuffle(a)
print(a)