def isPow2(x):
    return (x!=0) and (x & (x-1)) == 0

import numpy as np
PI = np.pi

def filterByThreshold(y, threshold):
    #shortcut, so that we don't need to iterate the whole signal
    if y.max() < threshold:
            # print("Values too small")
            return np.zeros(y.size)


    y = np.where(y < threshold, 0, y)  
    # for i in range(y.size):
    #     y[i] = 0 if abs(y[i]) < threshold else y[i]

    return y