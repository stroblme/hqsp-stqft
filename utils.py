import numpy as np
ctr = 0
def isPow2(x):
    return (x!=0) and (x & (x-1)) == 0

PI = np.pi

def calcThreshold(y):
    pass #needsidea

def filterByThreshold(y, threshold):
    global ctr
    #shortcut, so that we don't need to iterate the whole signal
    if max(y) < threshold:
        ctr += 1
        # print("Values too small")
        # print(ctr)
        return np.zeros(len(y))


    # y = np.where(abs(y) < threshold, 0, y)  
    # for i in range(y.size):
    #     y[i] = 0 if abs(y[i]) < threshold else y[i]

    return y

