def isPow2(x):
    return (x!=0) and (x & (x-1)) == 0