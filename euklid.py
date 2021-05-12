def gcd(a, b):
    # gcd(a,b) => a = b*q+r

    q = -1
    r = -1

    while (b != 0 and b != 1):
        print(f"a: {a} - b: {b}")
        q = int(a/b)
        r = a-b*q

        a = b
        b = r

    return a, b

def test():
    assert(gcd(2322, 654) == (6, 0))

test()