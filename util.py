"""
util.py

Created: December 2017
"""

def product(matrices):
    """
    Returns the product of a list of matrices.
    """
    if len(matrices) == 0:
        raise ValueError("Empty list of matrices. Cannot take the product.")

    prod = matrices[0]
    for i in range(1, len(matrices)):
        prod = prod.dot(matrices[i])
    return prod

def memoize(f):
    """
    Function decorator to memoize simple function calls.
    """
    class Memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            val = self[key] = self.f(*key)
            return val
    return Memodict(f)

def redis_encode(arr):
    """
    Turns a Numpy array or Python list into matlab matrix format
    """
    import numpy as np
    try:
        if isinstance(arr, np.ndarray):
            if len(arr.shape) == 1:
                return " ".join(map(str, arr.tolist()))
            else:
                return "; ".join([" ".join(map(str, row.tolist())) for row in arr])
        else:
            if not hasattr(arr[0], "__len__"):
                return " ".join(map(str, arr))
            else:
                return "; ".join([" ".join(map(str, arr)) for row in arr])
    except:
        return str(arr)

def redis_decode(s):
    """
    Turns matlab matrix string into a Numpy array
    """
    import numpy as np
    try:
        arr = [list(map(float, row.split(" "))) for row in s.split("; ")]
        if len(arr) == 1:
            return np.array(arr[0])
        return np.array(arr)
    except:
        return s

