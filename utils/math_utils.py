

def addition(*args):
    res = None
    for arg in args:
        if arg is None:
            continue
        if res is None:
            res = arg
        else: 
            res += arg
    return res

def multiplication(a, *args):
    if a is None:
        return None
    res = a
    for arg in args:
        res *= arg
    return res

def division(a, b, eps):
    if a is None:
        return None
    if eps is None:
        return a/b
    return a / (b + eps)
