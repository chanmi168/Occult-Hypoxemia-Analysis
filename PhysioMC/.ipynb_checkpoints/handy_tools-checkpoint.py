import numpy as np
import pandas as pd
import os
import sys

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def ordinal(n):
# ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    return "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

