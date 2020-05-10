import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import json
import re

a = np.array([1, 2, 3, 4])
b = np.array([1, 3, 2, 4])

print(np.count_nonzero(a == b))