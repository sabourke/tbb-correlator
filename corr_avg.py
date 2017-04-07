#!/usr/bin/env python

import numpy as np
import sys

def main():
    arr_list = sys.argv[1:-1]
    out = np.load(arr_list[0])
    for i, arr in enumerate(arr_list[1:]):
	print arr
	out += (np.load(arr) - out) / (i + 2) # 2 because we're going from 1:
    np.save(sys.argv[-1], out)

if __name__ == "__main__":
    main()
