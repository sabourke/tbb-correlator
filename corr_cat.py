#!/usr/bin/env python

import numpy as np
import sys

def main():
    arr_list = sys.argv[1:-1]
    out = np.concatenate([np.load(a)[np.newaxis,...] for a in arr_list])
    np.save(sys.argv[-1], out)

if __name__ == "__main__":
    main()
