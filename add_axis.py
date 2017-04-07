#!/usr/bin/env python

import numpy as np
import sys

def main():
    d = np.load(sys.argv[1])
    np.save(sys.argv[1], d[None,...])

if __name__ == "__main__":
    main()
