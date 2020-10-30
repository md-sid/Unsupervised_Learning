import itertools
from itertools import chain, combinations

import numpy as np
import pandas as pd

def main():
     df = pd.read_csv('iris.data', header=None)
     print(df.describe())
     print(df[0])
     print(len(df.columns))
     df[5] = np.nan
     print(df.describe())
     print(df.head())

if __name__ == "__main__":
    main()

