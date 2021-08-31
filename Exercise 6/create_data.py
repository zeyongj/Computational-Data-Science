import pandas as pd
import numpy as np
import time
from implementations import all_implementations

def main():
     col = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
     n = 250
     data = pd.DataFrame(columns = col, index = np.arange(n)) # As required by the instruction.

     # The following codes are adapted from the instruction.
     # Implementation:
     for i in range(n):
       # self_name = 0
       random_array = np.random.randint(-2500,2500,5000) # N = 5000, which is sufficient large.
       for sort in all_implementations:
           st = time.time()
           res = sort(random_array)
           en = time.time()
           duration = en - st
           # data.iloc[i, self_name] = duration
           # self_name = self_name + 1
           self_name  = sort.__name__
           data.iloc[i] [self_name] = duration

     # The following code is copied from the instruction.
     data.to_csv('data.csv', index = False)

if __name__ == '__main__':
    main()

# Restriction #1: processor time. After testing, the running time is about 38s.
