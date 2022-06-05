# %%

import numpy as np
import pandas as pd
import bz2
import compress_pickle
import sys
df = pd.DataFrame(np.random.randint(0, 100, size=(100000, 10)),
                  columns=list('ABCDEFGHIJ'))


df.to_csv('df.csv')
df.to_csv('df.csv', compression='bz2')

sys.getsizeof(df)

df.to_pickle("df.pkl")
compress_pickle.dump(df, "df.pkl.bz2")

# %%

pd.to_pickle()