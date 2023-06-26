# %% import packages
import numpy as np
import pandas as pd
import statsmodels.api as stat
import matplotlib.pyplot as plt

# %% simulate a data frame with 3 columns: x1, x2, y.
x1 = np.random.randn(100)
x2 = np.random.randn(100)
y = 2*x1 + 3*x2 + 4 + np.random.normal(0,4,100)

# %% fit a linear model called "lin_mod"
lin_mod = stat.OLS(y, stat.add_constant(np.column_stack((x1, x2))))
lin_mod_1 = lin_mod.fit()

# %% summary of the model
print(lin_mod_1.summary())

# %% plot the residuals of lin_mod
plt.scatter(range(len(lin_mod_1.resid)), lin_mod_1.resid)
plt.ylabel('Residuals')
plt.xlabel('Observations')
plt.show()
# %%
