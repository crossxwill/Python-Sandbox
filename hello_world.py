# %% import packages
import numpy as np
import modin.pandas as pd
import statsmodels.api as stat
import matplotlib.pyplot as plt
import ray

ray.shutdown()

ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}},
         include_dashboard=False)

# %% simulate a data frame with 3 columns: x1, x2, y.
x1 = np.random.randn(100)
x2 = np.random.randn(100)
y = 2*x1 + 3*x2 + 4 + np.random.normal(0,4,100)

my_df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# %% fit a linear model called "lin_mod" using my_df using the formula API
lin_mod = stat.OLS.from_formula('y ~ x1 + x2', data=my_df._to_pandas())
lin_mod_1 = lin_mod.fit()

# %% summary of the model
print(lin_mod_1.summary())

# %% plot the residuals of lin_mod against the prediction
plt.scatter(lin_mod_1.predict(), lin_mod_1.resid)
plt.ylabel('Residuals')
plt.xlabel('Predictions')
plt.show()
# %%


ray.shutdown()
# %%
