# %% import packages
import numpy as np
import vaex
import statsmodels.api as stat
import matplotlib.pyplot as plt
import duckdb


# %% simulate a data frame with 3 columns: x1, x2, y.
np.random.seed(123)
x1 = np.random.randn(65000)
x2 = np.random.randn(65000)
x3 = np.random.choice([1,2,3], 65000)
y = 2*x1 + 3*x2 + 4 + np.random.normal(0,4,65000)

my_df = vaex.from_dict({'x1': x1, 'x2': x2, 'x3':x3, 'y': y})

my_df_pandas = my_df.to_pandas_df()

mysummary = duckdb.sql("""select x3, sum(x1) as tot_x1,
           sum(x2) as tot_x2, avg(y) as avg_y
           from my_df_pandas
           group by x3""")

print(mysummary)

# %% fit a linear model called "lin_mod" using my_df using the formula API
lin_mod = stat.OLS.from_formula('y ~ x1 + x2', data=my_df_pandas)
lin_mod_1 = lin_mod.fit()

# %% summary of the model
print(lin_mod_1.summary())

# %% plot the residuals of lin_mod against the prediction
plt.scatter(lin_mod_1.predict(), lin_mod_1.resid)
plt.ylabel('Residuals')
plt.xlabel('Predictions')
plt.show()
# %%
# %%
