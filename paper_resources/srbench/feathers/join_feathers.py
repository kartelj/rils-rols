import pandas as pd

df1 = pd.read_feather('results/ground-truth_results.feather')
df2 = pd.read_feather('results/ground-truth_results_rils-rols.feather')
result = pd.concat([df1, df2]).reset_index()
result.to_feather('results/ground-truth_results_all.feather')
print('results saved to results/ground-truth_results_all.feather')