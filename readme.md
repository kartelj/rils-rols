RILS-ROLS is metaheuristic-based framework to deal with problems of symbolic regression. 

All of its aspects (method description, empirical results, etc.) are explained in the paper named:
"RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares" by Aleksandar Kartelj and Marko Djukanovic. 
This paper is currently under review in the Journal of Big Data, Springer. 

All RILS-ROLS resources can be found at https://github.com/kartelj/rils-rols

RILS-ROLS distribution is available as a pip package at https://pypi.org/project/rils-rols
so it can be easily installed with the following pip command:

```bat
pip install rils-rols
```

Minimal working example can be seen bellow:
```python
from rils_rols.rils_rols import RILSROLSRegressor
from math import sin, log

regr = RILSROLSRegressor()
''' regressor parameters:
    1. max_fit_calls=100000         -- maximal number of fitness function calls
    2. max_seconds=100              -- maximal running time in seconds
    3. complexity_penalty=0.001     -- expression complexity (size) penalty -- larger value means size is more important
    4. error_tolerance=1e-16        -- correlated with the level of expected noise in data -- higher value means higher expected noise
    5. random_state=0               -- random seed -- when 0 (default), the algorithm might produce different results in different runs
'''

# toy dataset 
X = [[3, 4], [1, 2], [-10, 20], [10, 10], [100, 100], [22, 23]]
y = [sin(x1)+2.3*log(x2) for x1, x2 in X]

# RILSROLSRegressor inherits BaseEstimator (sklearn), so we have well-known fit method
regr.fit(X, y)

# this prints out the learned model
print("Final model is "+str(regr.model))

# applies the model to a list of input vectors, also well-known predict method
X_test = [[4, 4], [3, 3]]
y_test = regr.predict(X_test)
print(y_test)
```
# Citation

```
@article{kartelj2022rils,
  title={RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares},
  author={Kartelj, Aleksandar and Djukanovi{\'c}, Marko},
  year={2022}
}
```
