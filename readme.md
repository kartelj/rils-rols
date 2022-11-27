RILS-ROLS is metaheuristic-based framework to deal with problems of symbolic regression. 

All of its aspects (method description, empirical results, etc.) are explained in the paper named:
"RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares" by Aleksandar Kartelj and Marko Djukanovic. 
This paper is currently under review in the Journal of Big Data, Springer. 

Minimal working example can be seen bellow:
```python
from rils_rols.rils_rols import RILSROLSRegressor
from math import sin, log

regr = RILSROLSRegressor()

X = [[3, 4], [1, 2], [-10, 20], [10, 10], [100, 100], [22, 23]]
y = [sin(x1)+2.3*log(x2) for x1, x2 in X]

regr.fit(X, y)

print("Final model is "+str(regr.model))
```

