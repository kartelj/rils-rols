# General

RILS-ROLS is metaheuristic-based framework to deal with problems of symbolic regression (and classification as of version 1.4). 

All of its aspects (method description, empirical results, etc.) are explained in the paper named:
"RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares" by Aleksandar Kartelj and Marko Djukanovic in the Journal of Big Data, Springer. 

All RILS-ROLS resources can be found at https://github.com/kartelj/rils-rols

RILS-ROLS distribution is available as a pip package at https://pypi.org/project/rils-rols
so it can be easily installed with the following pip command:

```bat
pip install rils-rols
```

# Minimal working example
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer
from rils_rols.rils_rols import RILSROLSRegressor, RILSROLSBinaryClassifier
from random import seed, randint
from math import sin, log


''' RILSROLSRegressor/RILSROLSClassifier parameters:
    1. max_fit_calls=100000             -- maximal number of fitness function calls
    2. max_seconds=100                  -- maximal running time in seconds
    3. complexity_penalty=0.001         -- expression size penalty (used for FitnessType.PENALTY) -- larger value means size is more important
    4. max_complexity = 50              -- the maximal size of internal expression (without symplification)
    5. sample_size=1                    -- the size of the sample taken from the training part
    6. verbose=False                    -- if True, the output during the program execution contains more details
    7. random_state=0                   -- random seed -- when 0 (default), the algorithm might produce different results in different runs
'''

random_state = 12345
samples = 200
train_size = 0.75
seed(random_state)

# toy regression dataset with known ground-truth 
X = list(zip([randint(1, 100) for _ in range(samples)], [randint(1, 100) for _ in range(samples)]))
y = [sin(x1)-78.8*log(x2)+4*x1+3.31*x2 for x1, x2 in X]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size, random_state=random_state)

# RILSROLSRegressor inherit BaseEstimator (sklearn), so we have fit, predict and score methods, where the score method is R2
regressor = RILSROLSRegressor(sample_size=1,random_state=random_state)
regressor.fit(X_train, y_train)
# this prints out the learned simplified model
print(f'Final model is:\t{regressor.model_string()}')
print(f'Training R2 score:\t{regressor.score(X_train, y_train)}')
print(f'Testing R2 score:\t{regressor.score(X_test, y_test)}')
# this prints some additional information as well, uncomment it to show it
#print(f'Other info:\t{regressor.fit_report_string()}')
print('--------------------------------------------------------------------------------------------------------------')

# now regression on the dataset without known ground-truth -- diabetes
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size, random_state=random_state)
regressor = RILSROLSRegressor(sample_size=1, max_complexity=20, random_state=random_state)
regressor.fit(X_train, y_train)
print(f'Final model is:\t{regressor.model_string()}')
print(f'Training R2 score:\t{regressor.score(X_train, y_train)}')
print(f'Testing R2 score:\t{regressor.score(X_test, y_test)}')
#print(f'Other info:\t{regressor.fit_report_string()}')
print('--------------------------------------------------------------------------------------------------------------')

# finally, binary classification on the sklearn toy dataset -- breast_cancer
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size, random_state=random_state)
regressor = RILSROLSBinaryClassifier(sample_size=1, max_complexity=20, random_state=random_state)
regressor.fit(X_train, y_train)
print(f'Final model is:\t{regressor.model_string()}')
print(f'Training accuracy score:\t{regressor.score(X_train, y_train)}')
print(f'Testing accuracy score:\t{regressor.score(X_test, y_test)}')
#print(f'Other info:\t{regressor.fit_report_string()}')
print('--------------------------------------------------------------------------------------------------------------')

```
# Citation

```
@article{kartelj2023rilsrols,
  title={RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares},
  author={Kartelj, Aleksandar and Djukanovi{\'c}, Marko},
  journal={Journal of Big Data},
  volume={10},
  number={71},
  year={2023},
  publisher={Springer}, 
  doi = {10.1186/s40537-023-00743-2},
}
```
