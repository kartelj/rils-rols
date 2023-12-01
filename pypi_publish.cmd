:: change version number in setup.py
:: then call build
python setup.py clean --all sdist bdist_wheel
:: now, new .whl and .tar.gz files will be added inside dist folder
:: then call 
 python -m twine upload --verbose dist/*
:: or just a targeted version 
:: python -m twine upload --verbose dist/*1.4*