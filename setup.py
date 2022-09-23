from setuptools import setup

setup(name='rils-rols',
    version='0.0',
    description='RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares',
    author='Aleksandar Kartelj, Marko Đjukanović',
    author_email='aleksandar.kartelj@gmail.com',
    url='https://github.com/kartelj/rils-rols',
    packages = ['rils-rols'],
    package_dir = {'rils-rols':'rils_rols'}, 
    install_requires=['joblib', "sklearn", "sympy", "chardet", "certifi", "idna"],
)