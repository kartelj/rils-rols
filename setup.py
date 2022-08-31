from setuptools import setup

setup(name='rils',
    version='0.0',
    description='Symbolic Regression Tool Based On The Iterated Local Search',
    author='Aleksandar Kartelj, Marko Đjukanović',
    author_email='aleksandar.kartelj@gmail.com',
    url='https://github.com/kartelj/rils',
    packages = ['rils-rols'],
    package_dir = {'rils':'rils_rols'}, 
    install_requires=['joblib', "sklearn", "sympy", "chardet", "certifi", "idna"],
)