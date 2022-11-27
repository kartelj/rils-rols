from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rils-rols',
    version='0.4',
    description='RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares',
    long_description= long_description,
    long_description_content_type  = "text/markdown",
    author='Aleksandar Kartelj, Marko Đukanović',
    author_email='aleksandar.kartelj@gmail.com',
    url='https://github.com/kartelj/rils-rols',
    packages = find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent"
    ],
    python_requires= ">=3.6",
    py_modules=["rils-rols"],
    package_dir = {'rils-rols':'rils_rols'}
)