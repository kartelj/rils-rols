from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules=[Pybind11Extension(
        'rils_rols_cpp', 
        sources=['rils_rols_cpp/rils_rols_cpp.cpp', 
                'rils_rols_cpp/node.cpp',
                'rils_rols_cpp/utils.cpp'], 
        extra_compile_args = ["/std:c++20", "/I C:\Python312\include", "/Ox"]
    )]

setup(
    name='rils-rols',
    version='1.4',
    description='RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares',
    long_description= long_description,
    long_description_content_type  = "text/markdown",
    author='Aleksandar Kartelj, Marko Đukanović',
    author_email='aleksandar.kartelj@gmail.com',
    url='https://github.com/kartelj/rils-rols',
    packages = find_packages(),
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe = False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent"
    ],
    python_requires= ">=3.6",
    py_modules=["rils-rols"],
    package_dir = {'rils-rols':'rils_rols'}, 
    install_requires=["scikit-learn", "sympy", "statsmodels"],
)