from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

#with open("readme.md", "r") as fh:
#    long_description = fh.read()
long_description = "You can find minimal working example and other information here: https://github.com/kartelj/rils-rols"

if os.name=='nt':
    compile_args = ["/std:c++17", "/I rils_rols_cpp", "/O2"]
else:
    os.environ['CCX'] = 'g++'
    os.environ['CC'] = 'g++'
    compile_args = ["-std=c++17", "-march=native", "-O3"]

ext_modules=[Pybind11Extension(
        'rils_rols_cpp', 
        sources=['rils_rols_cpp/rils_rols_cpp.cpp', 
                'rils_rols_cpp/node.cpp'], 
        extra_compile_args = compile_args
    )]

def copy_dir():
    dir_path = 'rils_rols_cpp'
    base_dir = os.path.join('.', dir_path)
    for (dirpath, _, files) in os.walk(base_dir):
        for f in files:
            print(f)
            yield os.path.join(dirpath.split('/', 1)[1], f)

setup(
    name='rils-rols',
    version='1.5.12',
    description='RILS-ROLS: Robust Symbolic Regression via Iterated Local Search and Ordinary Least Squares',
    long_description= long_description,
    long_description_content_type  = "text/markdown",
    author='Aleksandar Kartelj, Marko Đukanović, Ján Pigoš',
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
    python_requires= ">=3.8",
    install_requires=["scikit-learn", "sympy", "pybind11", "pandas"],
)