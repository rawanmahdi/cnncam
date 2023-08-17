from setuptools import setup, find_packages
with open("lib/README.md", 'r') as f:
    long_description = f.read()

setup(
    name="cnncam",
    package_dir={'':'lib'},
    packages=find_packages(where='lib'),
    # packages=['cnncam'],
    long_description=long_description, 
    long_description_content_type = 'text/markdown',
    version='0.0.2',
    description='Gradient Based Class Activation Maps for TensorFlow models.',
    py_modules=["cnncam"],
    install_requires= ["tensorflow>=2.0.0",
                       "numpy>=1.23",
                       "opencv-python>=4.7.0",
                       "matplotlib>=3.7.0"],
    extras_require = {
        "dev" : ["twine>=4.0.2"]
    }
)