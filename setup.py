from setuptools import setup
with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="cnn-cam",
    long_description=long_description, 
    long_description_content_type = 'text/markdown',
    version='0.0.1',
    description='Gradient Based Class Activation Maps for TensorFlow models.',
    py_modules=["tensorflow", "numpy"],
    package_dir={'': 'src'},
    extras_require = {
        "dev" : ["twine>=4.0.2"]
    }
)