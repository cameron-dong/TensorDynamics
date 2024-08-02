from setuptools import find_packages, setup

setup(
    name='TensorDynamics',
    packages=find_packages(include=['TensorDynamics']),
    version='0.1.0',
    description='Tensorflow atmospheric primitive equations solver',
    author='Cameron Dong',
    url="https://github.com/cameron-dong/TensorDynamics/" ,
    install_requires=['numpy==1.26.4', 'tensorflow==2.10','scipy==1.13.1','matplotlib','xarray','netcdf4','cartopy'],
)