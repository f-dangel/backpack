from setuptools import setup

# extract dependencies from 'requirements.txt' file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    author='F. Dangel',
    name='bpexts',
    version='0.2',
    description='Extended backward pass for feedforward networks in PyTorch',
    install_requires=requirements,
    url='https://github.com/f-dangel/bpexts',
    license='MIT',
    packages=['bpexts'],
    zip_safe=False)
