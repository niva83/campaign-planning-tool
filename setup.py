# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# import pypandoc

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# try:
#     import pypandoc
#     long_description = pypandoc.convert('README.md', 'rst')
# except(IOError, ImportError):
#     long_description = open('README.md').read()
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(

    name='campaign-planning-tool',  # Required
    version='0.1.3',  # Required
    description='Python library for scanning lidar campaign planning',  # Optional
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/niva83/campaign-planning-tool',  # Optional
    author='Nikola Vasiljevic',  # Optional
    author_email='niva@dtu.dk',  # Optional
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],

    packages=['campaign_planning_tool'],  # Required
    python_requires='>=3.7',
    install_requires=[
                      'numpy', 
                      'pandas', 
                      'geopandas', 
                      'whitebox', 
                      'srtm.py', 
                      'simplekml', 
                      'matplotlib', 
                      'pillow', 
                      'pyyaml', 
                      'dicttoxml',
                      'pylint',
                      'pathlib',
                      'jupyter', 
                      'rasterio',
                      ],
    zip_safe=False,
)