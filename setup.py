from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'studyProject', 'version.py')) as f:
    exec(f.read())
    
setup(
    name='studyProject',
    version=__version__,
    description='Conveniant Project Framework in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        'Source': 'https://github.com/luluperet/studyProject',
    },
    author='Lucas Iscovici',
    author_email='iscovici.lucas@yahoo.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='framework python project machine learning data science',
    packages=find_packages(exclude=[]),
    install_requires=[
        'scikit-learn==0.20.3',
        'numpy==1.16.2',
        'python-interface==1.5.1',
        'plotly_study @ git+git://github.com/lucasiscovici/plotly_py#egg=plotly_study',
        'cvopt_study @ git+git://github.com/lucasiscovici/cvopt#egg=cvopt_study',
        'studyPipe @ git+git://github.com/lucasiscovici/studyPipe#egg=studyPipe',
        'snakeviz_study @ git+git://github.com/lucasiscovici/snakeviz2#egg=snakeviz_study',

    ]
)
