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
        'scikit-learn>=0.22',
        'version_parser>=1.0.0',
        'numpy>=1.17.4',
        'python-interface>=1.5.1',
        "hiplot>=0.1.8.post3",
        'cufflinks-study @ git+git://github.com/lucasiscovici/cufflinks#egg=cufflinks_study',
        'plotly-study @ git+git://github.com/lucasiscovici/plotly_py#egg=plotly_study',
        'cvopt-study @ git+git://github.com/lucasiscovici/cvopt#egg=cvopt_study',
        'mpld3_study @ git+git://github.com/lucasiscovici/mpld3_study#egg=mpld3_study',
        'speedml_study @ git+git://github.com/lucasiscovici/speedml_study#egg=speedml_study',
        'dora_study @ git+git://github.com/lucasiscovici/Dora_study#egg=dora_study',
        'studyPipe @ git+git://github.com/lucasiscovici/studyPipe#egg=studyPipe',
        'snakeviz-study @ git+git://github.com/lucasiscovici/snakeviz2#egg=snakeviz_study',
        'pandas-profiling-study @ git+git://github.com/lucasiscovici/pandas-profiling-study#egg=pandas_profiling_study'
    ]
)
