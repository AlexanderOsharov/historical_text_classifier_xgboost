from setuptools import setup, find_packages

setup(
    name='historical_text_classifier_xgboost',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'xgboost',
        'seaborn',
        'matplotlib',
        'chardet',
        'nltk'
    ],
    package_data={
        'historical_text_classifier_xgboost': ['data/dataset.json']
    }
)