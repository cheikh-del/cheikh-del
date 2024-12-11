from setuptools import setup, find_packages

setup(
    name='pubmed_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'spacy',
        'tqdm',
        'transformers',
        'joblib',
        # ajoutez d'autres dépendances si nécessaire
    ],
)
