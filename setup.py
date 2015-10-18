"""
Use setup tools to setup the armory has a standard python module
"""
from setuptools import setup

setup(
    name="gifted",
    version="0.0.1",
    description="Gif creation and manipulation tool",
    packages=["gifted"],
    install_requires=[
        'numpy',
        'Pillow',
    ]
)
