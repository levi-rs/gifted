"""
Use setup tools to setup the repository
"""
from setuptools import setup, find_packages

setup(
    name="gifted",
    version="0.0.2",
    description="Gif creation and manipulation tool",
    url='https://github.com/levi-rs/gifted',
    author='Levi Noecker',
    author_email='levi.noecker@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',
    ],
    keywords="gif animated image",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ]
)
