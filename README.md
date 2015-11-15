# gifted 

GIF creation and manipulation tool

## Installation
```
virtualenv --python=`which python3` .venv
source .venv/bin/activate
pip install -e .
```

## To run the unit tests and linter
```
tox -r
```

## To run
```
gifted --help
```
```
gifted --directory ~/workspace/photos --extension png --output-file test.gif
```
