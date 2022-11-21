#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

conda activate condaenv36
pip install -r requirements.txt
python setup.py bdist_wheel

conda activate condaenv37
pip install -r requirements.txt
python setup.py bdist_wheel

conda activate condaenv38
pip install -r requirements.txt
python setup.py bdist_wheel


rename 's/linux/manylinux2014/' dist/*


pip install twine

twine upload --skip-existing -u alejomc dist/*
