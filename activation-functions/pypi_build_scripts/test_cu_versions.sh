#!/bin/bash

set -e

_V=0
while getopts "v" OPTION
do
  case $OPTION in
    v) _V=1
       ;;
  esac
done

CUDA_list=("cuda-10.2" "cuda-11.0")
CUDA_COMP_list=("cu102" "cu110")
######### CHECKING ###############


python_list=(python3.6 python3.7 python3.8)
py_list=(36 37 38)
torch_lib_list=(
  "/usr/local/lib64/python3.6/site-packages/torch/lib/"
  "/usr/local/lib/python3.7/site-packages/torch/lib/"
  "/usr/local/lib/python3.8/site-packages/torch/lib/")
 
for j in 0 1
do
  CUDA_V=${CUDA_list[$j]}
  CUDA_CP=${CUDA_COMP_list[$j]}
  CUDA_LIB="/usr/local/$CUDA_V/lib64/"
  export CUDA_HOME="/usr/local/$CUDA_V"
  CUDA_P="/usr/local/cuda"
  #if [[ -L "$CUDA_P" ]]
  #then
  #   echo "$CUDA_P is a symlink to a directory, changing it to correspond to $CUDA_V"
  #   rm -f $CUDA_P
  #else
  #   echo "Could not create symlink for $CUDA_HOME -> $CUDA_P"
  #fi
  #ln -s $CUDA_HOME $CUDA_P

  printf "Checking if python versions are correctly accessible and path to torch lib packages\n"
  for i in 0 1 2
  do
    PYTHON_V=${python_list[$i]}
    TORCH_LIB=${torch_lib_list[$i]}
    if [[ ! $TORCH_LIB =~ "$PYTHON_V" ]]
    then
      printf "Please provide a valid python torch lib path in \$TORCH_LIB\
              \ne.g. /usr/lib64/python3.6/site-packages/torch/lib/\
              \n$TORCH_LIB is not valid\n"
      exit 1
    fi

    if [ -z "$CUDA_LIB" ] || [ ! -d "$CUDA_LIB" ]
    then
      printf "Please provide a valid cuda lib path in \$CUDA_LIB\
              \ne.g. /usr/local/$CUDA_V/lib64/\n"
      exit 1
    fi
  done
  printf "All version correctly install with Rational's dependencies\n"
  unset PYTHON_V TORCH_LIB

  ######### CHECKING DONE  ###############


  # generate the wheels
  for i in 0 1 2
  do
    PYTHON_V=${python_list[$i]}
    PY_V=${py_list[$i]}
    TORCH_LIB=${torch_lib_list[$i]}
    export LD_LIBRARY_PATH=/usr/local/lib:$TORCH_LIB:$CUDA_LIB  # for it to be able to find the .so files
    # rm -f /usr/local/cuda
    # ln -s /usr/local/$CUDA_V ./cuda_path
    # $PYTHON_V -m venv testenv
    # echo "Created environment for $PYTHON_V"
    # source testenv/bin/activate
    # $PYTHON_V -m pip install -U pip wheel pytest
    # $PYTHON_V -m pip install -r pypi_build_scripts/tests_requirements.txt
    ls wheelhouse/$CUDA_V/rational_activations_$CUDA_CP*$PY_V*.whl
    $PYTHON_V -m pip install wheelhouse/$CUDA_V/rational_activations_$CUDA_CP*$PY_V*.whl
    cd tests
    $PYTHON_V -m pytest tests_keras
    $PYTHON_V -m pytest tests_mxnet 
    $PYTHON_V -m pytest tests_torch 
    
    exit
    rm -rf testenv
    exit
  done

done
