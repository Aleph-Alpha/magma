#!/bin/bash
set -e

cwd=$(pwd)

case $PYTHON_V in
  python3.6)
    PY_V="36"
    ;;
  python3.7)
    PY_V="37"
    ;;
  python3.8)
    PY_V="38"
    ;;
  python3.9)
    PY_V="39"
    ;;
  *)
    printf "Please provide a python in \$PYTHON_V
            \npython3.6 or python3.7 or python3.8 or python3.9"
    exit 1
    ;;
esac


function log () {
  # if [[ $_V -eq 1 ]]; then
    if [[ 1 ]]; then
        echo "$@"
    fi
}


printf "\n\n\nauditwheel repairing\n"
auditwheel -v repair --plat manylinux_2_17_x86_64 dist/rational_activations*$PY_V*.whl # Repairing the wheel and puting it inside wheelhouse
cd wheelhouse/
$PYTHON_V -m wheel unpack rational_activations*$PY_V*manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd rational_activations*/rational_activations*.libs/
CORRUPTED_FILES_DIR='corrupted_files/'
mkdir -p $CORRUPTED_FILES_DIR
CORRUPTED_FILES=`find . -maxdepth 1 -type f | grep .so | sed 's/.\///g' `
WRONG_FILENAMES=()
REQUIREMENTS=`patchelf --print-needed ../rational/cuda.cpython-*$PY_V*-x86_64-linux-gnu.so`
printf "patching all files by hand\n"
for CORRUPTED_F in $CORRUPTED_FILES
do
  if [[ $CORRUPTED_F == *"-"* ]]; then
    WRONG_FILENAMES+=($CORRUPTED_F)
    log "--------------------------------------------------------"
    ORI_FILENAME="${CORRUPTED_F%-*}.so${CORRUPTED_F##*.so}"
    log "Found $CORRUPTED_F, searching original $ORI_FILENAME"
    if [[ `find $TORCH_LIB -name $ORI_FILENAME` ]]; then
      log "Found $ORI_FILENAME, avoiding replacement, just removing for model size..."
      # cp $TORCH_LIB/$ORI_FILENAME .
      mv $CORRUPTED_F $CORRUPTED_FILES_DIR
      sed -i "/rational.libs\/$CORRUPTED_F/d" ../*.dist-info/RECORD
      if [[ "${REQUIREMENTS}" =~ "$CORRUPTED_F" ]]; then
        patchelf --replace-needed $CORRUPTED_F $ORI_FILENAME ../rational/cuda.cpython-*$PY_V*-x86_64-linux-gnu.so
      fi
      continue
    elif [[ `find $CUDA_LIB -name $ORI_FILENAME` ]]; then
      log "Found $ORI_FILENAME"
      cp $CUDA_LIB/$ORI_FILENAME .
      mv $CORRUPTED_F $CORRUPTED_FILES_DIR
    else
      printf "Haven't been able to locate $ORI_FILENAME\
              \nTrying with $CORRUPTED_F"
      if [[ `find $TORCH_LIB -name $CORRUPTED_F` ]]; then
        log "Found $CORRUPTED_F\n Nothing to be changed"
        continue
      elif [[ `find $CUDA_LIB -name $CORRUPTED_F` ]]; then
        log "Found $CORRUPTED_F\n Nothing to be changed"
        continue
      else
        exit 1
      fi
    fi
    if [[ "${REQUIREMENTS}" =~ "$CORRUPTED_F" ]]; then
      patchelf --replace-needed $CORRUPTED_F $ORI_FILENAME ../rational/cuda.cpython-*$PY_V*-x86_64-linux-gnu.so
    fi
    log "removing line \"rational.libs/$CORRUPTED_F ...\" from RECORD"
    sed -i "/rational.libs\/$CORRUPTED_F/d" ../*.dist-info/RECORD
    export SHA256=($(sha256sum $ORI_FILENAME))
    log "And adding line \"rational.libs/$ORI_FILENAME ...\" into it"
    echo "rational.libs/$ORI_FILENAME,sha256=$SHA256,`stat --printf="%s" ../rational.libs/$ORI_FILENAME`" >> ../*.dist-info/RECORD
  fi
done
rm -rf $CORRUPTED_FILES_DIR
cd ../../
rm rational_activations-*$PY_V*-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Rational_WHEEL_DIR=`find . -maxdepth 1 -type d | grep rational_activations`
$PYTHON_V -m wheel pack $Rational_WHEEL_DIR  # creates the new wheel
wname=`ls rational_activations-0.2.1-*$PY_V*.whl`
mv $wname "${wname/manylinux2014_x86_64.manylinux_2_17_x86_64/manylinux_2_17_x86_64.manylinux2014_x86_64}"
rm -R `ls -1 -d rational_activations*/`  # removes the rational directory only
mkdir -p $CUDA_V
mv rational_activations-*$PY_V*-manylinux_2_17_x86_64.manylinux2014_x86_64.whl $CUDA_V
cd $cwd

unset TORCH_LIB CUDA_LIB PY_V  # To be sure they are reseted
