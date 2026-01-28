#!/bin/bash

CALPRED_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Setting up the environment for calpred..."

# Setup the R environment:
module load gcc/12.3 r/4.3.1
mkdir -p "$CALPRED_PATH/calpred_R_env" || true

export R_LIBS="$CALPRED_PATH/calpred_R_env"

R -e 'install.packages("statmod", repos="https://cloud.r-project.org/")'
R -e 'install.packages("Rchoice", repos="https://cloud.r-project.org/")'

echo "= = = = = = == = =  = = = = == = = = = == = = ="
echo "Setting up the python environment for calpred..."

source "env/moe/bin/activate"
pip install git+https://github.com/KangchengHou/calpred.git

echo "Done!"
