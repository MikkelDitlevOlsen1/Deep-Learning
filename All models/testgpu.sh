#!/bin/sh
#BSUB -J torch
#BSUB -o torch_%J.out
#BSUB -e torch_%J.err
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 120
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
# module load scipy/VERSION
nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!


source deep_env/bin/activate
python Deep-Learning-main/Model_1.0_py.py
