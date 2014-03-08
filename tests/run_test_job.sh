#!/bin/bash
#PBS -N adaboy 
#PBS -joe
#PBS -q memroute
#PBS -l pvmem=32gb
cd /home/sawa6416/projects/trees
./build_models.py -i data/forest_train.csv -m /shared/write_test/models.pkl -s /shared/write_test/scale.pkl
./run_models.py -i data/forest_validation.csv --validation -s /shared/write_test/scale.pkl -m /shared/write_test/models.pkl -o ./preds_
