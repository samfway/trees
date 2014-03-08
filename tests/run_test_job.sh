#!/bin/bash
#PBS -N adaboy 
#PBS -joe
#PBS -q memroute
#PBS -l pvmem=32gb
cd /home/sawa6416/projects/trees
./build_models.py -i data/forest_train.csv -o /shared/write_test/models.pkl -s /shared/write_test/scale.pkl
./run_models.py -i data/forest_validation.csv --validation -s scale.pkl -m models.pkl -o ./preds_
