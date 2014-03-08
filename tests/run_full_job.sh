#!/bin/bash
#PBS -N adaboy 
#PBS -joe
#PBS -q memroute
#PBS -l pvmem=32gb
cd /home/sawa6416/projects/trees
./build_models.py -i data/forest_all.csv -m /shared/write_test/models.pkl -s /shared/write_test/scale.pkl
./run_models_batch.py -i data/forest_test.csv -s /shared/write_test/scale.pkl -m /shared/write_test/models.pkl -o ./preds_
