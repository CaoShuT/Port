#!/bin/bash
CONFIG=${1:-"configs/fcn_hr48_loveda.py"}
CHECKPOINT=${2:-"work_dirs/fcn_hr48_loveda/iter_80000.pth"}
SHOW_DIR=${3:-"work_dirs/fcn_hr48_loveda/vis_results"}

python test.py --config $CONFIG --checkpoint $CHECKPOINT --show-dir $SHOW_DIR
