#!/usr/bin/env bash

ENV=/data1/home/hippolyte.debernardi/anaconda3/envs/pneumonia/bin/python
SCRIPT="train.py"
MODEL="chest_xray"

oarsub -p "(gpu IS NOT NULL)" -l "walltime=96:0:0" -O "models/$MODEL/out.txt" -E "models/$MODEL/err.txt" "$ENV $SCRIPT"
