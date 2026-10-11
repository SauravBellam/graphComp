#!/bin/bash
cd ~/graphComp/governDynamics-upbeat-gauss-uswau0
mkdir -p ~/results
echo "START $(date)"
python3 planA/run_all.py --data ~/graphComp/archive --out ~/results --cache ~/cache --stages a1,a4,a5 --a1-shards 8 --jobs 8
echo "RUN_ALL_EXIT $? $(date)"
python3 planA/a3_encoder.py --data ~/graphComp/archive --out ~/results/a3 --stats-only --graphs 10
echo "A3_STATS_EXIT $? $(date)"
