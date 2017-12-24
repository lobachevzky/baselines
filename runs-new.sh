#! /usr/bin/env bash

port_num=${2:-8}

for i in $(seq 0 4);
do
  runs new "${1}${i}" "python baselines/ppo2/run_atari.py --seed={$i}" --overwrite
done
