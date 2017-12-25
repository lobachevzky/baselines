#! /usr/bin/env bash

for i in $(seq 0 2);
do
  runs new "${1}${i}" "python baselines/ppo2/run_mlp.py --env=CartPole-v0 --seed=${i}" --overwrite
done
