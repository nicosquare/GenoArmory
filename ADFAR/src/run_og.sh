#!/usr/bin/env bash
tasks=("og")

cd /projects/p32013/DNABERT-meta/ADFAR/src

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python run.py --task $task
done

echo "All tasks completed."
