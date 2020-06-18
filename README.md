## treeflow-experiments

## Installing perf:

    sudo apt-get install linux-perf

## Running perf:

    perf record -g -F 999 ./run.py
    perf report -g folded -i perf.data
