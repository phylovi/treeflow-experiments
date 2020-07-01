## treeflow-experiments

## Installing perf:

Debian:

    sudo apt-get install linux-perf

Ubuntu:

    sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
    sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
    sudo sh -c 'echo "kernel.perf_event_paranoid=-1" > /etc/sysctl.d/perf.conf'
    sudo sh -c " echo 0 > /proc/sys/kernel/kptr_restrict"

## Running perf:

    perf record -g -F 999 ./run.py
    perf report -g folded -i perf.data
