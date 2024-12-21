#! /usr/bin/env python3

import os
import sys
from datetime import datetime

##################################
RUN_OPENMP_INSTEAD_OF_MPI = False
RUN_SEQUENTIAL = False

RUN_1_THREADS_IN_MPI = True
RUN_4_THREADS_IN_MPI = False
# L = 1 OR PI in code

# ADD/REMOVE 2 FROM MPI processes if running with/without openmp
##################################

X_RUNS = 5
SERIAL_NAME = "serial"
OPENMP_NAME = "openmp"
MPI_NAME = "mpi"

if RUN_OPENMP_INSTEAD_OF_MPI:
    parallel_name = OPENMP_NAME
    processes = [1]
    threads = [1, 2, 4, 8]
    if not RUN_SEQUENTIAL:
        threads.remove(1)
else:
    parallel_name = MPI_NAME
    processes = [1, 2, 4, 8, 16, 32]
    threads = [1, 4]
    if not RUN_1_THREADS_IN_MPI:
        threads.remove(1)
    if not RUN_4_THREADS_IN_MPI:
        threads.remove(4)
    if not RUN_SEQUENTIAL:
        processes.remove(1)


current_time_str = datetime.now().strftime("%H_%M_%S")

with open("compile.sh", "w") as f:
    sys.stdout = f
    print("module load SpectrumMPI")
    print(f"g++ -fopenmp -std=c++11 {OPENMP_NAME}.cpp -O2 -o {OPENMP_NAME}")
    print(f"mpic++ -fopenmp -std=c++11 {MPI_NAME}.cpp -O2 -o {MPI_NAME}")
    print(f"g++ -std=c++11 serial.cpp -O2 -o {SERIAL_NAME}")
    print()

with open("submit.sh", "w") as f:
    sys.stdout = f
    for x in range(X_RUNS):
        for processes_number in processes:
            for threads_number in threads:
                for n in (128, 256, 512):
                    is_parallel = threads_number == 1 and processes_number == 1
                    executable = SERIAL_NAME if is_parallel else parallel_name
                    stdout_file = f"{current_time_str}_try_{x}_out_{n}_{processes_number}_{threads_number}.txt"
                    stderr_file = f"{current_time_str}_try_{x}_err_{n}_{processes_number}_{threads_number}.txt"
                    print(
                        f"mpisubmit.pl -p {processes_number} -t {threads_number} "
                        f"--stdout {stdout_file} --stderr {stderr_file} "
                        "-w 00:30 "
                        f"{executable} -- {n}"
                    )
sys.stdout = sys.__stdout__

# 1. ./compile on polus.
# 2. ENTER TMUX
# 3. python3 runner.py

print("GENERATED.")
# print("Uploading...")
# os.system(
#     f"scp compile.sh submit.sh {SERIAL_NAME}.cpp {OPENMP_NAME}.cpp {MPI_NAME}.cpp runner.py kill_all.sh polus:"
# )
