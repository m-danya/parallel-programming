module load SpectrumMPI
g++ -fopenmp -std=c++11 openmp.cpp -O2 -o openmp
mpic++ -fopenmp -std=c++11 mpi.cpp -O2 -o mpi
g++ -std=c++11 serial.cpp -O2 -o serial

