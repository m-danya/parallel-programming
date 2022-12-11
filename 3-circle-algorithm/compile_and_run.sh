source ../dockervars.sh
rm main
mpicc main.c -o main
mpirun --oversubscribe -n 36 main
