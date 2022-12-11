source ../dockervars.sh
rm main
mpicc main.c -o main
mpirun --oversubscribe -n 5 --with-ft ulfm main
