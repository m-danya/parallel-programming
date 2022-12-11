rm *.out
xlc rb.c -o rb -qsmp=omp
bsub < queue
