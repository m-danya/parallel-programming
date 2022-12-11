#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>


const int CIRCLE_START_RANK = 5;

const int ELECTION_TAG = 1;
const int I_AM_HERE_TAG = 2;
const int COORDINATOR_TAG = 3;

void roll_call(int rank) {
    printf("%d ", rank);
    fflush(stdout);
}

int send_to_next(int rank, int size, int *array, int tag) {
    // tries to send the message and returns the rank of the next alive process
    MPI_Request request;
    MPI_Status status;
    int answer;
    int successfully_sent = 0;
    int next = 1;
    while (!successfully_sent) {
        MPI_Isend(array, size, MPI_INT, (rank + next) % size, tag, MPI_COMM_WORLD, &request);
        MPI_Irecv(&answer, 1, MPI_INT, (rank + next) % size, I_AM_HERE_TAG, MPI_COMM_WORLD, &request);

        double start = MPI_Wtime();
        while (!successfully_sent) {
            MPI_Test(&request, &successfully_sent, &status);
            // the timeout is 1 second
            if (MPI_Wtime() - start >= 1) {
                printf("%2d: Didn't get a confirmation from %2d.\n", rank, (rank + next) % size);
                fflush(stdout);
                MPI_Cancel(&request);
                MPI_Request_free(&request);
                break;
            }
        }
        next++;
    }
    return status.MPI_SOURCE;
}

int calculate_coordinator(int *array, int size) {
    // set alive process with the highest rank as a coordinator
    for (int i = size - 1; i >= 0; i--) {
        if (array[i]) {
            return i;
        }
    }
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
    setvbuf(stderr, NULL, _IOLBF, BUFSIZ);
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL) * rank);

    bool am_i_alive = true;

    if (rank == CIRCLE_START_RANK) {
        printf("\n\nAlive processes:\n");
        fflush(stdout);
    } else {
        am_i_alive = rand() > 0.42 * RAND_MAX;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (am_i_alive) {
        roll_call(rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!am_i_alive) {
        MPI_Finalize();
        return 0;
    }

    int *array = (int *) calloc(size, sizeof(int));

    MPI_Request request;
    MPI_Status status;
    int next;

    if (rank == CIRCLE_START_RANK) {
        printf("\nStating the algorithm\n");
        fflush(stdout);
        array[rank] = 1;
        next = send_to_next(rank, size, array, ELECTION_TAG);
    }

    // the election process:
    int answer = 1;
    int new_coordinator;

    MPI_Irecv(array, size, MPI_INT, MPI_ANY_SOURCE, ELECTION_TAG, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
    printf("%2d: Got array from %2d\n", rank, status.MPI_SOURCE);
    fflush(stdout);

    if (array[rank]) {
        // the value is 1 -> we've already been here ->
        // we need to sum up the election results.
        new_coordinator = calculate_coordinator(array, size);
        printf("%2d: New coordinator: %2d\n", rank, new_coordinator);
        fflush(stdout);
        MPI_Isend(&answer, 1, MPI_INT, status.MPI_SOURCE, I_AM_HERE_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&new_coordinator, 1, MPI_INT, next, COORDINATOR_TAG, MPI_COMM_WORLD, &request);
    } else {
        // update array and send to next
        array[rank] = 1;
        MPI_Isend(&answer, 1, MPI_INT, status.MPI_SOURCE, I_AM_HERE_TAG, MPI_COMM_WORLD, &request);
        next = send_to_next(rank, size, array, ELECTION_TAG);
    }

    // the second lap: sending new coordinator
    MPI_Irecv(&new_coordinator, 1, MPI_INT, MPI_ANY_SOURCE, COORDINATOR_TAG, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
    if (rank != CIRCLE_START_RANK) {
        printf("%2d: New coordinator: %2d\n", rank, new_coordinator);
        fflush(stdout);
        MPI_Isend(&new_coordinator, 1, MPI_INT, next, COORDINATOR_TAG, MPI_COMM_WORLD, &request);
    }

    MPI_Finalize();
    return 0;
}

