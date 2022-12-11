#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "mpi.h"
#include <mpi-ext.h>

#include <signal.h>
#include <setjmp.h>


#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N 10

int PROCESS_RANK_TO_BE_KILLED = 2;

MPI_Comm mpi_comm_world_custom;  // represents a logical group of current processes
jmp_buf jbuf;

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;
double local_eps;

double A[N][N][N];
double A_checkpoint[N][N][N];
double temp_sausages_row[N * N]; // 1 x N x N

int rank, num_workers, code;
int sausages_rows_per_proc;

int first_s_row, last_s_row;

void relax();
void init();
void verify();

void redistribute_sausages() {
    sausages_rows_per_proc = (N - 2) / num_workers;

    first_s_row = sausages_rows_per_proc * rank + 1;
    last_s_row = first_s_row + sausages_rows_per_proc;

    if (rank == num_workers - 1) {
        last_s_row = N - 1; // (counting from zero) => except last
    }
    printf("rank %d, from %d to %d\n", rank, first_s_row, last_s_row);
    fflush(stdout);
}


void copy_matrices(double from_matrix[N][N][N], double to_matrix[N][N][N]) {
    for (i = 1; i < N - 2; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                to_matrix[i][j][k] = from_matrix[i][j][k];
            }
        }
    }
}

static void verbose_errhandler(MPI_Comm *comm, int *err, ...) {
    PROCESS_RANK_TO_BE_KILLED = -1; // do not kill anyone else

    int len;
    char errstr[MPI_MAX_ERROR_STRING];

    MPI_Error_string(*err, errstr, &len);
    errstr[len] = 0;

    // printf("captured error: %s\n",errstr);

    MPIX_Comm_shrink(*comm, &mpi_comm_world_custom);
    MPI_Comm_rank(mpi_comm_world_custom, &rank);
    MPI_Comm_size(mpi_comm_world_custom, &num_workers);
    MPI_Barrier(mpi_comm_world_custom);

    redistribute_sausages();

    copy_matrices(A_checkpoint, A);
    MPI_Barrier(mpi_comm_world_custom);

    longjmp(jbuf, 0);
}


int main(int an, char **as)
{
    mpi_comm_world_custom = MPI_COMM_WORLD;

    if (code = MPI_Init(&an, &as)) {
        printf("error on start\n");
        MPI_Abort(mpi_comm_world_custom, code);
        return code;
    };
    MPI_Comm_rank(mpi_comm_world_custom, &rank);
    MPI_Comm_size(mpi_comm_world_custom, &num_workers);

    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    MPI_Comm_set_errhandler(mpi_comm_world_custom, errh);

    MPI_Barrier(mpi_comm_world_custom);

    redistribute_sausages();

    int it;
	struct timeval t1, t2, elapsed_time;
	if (!rank) {
		gettimeofday(&t1, NULL);
	}

	init();
    copy_matrices(A, A_checkpoint);


    for (it = 1; it <= itmax; it++)
	{
        setjmp(jbuf);  // return here, if an error occurred during execution of relax();
        eps = 0.;
		local_eps = 0.;
		relax();
		// printf("it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) {
			break;
		}
        // checkpoint
        copy_matrices(A, A_checkpoint);
    }

	verify();

	if (!rank) {
		gettimeofday(&t2, NULL);
		timersub(&t2, &t1, &elapsed_time);
		printf("%f, num_workers = %d\n", \
			elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0,
			num_workers);
	}
	MPI_Finalize();

	return 0;
}

void init()
{
	for (i = 0; i <= N - 1; i++)
		for (j = 0; j <= N - 1; j++)
			for (k = 0; k <= N - 1; k++)
			{
				if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
					A[i][j][k] = 0.;
				else
					A[i][j][k] = (4. + i + j + k);
			}
}

void relax()
{
	MPI_Status status;  
	MPI_Barrier(mpi_comm_world_custom);

    int //tag_LEFT = 2, tag_RIGHT = 3, 
		tag_UP = 4, tag_DOWN = 5;

	for (i = first_s_row; i < last_s_row; i++) {
		for (j = 1; j <= N - 2; j++) {
			for (k = 1; k <= N - 2; k++) {
				if ((k + i + j) % 2 == 1) {
					double b;
					b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
					local_eps = Max(fabs(b), local_eps);
					A[i][j][k] = A[i][j][k] + b;
				}
			}
		}
	}
	
	if (first_s_row != 1) {
		// если есть кто-то, кто обрабатывает ряд выше, 
		// то надо послать вверх все сосиски из нашей строки
		MPI_Send(A[first_s_row], N * N, MPI_DOUBLE, rank - 1, tag_UP, mpi_comm_world_custom);
	}
    if (last_s_row != N - 1) {
		// если есть, от кого принимать, то принимаем
        MPI_Recv(temp_sausages_row, N * N, MPI_DOUBLE, rank + 1, tag_UP, mpi_comm_world_custom, &status);
		int m = 0;
		for (j = 0; j <= N - 1; ++j) {
			for (k = 0; k <= N - 1; ++k) {
				A[last_s_row][j][k] = temp_sausages_row[m++];
			} 
		}
    }

    if (rank == PROCESS_RANK_TO_BE_KILLED) {
        printf("killed.\n");
        fflush(stdout);
        raise(SIGKILL);
    }

	if (last_s_row != N - 1) {
		// отправляем вниз
		MPI_Send(A[last_s_row - 1], N * N, MPI_DOUBLE, rank + 1, tag_DOWN, mpi_comm_world_custom);
	}
    if (first_s_row != 1) {
		// если есть, от кого принимать, то принимаем
        MPI_Recv(temp_sausages_row, N * N, MPI_DOUBLE, rank - 1, tag_DOWN, mpi_comm_world_custom, &status);
		int m = 0;
		for (j = 0; j <= N - 1; ++j) {
			for (k = 0; k <= N - 1; ++k) {
				A[first_s_row - 1][j][k] = temp_sausages_row[m++];
			} 
		}
    }
    MPI_Barrier(mpi_comm_world_custom); 

	for (i = 1; i <= N - 2; i++)
		for (j = 1; j <= N - 2; j++)
			for (k = 1; k <= N - 2; ++k) {
				if ((k + i + j) % 2 == 0) {
					double b;
					b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
					A[i][j][k] = A[i][j][k] + b;
				}
			}

	if (first_s_row != 1) {
		// если есть кто-то, кто обрабатывает ряд выше, 
		// то надо послать вверх все сосиски из нашей строки
		MPI_Send(A[first_s_row], N * N, MPI_DOUBLE, rank - 1, tag_UP, mpi_comm_world_custom);
	}
    if (last_s_row != N - 1) {
		// если есть, от кого принимать, то принимаем
        MPI_Recv(temp_sausages_row, N * N, MPI_DOUBLE, rank + 1, tag_UP, mpi_comm_world_custom, &status);
		int m = 0;
		for (j = 0; j <= N - 1; ++j) {
			for (k = 0; k <= N - 1; ++k) {
				A[last_s_row][j][k] = temp_sausages_row[m++];
			} 
		}
    }
	if (last_s_row != N - 1) {
		// отправляем вниз
		MPI_Send(A[last_s_row - 1], N * N, MPI_DOUBLE, rank + 1, tag_DOWN, mpi_comm_world_custom);
	}
    if (first_s_row != 1) {
		// если есть, от кого принимать, то принимаем
        MPI_Recv(temp_sausages_row, N * N, MPI_DOUBLE, rank - 1, tag_DOWN, mpi_comm_world_custom, &status);
		int m = 0;
		for (j = 0; j <= N - 1; ++j) {
			for (k = 0; k <= N - 1; ++k) {
				A[first_s_row - 1][j][k] = temp_sausages_row[m++];
			} 
		}
    }
	MPI_Barrier(mpi_comm_world_custom);

	MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_world_custom);
}

void verify()
{
	double local_s, s;

	local_s = 0.;
	int from = (first_s_row == 1 ? 0 : first_s_row);
    int to = last_s_row;
	for (i = from; i < to; i++) {
        for (j = 0; j <= N - 1; j++) {
            for (k = 0; k <= N - 1; k++) {
                local_s += A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }
        }
    }

	MPI_Reduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_world_custom);
	
	if (rank == 0) {
		printf("sum = %lf\n", s);
	}
}
