#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N 516
double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;

double A[N][N][N];

void relax();
void init();
void verify();

int main(int an, char **as)
{
	int it;
	struct timeval t1, t2, elapsed_time;
	
	gettimeofday(&t1, NULL);

	init();

	for (it = 1; it <= itmax; it++)
	{
		eps = 0.;
		relax();
		// printf("it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) {
			break;
		}
	}

	verify();

	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &elapsed_time);
	// printf("%d threads: %f\n", \
	// 	omp_get_max_threads(), elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);
	printf("%f\n", \
		elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);


	return 0;
}

void init()
{
	#pragma omp parallel for collapse(3) default(none) \
		private(i, j, k) shared(A)
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
    #pragma omp parallel for collapse(3) default(none) \
		private(i, j, k) shared(A, w) reduction(max:eps)
	for (i = 1; i <= N - 2; i++)
		for (j = 1; j <= N - 2; j++)
			for (k = 1; k <= N - 2; ++k) {
				if ((k + i + j) % 2 == 1) {
					double b;
					b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
					eps = Max(fabs(b), eps);
					A[i][j][k] = A[i][j][k] + b;
				}
			}
	
    #pragma omp parallel for collapse(3) default(none) \
		private(i, j, k) shared(A, w)
	for (i = 1; i <= N - 2; i++)
		for (j = 1; j <= N - 2; j++)
			for (k = 1; k <= N - 2; ++k) {
				if ((k + i + j) % 2 == 0) {
					double b;
					b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
					A[i][j][k] = A[i][j][k] + b;
				}
			}
}

void verify()
{
	double s;

	s = 0.;
	#pragma omp parallel for collapse(3) default(none) \
		private(i, j, k) shared(A) reduction(+:s)
	for (i = 0; i <= N - 1; i++)
		for (j = 0; j <= N - 1; j++)
			for (k = 0; k <= N - 1; k++)
			{
				s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
			}
	// printf("  S = %f\n", s);
}
