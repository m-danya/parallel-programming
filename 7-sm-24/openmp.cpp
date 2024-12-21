#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <chrono>

int N_PLUS_1 = 0; // hack, used only for indexing

double u_analytical(double x, double Lx, double y, double Ly, double z, double Lz, double a_t, double t)
{
    return (sin(2.0 * M_PI * x / Lx + 3.0 * M_PI) *
            sin(2.0 * M_PI * y / Ly + 2.0 * M_PI) *
            sin(M_PI * z / Lz) *
            cos(a_t * t + M_PI));
}

int inline index(int i, int j, int k)
{
    return i * N_PLUS_1 * N_PLUS_1 + j * N_PLUS_1 + k;
};

double compute_laplacian(int i, int j, int k,
                         const std::vector<double> &u_smth,
                         double hx, double hy, double hz,
                         int axis_size,
                         double tau)
{
    // 0..N -> same
    // N + 1 -> 1
    int ip = (i + 1) % axis_size;

    // 1..N -> same
    // 0 -> N

    // i >= 1
    int im = i - 1;
    // int im = (i - 1 + axis_size) % axis_size;

    // 0..N -> same
    // N + 1 -> 1
    int jp = (j + 1) % axis_size;

    // j >= 1
    int jm = j - 1;
    // int jm = (j - 1 + axis_size) % axis_size;

    // 7 точек:
    double u_cp = u_smth[index(i, j, k)];       // i j k
    double u_imjk = u_smth[index(im, j, k)];    // i-1 j k
    double u_ipjk = u_smth[index(ip, j, k)];    // i+1 j k
    double u_ijmk = u_smth[index(i, jm, k)];    // i j-1 k
    double u_ijpk = u_smth[index(i, jp, k)];    // i j+1 k
    double u_ijkm = u_smth[index(i, j, k - 1)]; // i j k-1
    double u_ijkp = u_smth[index(i, j, k + 1)]; // i j k+1

    double laplacian = (u_ipjk - 2.0 * u_cp + u_imjk) /
                           (hx * hx) +
                       (u_ijpk - 2.0 * u_cp + u_ijmk) /
                           (hy * hy) +
                       (u_ijkp - 2.0 * u_cp + u_ijkm) /
                           (hz * hz);
    return laplacian;
}

void calculate_and_print_eps(const int axis_size,
                             const double hx,
                             const double hy,
                             const double hz,
                             const int TIME_STEPS,
                             const double tau,
                             const double Lx,
                             const double Ly,
                             const double Lz,
                             const double a_t,
                             std::vector<double> &u_curr)
{
    double error = 0.0;
#pragma omp parallel for collapse(3) reduction(max : error)
    for (int i = 0; i < axis_size; ++i)
    {
        for (int j = 0; j < axis_size; ++j)
        {
            for (int k = 0; k < axis_size; ++k)
            {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                double t = TIME_STEPS * tau;

                double u = u_analytical(x, Lx, y, Ly, z, Lz, a_t, t);

                double diff = fabs(u_curr[index(i, j, k)] - u);

                if (diff > error)
                {
                    error = diff;
                }
            }
        }
    }

    std::cout << "Max eps at current step: " << error << std::endl;
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    if (argc < 2)
    {
        std::cerr << "ERROR: provide N as an argument!.\n";
        return 1;
    }

    int N = std::atoi(argv[1]);

    std::cout << "Running with N = " << N << std::endl;

    const double Lx = 1.0;
    const double Ly = Lx;
    const double Lz = Lx;

    const int TIME_STEPS = 20;

    const double hx = Lx / N;
    const double hy = Ly / N;
    const double hz = Lz / N;

    const double tau = hx / 100;

    const int axis_size = N + 1;

    N_PLUS_1 = N + 1;

    // const param for u_analytical
    const double a_t = M_PI * sqrt(4.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 1.0 / (Lz * Lz));

    std::vector<double> u_prev(axis_size * axis_size * axis_size, 0.0);
    std::vector<double> u_curr(axis_size * axis_size * axis_size, 0.0);
    std::vector<double> u_next(axis_size * axis_size * axis_size, 0.0);

// u_0 = u_prev = φ(x, y, z, 0)
#pragma omp parallel for collapse(3)
    for (int i = 0; i < axis_size; ++i)
    {
        for (int j = 0; j < axis_size; ++j)
        {
            for (int k = 0; k < axis_size; ++k)
            {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                double phi = u_analytical(x, Lx, y, Ly, z, Lz, a_t, 0.0);
                u_prev[index(i, j, k)] = phi;
            }
        }
    }

// u_1 = u_curr
#pragma omp parallel for collapse(3)
    for (int i = 1; i < axis_size; ++i)
    {
        for (int j = 1; j < axis_size; ++j)
        {
            for (int k = 0; k < axis_size; ++k)
            {
                // граничное условие первого рода для z:
                if (k == 0 || k == N)
                {
                    u_curr[index(i, j, k)] = 0.0;
                    continue;
                }

                double laplacian = compute_laplacian(i, j, k, u_prev, hx, hy, hz, axis_size, tau);
                double u_cp = u_prev[index(i, j, k)];
                double value = u_cp + 0.5 * tau * tau * laplacian;
                u_curr[index(i, j, k)] = value;
                if (i == N)
                {
                    u_curr[index(0, j, k)] = value;
                }
                if (j == N)
                {
                    u_curr[index(i, 0, k)] = value;
                }
            }
        }
    }

    for (int n = 1; n < TIME_STEPS; ++n)
    {
        double t = n * tau;

#pragma omp parallel for collapse(3)
        for (int i = 1; i < axis_size; ++i)
        {
            for (int j = 1; j < axis_size; ++j)
            {
                for (int k = 0; k < axis_size; ++k)
                {

                    // граничное условие первого рода для z:
                    if (k == 0 || k == N)
                    {
                        u_next[index(i, j, k)] = 0.0;
                        continue;
                    }

                    double laplacian = compute_laplacian(i, j, k, u_curr, hx, hy, hz, axis_size, tau);
                    double u_cp = u_curr[index(i, j, k)];

                    double value = 2.0 * u_cp - u_cp + tau * tau * laplacian;

                    u_next[index(i, j, k)] = value;
                    if (i == N)
                    {
                        u_curr[index(0, j, k)] = value;
                    }
                    if (j == N)
                    {
                        u_curr[index(i, 0, k)] = value;
                    }
                }
            }
        }
        std::swap(u_prev, u_curr);
        std::swap(u_curr, u_next);

        calculate_and_print_eps(axis_size, hx, hy, hz, TIME_STEPS, tau, Lx, Ly, Lz, a_t, u_curr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds.\n";
    return 0;
}
