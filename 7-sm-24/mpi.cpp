#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <mpi.h>

int N_PLUS_1 = 0; // Used only for indexing

double u_analytical(double x, double Lx, double y, double Ly, double z, double Lz, double a_t, double t)
{
    return (sin(2.0 * M_PI * x / Lx + 3.0 * M_PI) *
            sin(2.0 * M_PI * y / Ly + 2.0 * M_PI) *
            sin(M_PI * z / Lz) *
            cos(a_t * t + M_PI));
}

inline int local_index(int i, int j, int k, int local_ny, int local_nz)
{
    return i * local_ny * local_nz + j * local_nz + k;
}

double compute_laplacian(int i, int j, int k,
                         const std::vector<double> &u_smth,
                         double hx, double hy, double hz,
                         int local_nx, int local_ny, int local_nz)
{
    double u_cp = u_smth[local_index(i, j, k, local_ny, local_nz)];

    double u_imjk = u_smth[local_index(i - 1, j, k, local_ny, local_nz)];
    double u_ipjk = u_smth[local_index(i + 1, j, k, local_ny, local_nz)];
    double u_ijmk = u_smth[local_index(i, j - 1, k, local_ny, local_nz)];
    double u_ijpk = u_smth[local_index(i, j + 1, k, local_ny, local_nz)];
    double u_ijkm = u_smth[local_index(i, j, k - 1, local_ny, local_nz)];
    double u_ijkp = u_smth[local_index(i, j, k + 1, local_ny, local_nz)];

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
                             std::vector<double> &u_curr,
                             int local_nx, int local_ny, int local_nz,
                             int x_start, int y_start, int z_start,
                             MPI_Comm cart_comm)
{
    double local_error = 0.0;
#pragma omp parallel for collapse(3) reduction(max : local_error)
    for (int i = 1; i <= local_nx; ++i)
    {
        for (int j = 1; j <= local_ny; ++j)
        {
            for (int k = 1; k <= local_nz; ++k)
            {
                int global_i = x_start + (i - 1);
                int global_j = y_start + (j - 1);
                int global_k = z_start + (k - 1);

                double x = global_i * hx;
                double y = global_j * hy;
                double z = global_k * hz;
                double t = TIME_STEPS * tau;

                double u = u_analytical(x, Lx, y, Ly, z, Lz, a_t, t);

                double diff = fabs(u_curr[local_index(i, j, k, local_ny + 2, local_nz + 2)] - u);

                if (diff > local_error)
                {
                    local_error = diff;
                }
            }
        }
    }

    double global_error = 0.0;
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Max eps at current step: " << global_error << std::endl;
    }
}

int compute_local_size(int axis_size, int dims_i, int coord_i, int *start_i)
{
    // return localsize, write start_i by pointer
    int local_size;
    int base = axis_size / dims_i;
    int remainder = axis_size % dims_i;
    if (coord_i < remainder)
    {
        local_size = base + 1;
        *start_i = coord_i * local_size;
    }
    else
    {
        local_size = base;
        *start_i = coord_i * local_size + remainder;
    }
    return local_size;
}

void exchange_boundaries(std::vector<double> &u,
                         int local_nx, int local_ny, int local_nz,
                         MPI_Comm cart_comm)
{
    int rank;
    MPI_Comm_rank(cart_comm, &rank);

    int neighbor_left, neighbor_right;
    int neighbor_down, neighbor_up;
    int neighbor_back, neighbor_front;
    MPI_Cart_shift(cart_comm, 0, 1, &neighbor_left, &neighbor_right); // x
    MPI_Cart_shift(cart_comm, 1, 1, &neighbor_down, &neighbor_up);    // y
    MPI_Cart_shift(cart_comm, 2, 1, &neighbor_back, &neighbor_front); // z

    MPI_Request requests[12]; // 6 направлений * (Isend + Irecv)

    int request_count = 0;

    // FOR X
    int nyz = (local_ny + 2) * (local_nz + 2);
    std::vector<double> send_buffer_left(nyz);
    std::vector<double> recv_buffer_right(nyz);
    std::vector<double> send_buffer_right(nyz);
    std::vector<double> recv_buffer_left(nyz);

    // send left wrapping
    int idx = 0;
    for (int j = 0; j < local_ny + 2; ++j)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            send_buffer_left[idx++] = u[local_index(1, j, k, local_ny + 2, local_nz + 2)];
        }
    }

    // send right wrapping
    idx = 0;
    for (int j = 0; j < local_ny + 2; ++j)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            send_buffer_right[idx++] = u[local_index(local_nx, j, k, local_ny + 2, local_nz + 2)];
        }
    }

    MPI_Isend(send_buffer_left.data(), nyz, MPI_DOUBLE, neighbor_left, 0, cart_comm, &requests[request_count++]);
    MPI_Irecv(recv_buffer_right.data(), nyz, MPI_DOUBLE, neighbor_right, 0, cart_comm, &requests[request_count++]);

    MPI_Isend(send_buffer_right.data(), nyz, MPI_DOUBLE, neighbor_right, 1, cart_comm, &requests[request_count++]);
    MPI_Irecv(recv_buffer_left.data(), nyz, MPI_DOUBLE, neighbor_left, 1, cart_comm, &requests[request_count++]);

    // FOR Y
    int nxz = (local_nx + 2) * (local_nz + 2);
    std::vector<double> send_buffer_down(nxz);
    std::vector<double> recv_buffer_up(nxz);
    std::vector<double> send_buffer_up(nxz);
    std::vector<double> recv_buffer_down(nxz);

    // send down wrapping
    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            send_buffer_down[idx++] = u[local_index(i, 1, k, local_ny + 2, local_nz + 2)];
        }
    }

    // send up wrapping
    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            send_buffer_up[idx++] = u[local_index(i, local_ny, k, local_ny + 2, local_nz + 2)];
        }
    }

    MPI_Isend(send_buffer_down.data(), nxz, MPI_DOUBLE, neighbor_down, 2, cart_comm, &requests[request_count++]);
    MPI_Irecv(recv_buffer_up.data(), nxz, MPI_DOUBLE, neighbor_up, 2, cart_comm, &requests[request_count++]);

    MPI_Isend(send_buffer_up.data(), nxz, MPI_DOUBLE, neighbor_up, 3, cart_comm, &requests[request_count++]);
    MPI_Irecv(recv_buffer_down.data(), nxz, MPI_DOUBLE, neighbor_down, 3, cart_comm, &requests[request_count++]);

    // FOR Z
    int nxy = (local_nx + 2) * (local_ny + 2);
    std::vector<double> send_buffer_back(nxy);
    std::vector<double> recv_buffer_front(nxy);
    std::vector<double> send_buffer_front(nxy);
    std::vector<double> recv_buffer_back(nxy);

    // send back wrapping
    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int j = 0; j < local_ny + 2; ++j)
        {
            send_buffer_back[idx++] = u[local_index(i, j, 1, local_ny + 2, local_nz + 2)];
        }
    }

    // send forward wrapping
    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int j = 0; j < local_ny + 2; ++j)
        {
            send_buffer_front[idx++] = u[local_index(i, j, local_nz, local_ny + 2, local_nz + 2)];
        }
    }

    MPI_Isend(send_buffer_back.data(), nxy, MPI_DOUBLE, neighbor_back, 4, cart_comm, &requests[request_count++]);
    MPI_Irecv(recv_buffer_front.data(), nxy, MPI_DOUBLE, neighbor_front, 4, cart_comm, &requests[request_count++]);

    MPI_Isend(send_buffer_front.data(), nxy, MPI_DOUBLE, neighbor_front, 5, cart_comm, &requests[request_count++]);
    MPI_Irecv(recv_buffer_back.data(), nxy, MPI_DOUBLE, neighbor_back, 5, cart_comm, &requests[request_count++]);

    // WAIT
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    // UNWRAPPING

    idx = 0;
    for (int j = 0; j < local_ny + 2; ++j)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            u[local_index(local_nx + 1, j, k, local_ny + 2, local_nz + 2)] = recv_buffer_right[idx++];
        }
    }

    idx = 0;
    for (int j = 0; j < local_ny + 2; ++j)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            u[local_index(0, j, k, local_ny + 2, local_nz + 2)] = recv_buffer_left[idx++];
        }
    }

    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            u[local_index(i, local_ny + 1, k, local_ny + 2, local_nz + 2)] = recv_buffer_up[idx++];
        }
    }

    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int k = 0; k < local_nz + 2; ++k)
        {
            u[local_index(i, 0, k, local_ny + 2, local_nz + 2)] = recv_buffer_down[idx++];
        }
    }

    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int j = 0; j < local_ny + 2; ++j)
        {
            u[local_index(i, j, local_nz + 1, local_ny + 2, local_nz + 2)] = recv_buffer_front[idx++];
        }
    }

    idx = 0;
    for (int i = 0; i < local_nx + 2; ++i)
    {
        for (int j = 0; j < local_ny + 2; ++j)
        {
            u[local_index(i, j, 0, local_ny + 2, local_nz + 2)] = recv_buffer_back[idx++];
        }
    }
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "ERROR: provide N as an argument!.\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);

    int N_PLUS_1 = N + 1;

    const double Lx = M_PI;
    const double Ly = Lx;
    const double Lz = Lx;

    const int TIME_STEPS = 20;

    const double hx = Lx / N;
    const double hy = Ly / N;
    const double hz = Lz / N;

    const double tau = hx / 100;

    const int axis_size = N + 1;

    // const param for u_analytical
    const double a_t = M_PI * sqrt(4.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 1.0 / (Lz * Lz));

    int dims[3] = {0, 0, 0}; // 3д сетка декартова
    MPI_Dims_create(size, 3, dims);

    int periods[3] = {1, 1, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int x_start, y_start, z_start;
    int local_nx = compute_local_size(axis_size, dims[0], coords[0], &x_start);
    int local_ny = compute_local_size(axis_size, dims[1], coords[1], &y_start);
    int local_nz = compute_local_size(axis_size, dims[2], coords[2], &z_start);

    int alloc_nx = local_nx + 2;
    int alloc_ny = local_ny + 2;
    int alloc_nz = local_nz + 2;

    std::vector<double> u_prev(alloc_nx * alloc_ny * alloc_nz, 0.0);
    std::vector<double> u_curr(alloc_nx * alloc_ny * alloc_nz, 0.0);
    std::vector<double> u_next(alloc_nx * alloc_ny * alloc_nz, 0.0);

// u_0 = u_prev = φ(x, y, z, 0)
#pragma omp parallel for collapse(3)
    for (int i = 1; i <= local_nx; ++i)
    {
        for (int j = 1; j <= local_ny; ++j)
        {
            for (int k = 1; k <= local_nz; ++k)
            {
                int global_i = x_start + (i - 1);
                int global_j = y_start + (j - 1);
                int global_k = z_start + (k - 1);

                double x = global_i * hx;
                double y = global_j * hy;
                double z = global_k * hz;
                double phi = u_analytical(x, Lx, y, Ly, z, Lz, a_t, 0.0);
                u_prev[local_index(i, j, k, alloc_ny, alloc_nz)] = phi;
            }
        }
    }

    // u_1 = u_curr
    exchange_boundaries(u_prev, local_nx, local_ny, local_nz, cart_comm);
#pragma omp parallel for collapse(3)
    for (int i = 1; i <= local_nx; ++i)
    {
        for (int j = 1; j <= local_ny; ++j)
        {
            for (int k = 1; k <= local_nz; ++k)
            {
                int global_k = z_start + (k - 1);

                if (global_k == 0 || global_k == N)
                {
                    u_curr[local_index(i, j, k, alloc_ny, alloc_nz)] = 0.0;
                    continue;
                }

                double laplacian = compute_laplacian(i, j, k, u_prev, hx, hy, hz, alloc_nx, alloc_ny, alloc_nz);
                double u_cp = u_prev[local_index(i, j, k, alloc_ny, alloc_nz)];
                double value = u_cp + 0.5 * tau * tau * laplacian;
                u_curr[local_index(i, j, k, alloc_ny, alloc_nz)] = value;
            }
        }
    }

    for (int n = 1; n < TIME_STEPS; ++n)
    {
        double t = n * tau;

        exchange_boundaries(u_curr, local_nx, local_ny, local_nz, cart_comm);

#pragma omp parallel for collapse(3)
        for (int i = 1; i <= local_nx; ++i)
        {
            for (int j = 1; j <= local_ny; ++j)
            {
                for (int k = 1; k <= local_nz; ++k)
                {
                    int global_k = z_start + (k - 1);

                    if (global_k == 0 || global_k == N)
                    {
                        u_next[local_index(i, j, k, alloc_ny, alloc_nz)] = 0.0;
                        continue;
                    }

                    double laplacian = compute_laplacian(i, j, k, u_curr, hx, hy, hz, alloc_nx, alloc_ny, alloc_nz);
                    double u_cp = u_curr[local_index(i, j, k, alloc_ny, alloc_nz)];

                    double value = 2.0 * u_cp - u_prev[local_index(i, j, k, alloc_ny, alloc_nz)] + tau * tau * laplacian;

                    u_next[local_index(i, j, k, alloc_ny, alloc_nz)] = value;
                }
            }
        }

        std::swap(u_prev, u_curr);
        std::swap(u_curr, u_next);

        calculate_and_print_eps(axis_size, hx, hy, hz, TIME_STEPS, tau, Lx, Ly, Lz, a_t, u_curr,
                                local_nx, local_ny, local_nz, x_start, y_start, z_start, cart_comm);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Finalize();
    if (rank == 0)
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds.\n";
    }
    return 0;
}
