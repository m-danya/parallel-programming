#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <limits>
#include <algorithm>

#include "mpi.h"
#include <mpi-ext.h>

using std::cin;
using std::cout;
using std::endl;

int n_proc, rank;

typedef std::vector<std::pair<int, int>> Comparators_vector;

void fill_in_array_with_random_numbers(double *full_values, int N)
{
    // this function is run only by 0-rank process
    srand(42);
    for (int i = 0; i < N; ++i)
    {
        full_values[i] = static_cast<double>(rand()) / RAND_MAX;
        // cout << i << " " << full_values[i] << endl;
    }
}

void add_comp(int a, int b, Comparators_vector &c)
{
    c.push_back(std::make_pair(a, b));
}

// слить отсортированные массивы (a_start, step, n_a) и (b_start, step, n_b)
void odd_even_merge(int a_start, int b_start, int step, int n_a, int n_b, Comparators_vector &c)
{
    if (!n_a || !n_b)
    {
        return;
    }
    if (n_a == 1 && n_b == 1)
    {
        add_comp(a_start, b_start, c);
        return;
    }
    // количество нечетных строк в a
    int odd_lines_number_in_a = n_a - n_a / 2;
    // количество нечетных строк в b
    int odd_lines_number_in_b = n_b - n_b / 2;

    // слияние всех нечетных строк. Так как массивы a и b отсортированы,
    // то и a[::2] и b[::2] тоже отсортированы!
    odd_even_merge(a_start, b_start, 2 * step, odd_lines_number_in_a, odd_lines_number_in_b, c);
    // слияние всех четных строк
    odd_even_merge(a_start + step, b_start + step, 2 * step, n_a - odd_lines_number_in_a, n_b - odd_lines_number_in_b, c);

    // все четные и все нечетные строки уже слиты к этому моменту,
    // осталось добавить компараторы  [i:i+1] для всех i \in {1, 3, 5, 7, ..., n-3}

    for (int i = 1; i < n_a - 1; i += 2)
    {
        add_comp(a_start + step * i, a_start + step * (i + 1), c);
    }

    // если количество элементов первого массива чётно, то нужно добавить дополнительный
    // компаратор между массивами:
    //
    // n_a = 5:
    // a1 (a2 a3) (a4 a5)      (b1 b2) (b3 b4) b5
    // n_a = 6:
    // a1 (a2 a3) (a4 a5) (a6  b1) (b2 b3) (b4 b5)

    if (!(n_a % 2))
    {
        add_comp(a_start + step * (n_a - 1), b_start, c);
    }

    for (int i = !(n_a % 2); i < n_b - 1; i += 2)
    {
        add_comp(b_start + step * i, b_start + step * (i + 1), c);
    }
}

void odd_even_sort(int first, int step, int n, Comparators_vector &c)
{
    if (n < 2)
    {
        return;
    }
    if (n == 2)
    {
        add_comp(first, first + step, c);
        return;
    }

    int first_half_n = (n + 1) / 2;
    int second_half_n = n - first_half_n;

    odd_even_sort(first, step, first_half_n, c);
    odd_even_sort(first + step * first_half_n, step, second_half_n, c);

    odd_even_merge(first, first + step * first_half_n, step, first_half_n, second_half_n, c);
}

void join(double *a, double *b, double *c, int n, int rank1, int rank2, bool reverse)
{
    if (rank == rank1)
    {
        int ia = 0;
        int ib = 0;
        int k = 0;
        while (k < n)
        {
            if (!reverse && a[ia] < b[ib] || reverse && a[ia] > b[ib])
            {
                c[k++] = a[ia++];
            }
            else
            {
                c[k++] = b[ib++];
            }
        }
    }
    else
    {
        int ia = n - 1;
        int ib = n - 1;
        int k = n - 1;
        while (k >= 0)
        {
            if (!reverse && a[ia] > b[ib] || reverse && a[ia] < b[ib])
            {
                c[k--] = a[ia--];
            }
            else
            {
                c[k--] = b[ib--];
            }
        }
    }
}

void participate_in_sorting_network(double *local_values, int elements_per_process, Comparators_vector &comparators, bool reverse)
{
    double *other_values = (double *)malloc(sizeof(double) * elements_per_process);
    double *local_values_copy = (double *)malloc(sizeof(double) * elements_per_process);

    for (auto const &c : comparators)
    {
        int other_rank;
        if (c.first != rank && c.second != rank)
        {
            continue;
        }
        if (c.first == rank)
        {
            other_rank = c.second;
            MPI_Send(local_values, elements_per_process, MPI_DOUBLE,
                     other_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(other_values, elements_per_process, MPI_DOUBLE,
                     other_rank, 0, MPI_COMM_WORLD, nullptr);
        }
        else
        {
            other_rank = c.first;
            MPI_Recv(other_values, elements_per_process, MPI_DOUBLE,
                     other_rank, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(local_values, elements_per_process, MPI_DOUBLE,
                     other_rank, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < elements_per_process; ++i)
        {
            local_values_copy[i] = local_values[i];
        }
        join(local_values_copy, other_values, local_values, elements_per_process, c.first, c.second, reverse);
    }
    free(other_values);
    free(local_values_copy);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Comparators_vector comparators;
    odd_even_sort(0, 1, n_proc, comparators);
    cout << "(elapsed) tacts = " << comparators.size() << endl;
    // if (argc < 2)
    // {
    //     throw std::runtime_error("Specify N as the first argument");
    // }
    // bool reverse = false;
    // if (argc == 3)
    // {
    //     reverse = true;
    // }
    // int N_theoretical = std::stoi(argv[1]);

    // int elements_per_process = (N_theoretical + n_proc - 1) / n_proc;
    // int N = elements_per_process * n_proc;

    // double *local_values = (double *)malloc(sizeof(double) * elements_per_process);

    // if (rank == 0)
    // {
    //     cout << "N = " << N_theoretical << ", elements_per_process = " << elements_per_process << endl;
    //     double *full_values = (double *)malloc(sizeof(double) * N);
    //     double start_time, end_time;
    //     fill_in_array_with_random_numbers(full_values, N_theoretical);
    //     start_time = MPI_Wtime();
    //     for (int i = N_theoretical; i < N; ++i)
    //     {
    //         if (!reverse)
    //         {
    //             full_values[i] = std::numeric_limits<double>::max();
    //         }
    //         else
    //         {
    //             full_values[i] = std::numeric_limits<double>::min();
    //         }
    //     }
    //     MPI_Scatter(full_values, elements_per_process, MPI_DOUBLE,
    //                 local_values, elements_per_process, MPI_DOUBLE,
    //                 0, MPI_COMM_WORLD);
    //     // 1. sort its own part
    //     if (reverse)
    //     {

    //         std::sort(local_values, local_values + elements_per_process, [](const double &a, const double &b)
    //                   { return a > b; });
    //     }
    //     else
    //     {
    //         std::sort(local_values, local_values + elements_per_process);
    //     }
    //     // 2. participate in sorting network
    //     participate_in_sorting_network(local_values, elements_per_process, comparators, reverse);

    //     MPI_Gather(local_values, elements_per_process, MPI_DOUBLE,
    //                full_values, elements_per_process, MPI_DOUBLE,
    //                0, MPI_COMM_WORLD);
    //     end_time = MPI_Wtime();
    //     cout << "time elapsed: " << end_time - start_time << endl;

    //     for (int i = 1; i < N; ++i)
    //     {
    //         cout << full_values[i - 1] << " ";
    //         if (!reverse && (full_values[i - 1] > full_values[i]) ||
    //             reverse && (full_values[i - 1] < full_values[i]))
    //         {
    //             throw std::runtime_error("Sorting is incorrect!");
    //         }
    //     }
    //     cout << endl;
    //     free(full_values);
    // }
    // else
    // {
    //     MPI_Scatter(nullptr, elements_per_process, MPI_DOUBLE,
    //                 local_values, elements_per_process, MPI_DOUBLE,
    //                 0, MPI_COMM_WORLD);

    //     // 1. sort its own part
    //     if (reverse)
    //     {

    //         std::sort(local_values, local_values + elements_per_process, [](const double &a, const double &b)
    //                   { return a > b; });
    //     }
    //     else
    //     {
    //         std::sort(local_values, local_values + elements_per_process);
    //     }
    //     // 2. participate in sorting network
    //     participate_in_sorting_network(local_values, elements_per_process, comparators, reverse);

    //     MPI_Gather(local_values, elements_per_process, MPI_DOUBLE,
    //                nullptr, elements_per_process, MPI_DOUBLE,
    //                0, MPI_COMM_WORLD);
    // }

    // std::cout << "rank " << rank << "my values are: ";
    // for (auto const &elem : local_values)
    // {
    //     cout << elem << " ";
    // }
    // cout << endl;
    // free(local_values);
    MPI_Finalize();
    return 0;
}
