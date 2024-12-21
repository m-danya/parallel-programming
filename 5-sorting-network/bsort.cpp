#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

using std::cin;
using std::cout;
using std::endl;

typedef std::vector<std::pair<int, int>> Comparators_vector;

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

int count_tacts_number(Comparators_vector const &c)
{
    std::vector<int> tacts_number(c.size());
    for (auto comparator : c)
    {
        tacts_number[comparator.first] += 1;
        tacts_number[comparator.second] += 1;
        tacts_number[comparator.first] = std::max(
            tacts_number[comparator.first],
            tacts_number[comparator.second]);
        tacts_number[comparator.second] = std::max(
            tacts_number[comparator.first],
            tacts_number[comparator.second]);
    }
    int max_tacts_number = 0;
    for (auto t : tacts_number)
    {
        max_tacts_number = std::max(max_tacts_number, t);
    }
    return max_tacts_number;
}

void test_sorting(int n)
{
    Comparators_vector comparators;
    odd_even_sort(0, 1, n, comparators);
    std::vector<int> zeros_and_ones(n, 0), working_copy(n, 0);

    while (true)
    {
        working_copy = zeros_and_ones;

        for (auto const &c : comparators)
        {
            if (working_copy[c.first] > working_copy[c.second])
            {
                std::swap(working_copy[c.first], working_copy[c.second]);
            }
        }
        int m = 0;
        for (auto const &e : working_copy)
        {
            if (m > e)
            {
                throw std::runtime_error("Sorting is incorrect");
            }
            if (m < e)
            {
                m = e;
            }
        }
        zeros_and_ones[0] += 1;
        int i = 0;

        while (zeros_and_ones[i] > 1)
        {
            zeros_and_ones[i] -= 2;
            i++;
            if (i == n)
            {
                break;
            }
            zeros_and_ones[i] += 1;
        }
        if (i == n)
        {
            break;
        }
    }
}

int main(int argc, char **argv)
{
    int n = 6;
    if (argc >= 2)
    {
        n = std::stoi(argv[1]);
    }
    else
    {
        cout << "usage: ./bsort n (use -1 for running tests)" << endl
             << endl
             << "using " << n << " as a default value for n" << endl
             << endl;
    }
    if (n > 0)
    {
        Comparators_vector comparators;
        odd_even_sort(0, 1, n, comparators);
        cout << n << " 0 0" << endl;
        for (auto comp : comparators)
        {
            cout << comp.first << " " << comp.second << endl;
        }
        cout << comparators.size() << endl
             << count_tacts_number(comparators) << endl;
    }
    else
    {
        for (int n = 1; n <= 24; ++n)
        {
            test_sorting(n); // can throw an exception
            cout << "n = " << n << ": PASSED" << endl;
        }
    }

    return 0;
}
