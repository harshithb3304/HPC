#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void read_vector_from_file(const char *filename, double **vec, int *size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open input file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double temp;
    *size = 0;

    // Count elements
    while (fscanf(file, "%lf", &temp) == 1)
        (*size)++;

    rewind(file);

    // Allocate and read
    *vec = (double *)malloc((*size) * sizeof(double));
    for (int i = 0; i < *size; i++)
        fscanf(file, "%lf", &((*vec)[i]));

    fclose(file);
}

int main(int argc, char **argv)
{
    int rank, size, global_n;
    double *global_x = NULL, *global_y = NULL;
    double *local_x, *local_y, *local_sum, *local_product;
    int *recvcounts = NULL, *displs = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start_all = MPI_Wtime();

    if (rank == 0)
    {
        read_vector_from_file("output1.txt", &global_x, &global_n);
        int y_size;
        read_vector_from_file("output2.txt", &global_y, &y_size);

        if (global_n != y_size)
        {
            fprintf(stderr, "Mismatched vector sizes\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int sum = 0;
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = global_n / size + (i < global_n % size);
            displs[i] = sum;
            sum += recvcounts[i];
        }
    }

    MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int local_n = global_n / size + (rank < global_n % size ? 1 : 0);

    local_x = malloc(local_n * sizeof(double));
    local_y = malloc(local_n * sizeof(double));
    local_sum = malloc(local_n * sizeof(double));
    local_product = malloc(local_n * sizeof(double));

    MPI_Scatterv(global_x, recvcounts, displs, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_y, recvcounts, displs, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Addition
    double t_start_add = MPI_Wtime();
    double local_sum_total = 0.0;
    for (int i = 0; i < local_n; i++)
    {
        local_sum[i] = local_x[i] + local_y[i];
        local_sum_total += local_sum[i];
    }
    double t_end_add = MPI_Wtime();

    // Multiplication
    double t_start_mult = MPI_Wtime();
    double local_product_total = 0.0;
    for (int i = 0; i < local_n; i++)
    {
        local_product[i] = local_x[i] * local_y[i];
        local_product_total += local_product[i];
    }
    double t_end_mult = MPI_Wtime();

    double global_sum = 0.0, global_product = 0.0;
    MPI_Reduce(&local_sum_total, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_product_total, &global_product, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double add_time = t_end_add - t_start_add;
        double mult_time = t_end_mult - t_start_mult;

        printf("Processes: %d\n", size);
        printf("  ➤ Addition    | Time: %.6f sec | Sum: %.2f\n", add_time, global_sum);
        printf("  ➤ Multiplica. | Time: %.6f sec | Product Sum: %.2f\n", mult_time, global_product);

        FILE *fadd = fopen("benchmark_add_result.txt", "a");
        FILE *fmult = fopen("benchmark_mult_result.txt", "a");

        if (fadd)
        {
            fprintf(fadd, "%d %.6f %.2f\n", size, add_time, global_sum);
            fclose(fadd);
        }
        if (fmult)
        {
            fprintf(fmult, "%d %.6f %.2f\n", size, mult_time, global_product);
            fclose(fmult);
        }

        free(global_x);
        free(global_y);
        free(recvcounts);
        free(displs);
    }

    free(local_x);
    free(local_y);
    free(local_sum);
    free(local_product);

    MPI_Finalize();
    return 0;
}
