#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

int main(int argc, char *argv[])
{
    int myid, numprocs;
    double start_time, end_time;
    int name_len;
    double *arr = NULL;
    int i, N = 0;
    double sum = 0.0, part_sum = 0.0;
    int s = 0, s0 = 0;

    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &name_len);

    if (myid == MASTER)
    {
        FILE *fptr = fopen("output.txt", "r");
        if (!fptr)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // First pass: count the number of doubles
        double temp;
        while (fscanf(fptr, "%lf", &temp) == 1)
        {
            N++;
        }

        rewind(fptr); // Move file pointer back to start
        arr = (double *)malloc(N * sizeof(double));
        for (i = 0; i < N; i++)
        {
            fscanf(fptr, "%lf", &arr[i]);
        }

        fclose(fptr);

        s = N / numprocs;
        s0 = s + (N % numprocs);
    }

    // Broadcast the value of N, s, s0 to all processes
    MPI_Bcast(&N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&s, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&s0, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    int my_count = (myid == MASTER) ? s0 : s;
    double *my_data = (double *)malloc(my_count * sizeof(double));

    if (myid == MASTER)
    {
        for (i = 1; i < numprocs; i++)
        {
            MPI_Send(&arr[s0 + (i - 1) * s], s, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        for (i = 0; i < s0; i++)
        {
            my_data[i] = arr[i];
        }
    }
    else
    {
        MPI_Recv(my_data, my_count, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (myid == MASTER)
    {
        start_time = MPI_Wtime();
    }

    for (i = 0; i < my_count; i++)
    {
        part_sum += my_data[i];
    }

    MPI_Reduce(&part_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (myid == MASTER)
    {
        end_time = MPI_Wtime();
        printf("Total sum = %.2f\n", sum);
        printf("Time taken = %.6f seconds\n", end_time - start_time);
        free(arr);
    }

    free(my_data);
    MPI_Finalize();
    return 0;
}
