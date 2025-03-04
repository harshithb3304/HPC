#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 10000

double matrix1[N][N], matrix2[N][N], result[N][N];

void read_matrix_from_file(const char *filename, double matrix[N][N]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}


void multiply_matrices_parallel(double matrix1[N][N], double matrix2[N][N], double result[N][N], int thread_count, FILE *file) {
    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) num_threads(thread_count)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
            if (j % 1000 == 0) {
                printf("result[%d][%d] = %f, Threads: %d\n", i, j, result[i][j], thread_count);
            }
        }
    }

    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("Threads: %d, Parallel Time: %f seconds\n", thread_count, time_spent);
    fprintf(file, "%d %f\n", thread_count, time_spent);
}

int main() {
    read_matrix_from_file("output.txt", matrix1);
    read_matrix_from_file("output2.txt", matrix2);

    FILE *fp = fopen("matrix_multiply_results_parallel.txt", "w");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int thread_counts[] = {2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    int num_options = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    for (int t = 0; t < num_options; t++) {
        multiply_matrices_parallel(matrix1, matrix2, result, thread_counts[t], fp);
    }


    fclose(fp);
    return 0;
}
