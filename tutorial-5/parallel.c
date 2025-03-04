#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

double matrix1[N][N], matrix2[N][N], result[N][N];

void read_matrix_from_file(const char *filename, double matrix[N][N], int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int total_elements = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) == 1) {
                total_elements++;
            } else {
                perror("Error reading file or insufficient data");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);

    rows = N;
    cols = N;

    // Additional validation
    if (total_elements != N * N) {
        printf("Warning: Expected %d elements but found %d in the file.\n", N * N, total_elements);
    }
}


void add_matrices_parallel(double matrix1[N][N], double matrix2[N][N], double result[N][N], int num_threads, FILE *file) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for collapse(2) num_threads(num_threads)   
    //Here collapse(2) means that the two loops are collapsed into one loop
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    
    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("Threads: %d, Time: %f seconds\n", num_threads, time_spent);
    fprintf(file, "%d %f\n", num_threads, time_spent);
}

int main() {
    read_matrix_from_file("output.txt", matrix1,N,N);
    read_matrix_from_file("output2.txt", matrix2,N,N);
    
    FILE *fp = fopen("matrix_add_parallel.txt", "w");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    int num_options = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    for (int t = 0; t < num_options; t++) {
        add_matrices_parallel(matrix1, matrix2, result, thread_counts[t], fp);
    }
    
    fclose(fp);
    return 0;
}
