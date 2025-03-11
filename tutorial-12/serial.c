#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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


void multiply_matrices_serial(double matrix1[N][N], double matrix2[N][N], double result[N][N], FILE *file) {
    clock_t start, end;
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
            // Print only for every 1000th column
            if (j % 1000 == 0) {
                printf("result[%d][%d]: %lf\n", i, j, result[i][j]);
            }
        }
    }

    end = clock();
    double time_spent =(double) end - start / CLOCKS_PER_SEC;
    printf("Serial Time: %f seconds\n", time_spent);
    fprintf(file, "Serial Time: %f\n", time_spent);
}


// void multiply_matrices_parallel(double matrix1[N][N], double matrix2[N][N], double result[N][N], FILE *file) {
//     double start = omp_get_wtime();

//     #pragma omp parallel for num_threads(8) collapse(2)
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             result[i][j] = 0;
//             for (int k = 0; k < N; k++) {
//                 result[i][j] += matrix1[i][k] * matrix2[k][j];
//             }
//         }
//     }

//     double end = omp_get_wtime();
//     double time_spent = end - start;
//     printf("Parallel Time: %f seconds\n", time_spent);
//     fprintf(file, "Parallel Time: %f\n", time_spent);
// }

void print_actual_matrix_info(int rows, int cols, const char *matrix_name) {
    printf("%s: Actual Dimensions: %dx%d\n", matrix_name, rows, cols);
}

int main() {
    int rows1 = 0, cols1 = 0;
    int rows2 = 0, cols2 = 0;

    read_matrix_from_file("output.txt", matrix1, N, N);
    read_matrix_from_file("output2.txt", matrix2, N, N);


    FILE *fp = fopen("matrix_multiply_results_serial.txt", "w");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    multiply_matrices_serial(matrix1, matrix2, result, fp);
    //multiply_matrices_parallel(matrix1, matrix2, result, fp);

    fclose(fp);
    return 0;
}
