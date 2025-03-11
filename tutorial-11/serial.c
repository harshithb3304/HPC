#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000

double matrix1[N][N], matrix2[N][N], result[N][N];

void read_matrix_from_file(const char *filename, double matrix[N][N]) {
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

    if (total_elements != N * N) {
        printf("Warning: Expected %d elements but found %d in the file.\n", N * N, total_elements);
    }
}

void add_matrices(double matrix1[N][N], double matrix2[N][N], double result[N][N], FILE *file) {
    clock_t start, end;
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
            // fprintf(file, "%lf\n", result[i][j]);
        }
    }

    end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %f seconds\n", time_spent);
}

int main() {
    read_matrix_from_file("output.txt", matrix1);
    read_matrix_from_file("output2.txt", matrix2);

    FILE *fp = fopen("serial_result.txt", "w");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    add_matrices(matrix1, matrix2, result, fp);
    
    fclose(fp);
    return 0;
}
