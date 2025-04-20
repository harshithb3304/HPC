#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Function to read matrix dimensions from a file
// Returns 1 if successful, 0 if failed
int read_matrix_dimensions(const char *filename, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open input file");
        return 0;
    }
    
    // Count rows and columns
    *rows = 0;
    *cols = 0;
    char line[4096];
    
    // Read first line to determine number of columns
    if (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, " \t\n");
        while (token) {
            (*cols)++;
            token = strtok(NULL, " \t\n");
        }
        (*rows)++;
    }
    
    // Count remaining rows
    while (fgets(line, sizeof(line), file)) {
        (*rows)++;
    }
    
    fclose(file);
    return 1;
}

// Function to read a matrix from a file
// Returns 1 if successful, 0 if failed
int read_matrix(const char *filename, double **matrix, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open input file");
        return 0;
    }
    
    // Allocate memory for the matrix
    *matrix = (double *)malloc(rows * cols * sizeof(double));
    if (!(*matrix)) {
        perror("Memory allocation failed");
        fclose(file);
        return 0;
    }
    
    // Read matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &((*matrix)[i * cols + j])) != 1) {
                fprintf(stderr, "Error reading matrix element at (%d,%d)\n", i, j);
                free(*matrix);
                fclose(file);
                return 0;
            }
        }
    }
    
    fclose(file);
    return 1;
}

// Function to print a matrix
void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int A_rows, A_cols, B_rows, B_cols;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_C = NULL;
    int local_rows;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double start_time = MPI_Wtime();
    
    // Process 0 reads the input matrices
    if (rank == 0) {
        // Read matrix A dimensions and data
        if (!read_matrix_dimensions("/home/harshith/Projects /SEM 6/HPC/tutorial-16/output1.txt", &A_rows, &A_cols)) {
            fprintf(stderr, "Error reading matrix A dimensions\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (!read_matrix("/home/harshith/Projects /SEM 6/HPC/tutorial-16/output1.txt", &A, A_rows, A_cols)) {
            fprintf(stderr, "Error reading matrix A\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read matrix B dimensions and data
        if (!read_matrix_dimensions("/home/harshith/Projects /SEM 6/HPC/tutorial-16/output2.txt", &B_rows, &B_cols)) {
            fprintf(stderr, "Error reading matrix B dimensions\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (!read_matrix("/home/harshith/Projects /SEM 6/HPC/tutorial-16/output2.txt", &B, B_rows, B_cols)) {
            fprintf(stderr, "Error reading matrix B\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Check if matrices can be multiplied
        if (A_cols != B_rows) {
            fprintf(stderr, "Error: Matrix dimensions don't match for multiplication. A: %dx%d, B: %dx%d\n", 
                    A_rows, A_cols, B_rows, B_cols);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("Matrix A: %d x %d\n", A_rows, A_cols);
        printf("Matrix B: %d x %d\n", B_rows, B_cols);
    }
    
    // Broadcast matrix dimensions
    MPI_Bcast(&A_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate local number of rows for matrix A
    local_rows = A_rows / size;
    if (rank < A_rows % size) {
        local_rows++;
    }
    
    // Allocate memory for local matrices
    local_A = (double *)malloc(local_rows * A_cols * sizeof(double));
    local_C = (double *)malloc(local_rows * B_cols * sizeof(double));
    
    // Broadcast matrix B to all processes
    if (rank != 0) {
        B = (double *)malloc(B_rows * B_cols * sizeof(double));
    }
    MPI_Bcast(B, B_rows * B_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Distribute matrix A among processes
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_proc = A_rows / size;
            if (i < A_rows % size) {
                rows_for_proc++;
            }
            
            sendcounts[i] = rows_for_proc * A_cols;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // Scatter matrix A
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, local_rows * A_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            local_C[i * B_cols + j] = 0.0;
            for (int k = 0; k < A_cols; k++) {
                local_C[i * B_cols + j] += local_A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
    
    // Gather results back to process 0
    if (rank == 0) {
        C = (double *)malloc(A_rows * B_cols * sizeof(double));
        
        // Update sendcounts and displs for the result matrix C
        for (int i = 0; i < size; i++) {
            int rows_for_proc = A_rows / size;
            if (i < A_rows % size) {
                rows_for_proc++;
            }
            
            sendcounts[i] = rows_for_proc * B_cols;
            displs[i] = i == 0 ? 0 : displs[i-1] + sendcounts[i-1];
        }
    }
    
    MPI_Gatherv(local_C, local_rows * B_cols, MPI_DOUBLE, C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    
    // Output results
    if (rank == 0) {
        printf("Matrix multiplication completed in %.6f seconds with %d processes\n", execution_time, size);
        
        // Write results to file for benchmark script
        FILE *output_file = fopen("clustering_results_mpi.txt", "w");
        if (output_file) {
            fprintf(output_file, "# Matrix Multiplication Results\n");
            fprintf(output_file, "# Matrix A: %d x %d\n", A_rows, A_cols);
            fprintf(output_file, "# Matrix B: %d x %d\n", B_rows, B_cols);
            fprintf(output_file, "# Result Matrix C: %d x %d\n\n", A_rows, B_cols);
            
            // Write a few elements of the result matrix (if it's large)
            int max_print = 5;
            fprintf(output_file, "# Sample of result matrix (up to %d x %d):\n", max_print, max_print);
            for (int i = 0; i < (A_rows < max_print ? A_rows : max_print); i++) {
                for (int j = 0; j < (B_cols < max_print ? B_cols : max_print); j++) {
                    fprintf(output_file, "%.4f ", C[i * B_cols + j]);
                }
                fprintf(output_file, "\n");
            }
            
            // Write the benchmark line (for the benchmark script)
            fprintf(output_file, "%d %.6f %d\n", size, execution_time, A_rows * B_cols);
            
            fclose(output_file);
        }
        
        // Clean up
        free(A);
        free(C);
        free(sendcounts);
        free(displs);
    }
    
    // Clean up
    free(local_A);
    free(local_C);
    free(B);
    
    MPI_Finalize();
    return 0;
}