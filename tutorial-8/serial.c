#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    
    FILE *fp = fopen("output.txt", "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    
    int count = 0;
    double temp, sum = 0.0;
    while (fscanf(fp, "%lf", &temp) != EOF) {
        count++;
    }
    
    double *arr = (double *)malloc(count * sizeof(double));
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    rewind(fp);
    clock_t start = clock();
    for (int i = 0; i < count; i++) {
        fscanf(fp, "%lf", &arr[i]);
        sum += arr[i];
    }
    clock_t end = clock();
    fclose(fp);
    free(arr);
    
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Sum: %.2lf\n", sum);
    printf("Time taken: %.6f seconds\n", time_taken);
    
    return 0;
}
