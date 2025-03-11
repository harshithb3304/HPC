#include <stdio.h>
#include <stdlib.h>
#include<time.h> 

int main() {
    FILE *fp1 = fopen("output1.txt", "r");
    FILE *fp2 = fopen("output2.txt", "r");
    if (fp1 == NULL || fp2 == NULL) {
        printf("Error opening files\n");
        exit(1);
    }
    
    int count = 0;
    double temp, dot_product = 0.0;
    while (fscanf(fp1, "%lf", &temp) != EOF) {
        count++;
    }
    
    double *h_arr1 = (double *)malloc(count * sizeof(double));
    double *h_arr2 = (double *)malloc(count * sizeof(double));
    if (h_arr1 == NULL || h_arr2 == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    rewind(fp1);
    rewind(fp2);
    
    
    for (int i = 0; i < count; i++) {
        fscanf(fp1, "%lf", &h_arr1[i]);
        fscanf(fp2, "%lf", &h_arr2[i]);
    }
    fclose(fp1);
    fclose(fp2);
    
    clock_t start = clock();
    for (int i = 0; i < count; i++) {
        dot_product += h_arr1[i] * h_arr2[i];
    }
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Dot Product: %.2lf\n", dot_product);
    printf("Time taken: %lf\n", time_taken);
    free(h_arr1);
    free(h_arr2);
    
    return 0;
}
