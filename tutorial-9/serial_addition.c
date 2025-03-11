#include<stdio.h>
#include<stdlib.h>
#include<time.h>

struct vectors{
    double *arr1;
    double *arr2;
    double *sum_result;
    double *product_result;
};

int main(){
    FILE *fp1 = fopen("output1.txt","r");
    if(fp1 == NULL){
        printf("Error opening file 1\n");
        exit(1);
    }

    FILE *fp2 = fopen("output2.txt","r");
    if(fp2 == NULL){
        printf("Error opening file 2\n");
        exit(1);
    }

    int count = 0;
    double temp; 
    while(fscanf(fp1, "%lf", &temp) != EOF){
        count++;
    }

    double *arr1 = (double *)malloc(count * sizeof(double));
    double *arr2 = (double *)malloc(count * sizeof(double));
    if(arr1 == NULL || arr2 == NULL){
        printf("Memory allocation failed\n");
        exit(1);
    }
    rewind(fp1);
    rewind(fp2);
    for(int i=0; i<count; i++){
        fscanf(fp1, "%lf", &arr1[i]);
        fscanf(fp2, "%lf", &arr2[i]);
    }
    fclose(fp1);
    fclose(fp2);

    FILE *f_add = fopen("addition.txt", "w");
    FILE *f_mul = fopen("multiplication.txt", "w"); 
    if(f_add == NULL || f_mul == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    struct vectors vec;
    vec.arr1 = arr1;
    vec.arr2 = arr2;
    vec.sum_result = (double *)malloc(count * sizeof(double));
    vec.product_result = (double *)malloc(count * sizeof(double));
    if(vec.sum_result == NULL || vec.product_result == NULL){
        printf("Memory allocation failed\n");
        exit(1);
    }

    //int thread_counts[] = {1, 2, 4,6,8,10,12 ,16,20, 32, 64};
    //int num_options = sizeof(thread_counts) / sizeof(thread_counts[0]);
    //double start, end;

    printf("Parallel Vector Addition\n");    
    int j;
    clock_t start = clock();
    
    for(j=0;j<count;j++){
        vec.sum_result[j] = vec.arr1[j] + vec.arr2[j];
    }
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %lf\n", time_taken);    
    
    fclose(f_add);
    fclose(f_mul);
    free(arr1);
    free(arr2);
    free(vec.sum_result);


    return 0;
}