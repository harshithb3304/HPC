#include<stdio.h>
#include<stdlib.h>
#include<time.h>

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

    int thread_counts[] = {1,2,4,6,8,10,12,16,20,32,64};
    int num_options = sizeof(thread_counts) / sizeof(thread_counts[0]);
    double start,end; 

    FILE *f_dot = fopen("dot_product.txt", "w");    

    printf("Vector Dot Product\n");
    for(int t = 0; t < num_options; t++){
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);

        double dot_product = 0.0;
        start = omp_get_wtime();

        #pragma omp parallel 
        {
            double local_sum = 0.0;
            #pragma omp for
            for(int i = 0; i < count; i++){
                local_sum += arr1[i] * arr2[i];
            }

            #pragma omp critical
            dot_product += local_sum;
        }

        end = omp_get_wtime();
        printf("%d\t%.6f\t%.12f\n", num_threads, end - start, dot_product);
        fprintf(f_dot, "%d %lf\n", num_threads, end - start);
    }

    free(arr1);
    free(arr2);
    return 0;
}