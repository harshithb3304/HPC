#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(){
    FILE *fp = fopen("output.txt", "r");
    if(fp == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    int count = 0;
    double temp;
    while(fscanf(fp, "%lf", &temp) != EOF){
        count++;
    }

    double *arr = (double *)malloc(count * sizeof(double));
    if(arr == NULL){
        printf("Memory allocation failed\n");
        exit(1);
    }
    rewind(fp);
    for(int i=0; i<count; i++){
        fscanf(fp, "%lf", &arr[i]);
    }
    fclose(fp);

    FILE *f_reduc = fopen("reduction.txt", "w");
    FILE *f_crit = fopen("critical.txt", "w");
    if(f_reduc == NULL || f_crit == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_options = sizeof(thread_counts) / sizeof(thread_counts[0]);

    double start,end;
    double sum_reduction, sum_critical;

    printf("Parallel Reduction Sum\n");
    for (int i=0; i<num_options; i++){
        int T = thread_counts[i];
        omp_set_num_threads(T); //Set the number of threads (1,2,4,8,16,32,64,128)
        sum_reduction = 0.0;
        start = omp_get_wtime(); //Wall Clock Time

        #pragma omp parallel for reduction(+:sum_reduction)
        for(int j=0; j<count; j++){
            sum_reduction += arr[j];
        }

        end = omp_get_wtime();
        double time_taken = end - start;
        printf("Threads: %d, Sum: %lf, Time: %lf\n", T, sum_reduction, time_taken);
        fprintf(f_reduc, "%d %lf %lf\n", T, sum_reduction, time_taken);
    }

    printf("Parallel Critical Sum\n");
    for (int i=0; i<num_options; i++){
        int T = thread_counts[i];
        omp_set_num_threads(T); 
        sum_critical = 0.0;
        start = omp_get_wtime();

        #pragma omp parallel for
        for(int j=0;j<count;j++){
            #pragma omp critical
            sum_critical += arr[j];
        }

        end = omp_get_wtime();
        double time_taken = end - start;
        printf("Threads: %d, Sum: %lf, Time: %lf\n", T, sum_critical, time_taken);
        fprintf(f_crit, "%d %lf %lf\n", T, sum_critical, time_taken);
    }

    fclose(f_reduc);
    fclose(f_crit);
    free(arr);
    
    return 0;
}