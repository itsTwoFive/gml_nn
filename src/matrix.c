#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

matrix* mat_alloc(int rows, int cols){
    matrix *mat = malloc(sizeof(matrix));
    
    if (mat==NULL){
        perror("ERROR: Matrix malloc failed");
        exit(EXIT_FAILURE);
    }

    mat->cols = cols;
    mat->rows = rows; 
    // printf("%i\n",mat->cols);

    mat->d = malloc(sizeof(mat->d)*rows*cols);
    mat_set_zeros(*mat);
    return mat;
};

void mat_free(matrix* mat){
    free(mat->d);
    free(mat);
}

double* mat_seek(matrix m,int i,int j){
    return m.d +i*m.cols + j;
}
void mat_print(matrix m){
    printf("[%u, %u]\n",m.rows,m.cols);
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            printf("%f ",*mat_seek(m,i,j));
        }
        printf("\n");
    }
}
void mat_set_number(matrix m, int i, int j, double value){
    *mat_seek(m,i,j) = value;
}

void mat_randf(matrix m){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            mat_set_number(m,i,j,rand()/(double) RAND_MAX * (MAX_R - MIN_R) + MIN_R);
        }
    }
}

void notify_mat_error(char *op){
    printf("Error al realizar la operacion matricial %s.\n",op);
    exit(1);
}

void mat_sumf(matrix m1, matrix m2, matrix *result){
    if (m1.cols != m2.cols || m1.rows != m2.rows){
        notify_mat_error("suma");
    }
    else{
        for (int i = 0; i < m1.rows; i++){
            for (int j = 0; j < m1.cols; j++)
            {
                *mat_seek(*result,i,j) = *mat_seek(m1,i,j) + *mat_seek(m2,i,j);
            }
        }
    }
}

void mat_subsf(matrix m1, matrix m2, matrix *result){
    if (m1.cols != m2.cols || m1.rows != m2.rows){
        notify_mat_error("resta");
    }
    else{
        for (int i = 0; i < m1.rows; i++){
            for (int j = 0; j < m1.cols; j++)
            {
                *mat_seek(*result,i,j) = *mat_seek(m1,i,j) - *mat_seek(m2,i,j);
            }
        }
    }
}

void mat_productf(matrix m1, matrix m2, matrix *result){
    if (m1.cols != m2.rows){
        notify_mat_error("producto");
    }
    else{
        for (int i = 0; i < result->rows; i++){
            for (int j = 0; j < result->cols; j++){
                *mat_seek(*result,i,j) = 0.0f;
                for (int k = 0; k < m1.cols; k++){
                    *mat_seek(*result,i,j) += *mat_seek(m1,i,k) * *mat_seek(m2,k,j);
                }
            }
        }
    }
}

void mat_dot_productf(matrix m, double coeficent, matrix *result){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++)
        {
            *mat_seek(*result,i,j) = *mat_seek(m,i,j) * coeficent;
        }
    }
}

double mat_column_sum(matrix m, int col){
    double total_sum = 0;
    for (int i = 0; i < m.cols; i++)
    {
        total_sum += *mat_seek(m,i,col);
    }
    return total_sum;
}

void mat_set_zeros(matrix m){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            mat_set_number(m,i,j,0);
        }
    }
}

matrix* mat_fromarray(int length,double array[]){
    matrix* mat = mat_alloc(1,length);
    for (int j = 0; j < mat->cols; j++)
    {
        mat_set_number(*mat,0,j,array[j]);
    }
    return mat;
}