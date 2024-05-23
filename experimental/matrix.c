// This matrix.h code version differs from the finals version code founded in src

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

matrix mat_alloc(int rows, int cols){
    matrix mat;
    mat.cols = cols;
    mat.rows = rows;

    mat.d = malloc(sizeof(*mat.d)*rows*cols);
    mat_set_zeros(mat);
    return mat;
};
void mat_free(matrix* mat){
    free(mat->d);
}

float* mat_seek(matrix m,int i,int j){
    return m.d +i*m.cols + j;
}
void mat_print(matrix m){
    printf("%u, %u\n",m.rows,m.cols);
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            printf("%f ",*mat_seek(m,i,j));
        }
        printf("\n");
    }
}
void mat_set_number(matrix m, int i, int j, float value){
    *mat_seek(m,i,j) = value;
}

void mat_randf(matrix m){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            mat_set_number(m,i,j,rand()/(float) RAND_MAX * (MAX_R - MIN_R) + MIN_R);
        }
    }
}

void notify_mat_error(char *op){
    printf("Error al realizar la operacion matricial %s.\n",op);
}

void mat_sumf(matrix m1, matrix m2, matrix *result){
    if (m1.cols != m2.cols || m1.rows != m2.rows){
        notify_mat_error("suma");
    }
    else{
        *result = mat_alloc(m1.rows,m1.cols);
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
        *result = mat_alloc(m1.rows,m1.cols);
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
        *result = mat_alloc(m1.rows,m2.cols);
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

void mat_dot_productf(matrix m, float coeficent, matrix *result){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++)
        {
            *mat_seek(*result,i,j) = *mat_seek(m,i,j) * coeficent;
        }
    }
}

float mat_column_sum(matrix m, int col){
    float total_sum = 0;
    for (int i = 0; i < m.rows; i++)
    {
        total_sum += *mat_seek(m,i,col);
    }
    return total_sum;
}

float mat_row_sum(matrix m, int row){
    float total_sum = 0;
    for (int j = 0; j < m.cols; j++)
    {
        total_sum += *mat_seek(m,row,j);
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
