#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(int argc, char const *argv[])
{
    int num_inputs = 3;
    int num_neurons = 4;

    /* Se crean las matrices X, W, B y A */
    matrix *X = mat_alloc(1,num_inputs);
    matrix *W = mat_alloc(num_inputs,num_neurons);
    matrix *B = mat_alloc(1,num_neurons);
    matrix *A = mat_alloc(1,num_neurons);

    /* Se le agregan valores a X */
    *mat_seek(*X,0,0) = 1.5;
    *mat_seek(*X,0,1) = 1.0;
    *mat_seek(*X,0,2) = 2.25;

    /* Se inicializan los valores de W y B de forma aleatoria */
    mat_randf(*W);
    mat_randf(*B);

    /* Realizamos la multiplicacion X * W */
    matrix * temp = mat_alloc(1,num_neurons);
    mat_productf(*X,*W,temp);

    /* Sumamos el resultado de la multiplicacion con B */
    mat_sumf(*temp,*B,A);

    /* Liberamos la matriz temporal */
    mat_free(temp);

    /* Imprimimos el resultado */
    mat_print(*A);

    /* Liberamos las demas matrices */
    mat_free(X);
    mat_free(W);
    mat_free(B);
    mat_free(A);

    return 0;
}
