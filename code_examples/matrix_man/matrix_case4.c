#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

int main(int argc, char const *argv[])
{
    /* Establecemos el numero de valores a sumar*/
    int num_values = 100;

    /* Creamos un vector vertical de tamano 100 */
    matrix * mat = mat_alloc(num_values,1);

    /* Establecemos sus valores a valores aleatorios */
    time_t t;
    time(&t);
    srand((unsigned int) t);

    mat_randf(*mat);

    /* Visualizamos en consola los 100 valores */
    mat_print(*mat);

    /* Realizamos la suma de toda la columna */
    double sum = mat_column_sum(*mat,0);

    /* Dividimos entre el numero de valores para calcular la media*/
    double mean = sum/num_values;
    printf("Media = %f\n",mean);

    /* Liberamos la matriz */
    mat_free(mat);

    return 0;
}
