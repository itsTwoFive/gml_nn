#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(int argc, char const *argv[])
{
    /* Se crean 2 matrices de tamano 4x3 */
    matrix * mat1 = mat_alloc(4,3);
    matrix * mat2 = mat_alloc(4,3);

    /* Establecemos valores aleatorios ara todas la entradas de la matriz1 */
    srand(25);
    mat_randf(*mat1);

    /* La sengunda matriz la transformamos sus valores a todo unos */
    for (int i = 0; i < mat2->rows; i++)
    {
        for (int j = 0; j < mat2->cols; j++)
        {
            mat_set_number(*mat2,i,j,1.0);
        }
    }

    /* Sumamos los mat1 y mat2 */
    matrix * mat3 = mat_alloc(4,3);
    mat_sumf(*mat1,*mat2,mat3);

    /* Imprimimos valores de ambas matrices y el resultado por consola */
    mat_print(*mat1);
    mat_print(*mat2);
    mat_print(*mat3);

    /* Liberamos la memoria alocada a las tres matrices */
    mat_free(mat1);
    mat_free(mat2);
    mat_free(mat3);

    return 0;
}
