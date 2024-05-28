#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(int argc, char const *argv[])
{
    /* Se crean 2 matrices de tamano 2x5 y 5x3 */
    matrix * mat1 = mat_alloc(2,5);
    matrix * mat2 = mat_alloc(5,3);

    /* Establecemos valores aleatorios ara todas la entradas de ambas matrices */
    srand(25);
    mat_randf(*mat1);
    mat_randf(*mat2);

    /* Multiplicamos los mat1 y mat2 */
    matrix * mat3 = mat_alloc(2,3);
    mat_productf(*mat1,*mat2,mat3);

    /* Imprimimos valores de ambas matrices y el resultado por consola */
    mat_print(*mat1);
    mat_print(*mat2);
    mat_print(*mat3);

    /* Realizamos una multipplicacion escalar por 10 */
    mat_dot_productf(*mat3,10,mat3);
    
    /* Volvemos a imprimir el resultado para comprobar su funcionamiento */
    mat_print(*mat3);

    /* Liberamos la memoria alocada a las tres matrices */
    mat_free(mat1);
    mat_free(mat2);
    mat_free(mat3);

    return 0;
}
