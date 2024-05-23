#include <stdio.h>
#include <stdlib.h>
#include "data_handler.h"

int main(int argc, char const *argv[])
{
    /* Establecemos el nombre de fichero a leer */
    char filename[] = "FicheroDePrueba.fakecsv";

    /* Obtenemos el numero de casos y atributos */
    int num_casos = get_number_cases(filename);
    int num_atrib = get_number_atributes(filename);

    /* Los mostramos por consola */
    printf("El fichero %s tiene %i casos con %i atributos\n"
        ,filename
        ,num_casos
        ,num_atrib);

    return 0;
}
