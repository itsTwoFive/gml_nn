#include <stdio.h>
#include <stdlib.h>
#include "data_handler.h"

int main(int argc, char const *argv[])
{
    /* Establecemos el nombre de fichero a leer */
    char filename[] = "FicheroDePrueba.fakecsv";

    /* Leemos el archivo y guardamos los datos contenidos en pr */
    int num_inputs = get_number_atributes(filename)-1;
    parser_result pr = parse_data(filename,num_inputs);

    /* Dividimos los datos en dos set, de entrenamiento y prueba */
    parser_result* sets = data_div(pr,3);
    parser_result set_entrenamiento = sets[0];
    parser_result set_prueba = sets[1];

    /* ...... */

    return 0;
}
