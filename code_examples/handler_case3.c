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

    /* Separamos la columna final en casos */
    int num_clases = 0;
    double **separados = from_integer_to_binary_classes(pr.data_output,pr.num_case,&num_clases);

    /* Cambiamos el valor de 0 a -1 en todos los atributos de clase */
    change_all_values_for(separados,num_clases,pr.num_case,0.0,-1.0);

    /* Modificamos los valores del parser_result para utilizar los preprocesados */
    pr.data_output = separados;
    pr.num_out = num_clases;

    /* ...... */

    return 0;
}
