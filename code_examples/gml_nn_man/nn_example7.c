#include <stdio.h>
#include <stdlib.h>
#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    // Cargamos la red guardada como xor_model
    neural_net nn = nn_load("xor_model");

    // Cargamos el caso a probar
    double xor_case[] = {1,1};

    // Ejecutamos el feedforward sobre la red con el caso
    matrix *result = feed_forward(nn,xor_case,2);

    // Mostramos por consola el resultado de la alimentacion
    mat_print(*result);

    // Se comprueba que el resultado contenido en la neurona es el esperado
    if(*mat_seek(*result,0,0) < 0.1){
        printf("El feedforward ha sido correcto\n");
    }
    else{
        printf("El feedforward ha sido erroneo\n");
    }
    return 0;
}
