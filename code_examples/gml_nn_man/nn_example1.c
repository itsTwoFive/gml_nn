#include <stdio.h>
#include <stdlib.h>
#include "gml_nn.h"
#include "data_handler.h"

int main(int argc, char const *argv[])
{
    // Creacion de la red
    int widths[] = {8,6,7,4};
    neural_net nn =nn_create(ACT_SOFTPLUS,4,widths,5);

    // Inicializacion de los pesos de forma aleatoria
    nn_weight_randf(&nn);
    
    // Seleccion de funciones de activacion para las capas 1 y 4
    layer_set_act_func(nn,1,ACT_LRELU);
    layer_set_act_func(nn,4,ACT_SIGMOID);

    return 0;
}
