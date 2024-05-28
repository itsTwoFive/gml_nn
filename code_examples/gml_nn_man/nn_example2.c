#include <stdio.h>
#include <stdlib.h>
#include "gml_nn.h"
#include "data_handler.h"

double funcion_de_prueba_act(double a){
    return 7.0f/(1.0f+ exp(-a));
}

double funcion_de_prueba_err(double a, double b){
    return pow(a-b,4);
}

int main(int argc, char const *argv[])
{
    // Creacion de la red
    int widths[] = {4,2};
    neural_net nn =nn_create(ACT_TANH,2,widths,3);

    // Establece funcion de activacion personalizada
    layer_custom_act_func(nn,1,&funcion_de_prueba_act);
    layer_custom_act_func(nn,2,&funcion_de_prueba_act);

    // Establece funcion de error personalizada
    nn_custom_err_func(&nn,&funcion_de_prueba_err);

    return 0;
}
