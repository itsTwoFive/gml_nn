#include <stdio.h>
#include <stdlib.h>

#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    // Seleccionamos el fichero que contenga el set de datos
    char filename[]  = "../datasets/4class.csv";

    int num_output = 4;
    int num_input = get_number_atributes(filename)-num_output;

    // Lo leemos y guardamos en memoria los datos
    parser_result data = parse_data(filename,num_input);

    // Para mejor clasificacion cambiamos los 0 de los outputs por -1
    change_all_values_for(data.data_output,data.num_out,data.num_case,0.0,-1.0);

    // Creamos la Red Neuronal 
    int lay_count[] = {12,num_output};
    neural_net nn = nn_create(ACT_OPSIGMOID,2,lay_count,num_input);

    // Configuramos diversos parametros
    nn_set_decay_rate(&nn,0.0);
    nn_set_lerning_rate(&nn,0.001);

    // Dar valor a la semilla del generador de pesos iniciales e inicializar estos pesos
    // nn_set_rand_seed(&nn,0);

    nn_weight_randf(nn);

    // Podemos configurar tambien distintas Funciones de activacion para cada capa
    layer_set_act_func(nn,2,ACT_OPSIGMOID);

    // Entrenamos la red
    int epoch = 2e4;
    int print_each = 1000;
    for (int i = 0; i < epoch; i++)
    {
        train_network(nn,data.num_case,data.data_input,data.data_output);
        if(i%print_each == 0){
            matrix * act_cost = cost(nn,data.num_case,data.data_input,data.data_output);
            printf("EPOCH %i Train: ",i);
            mat_print(*act_cost);
            mat_free(act_cost);
        }
    }

    //Podemos calcular la tasa de acierto 
   
    plot2DDataForBinary(data.data_input,data.data_output,data.num_case,data.num_out);
    showAreas2DPlot(nn,data.num_out);

    return 0;
}
