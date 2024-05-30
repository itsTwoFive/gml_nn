#include <stdio.h>
#include <stdlib.h>

#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    // Seleccionamos el fichero que contenga el set de datos
    char filename[]  = "../datasets/spiral.csv";

    // Lo leemos y guardamos en memoria los datos
    int num_input =2;
    parser_result pre_data = parse_data(filename,num_input);

    int num_classes;
    double ** res_data = from_integer_to_binary_classes(pre_data.data_output,pre_data.num_case,&num_classes);
    parser_result data;

    data.num_case = pre_data.num_case;
    data.num_in = pre_data.num_in;
    data.num_out = num_classes;

    data.data_input = pre_data.data_input;
    data.data_output = res_data;

    printf("%i,%i,%i\n",data.num_case,data.num_in,data.num_out);


    // Creamos la Red Neuronal 
    int lay_count[] = {32,num_classes};
    neural_net nn = nn_create(ACT_SOFTPLUS,2,lay_count,num_input);

    // Configuramos diversos parametros
    nn_set_learning_rate(&nn,0.001);

    // Dar valor a la semilla del generador de pesos iniciales e inicializar estos pesos
    // nn_set_rand_seed(&nn,0);

    nn_weight_randf(&nn);
    nn_set_batch_size(&nn,5);

    // Podemos configurar tambien distintas Funciones de activacion para cada capa
    // layer_set_act_func(nn,1,ACT_RELU);
    // Entrenamos la red
    int epoch = 160000;
    int print_each = 1000;
    for (int i = 0; i < epoch; i++)
    {
        train_network_epoch(nn,data.num_case,data.data_input,data.data_output);
        if(i%print_each == 0){
            matrix * act_cost = cost(nn,data.num_case,data.data_input,data.data_output);
            printf("EPOCA %i Coste de entrenamiento: ",i);
            mat_print(*act_cost);
            mat_free(act_cost);
        }
    }

    //Podemos calcular la tasa de acierto 
    double *ranges = (double *) malloc(sizeof(double)*4);
    ranges[0] = -20;
    ranges[1] = 20;
    ranges[2] = -15;
    ranges[3] = 15;
    plot_2d_data_for_binary(pre_data.data_input,pre_data.data_output,pre_data.num_case,num_classes,ranges);
    show_areas_2d_plot(nn,num_classes,0.5,ranges);

    return 0;
}
