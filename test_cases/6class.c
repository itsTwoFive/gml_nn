#include <stdio.h>
#include <stdlib.h>

#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    // Seleccionamos el fichero que contenga el set de datos
    char filename[]  = "../datasets/6class.csv";

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

    // Creamos la Red Neuronal 
    int lay_count[] = {32,num_classes};
    neural_net nn = nn_create(ACT_SOFTPLUS,2,lay_count,num_input);
    nn_set_learning_rate(&nn,0.005);
    nn_weight_randf(&nn);
    nn_set_batch_size(&nn,5);

    // Entrenamos la red
    int epoch = 5000;
    int print_each = 100;
    nn_set_training_data(nn,data.num_case,data.data_input,data.data_output);
    train_network(nn,epoch,print_each,COST_TRAIN);

    //Podemos calcular la tasa de acierto 
    double *ranges = (double *) malloc(sizeof(double)*4);
    ranges[0] = -8;
    ranges[1] = 15;
    ranges[2] = -8;
    ranges[3] = 15;
    plot_2d_data_for_binary(pre_data.data_input,pre_data.data_output,pre_data.num_case,num_classes,ranges);
    show_areas_2d_plot(nn,num_classes,0.2,ranges);
    free(ranges);

    return 0;
}
