#include <stdio.h>
#include <stdlib.h>
#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    // Se crean los casos de entrenamiento y se cargan
    int num_in = 6;
    parser_result data = parse_data("../datasets/logic.csv",num_in);

    // Estalecemos los tamanos de la red y creamos la Red Neuronal 
    int lay_count[] = {6,2};
    neural_net nn = nn_create(ACT_OPSIGMOID,2,lay_count,num_in);
    nn_set_training_data(nn,data.num_case,data.data_input,data.data_output);

    // Configuramos la tasa de aprendizaje
    nn_set_learning_rate(&nn,0.05);

    // Normalizamos los datos usando Minmax
    change_all_values_for(data.data_input,data.num_in,data.num_case,0.0,-1.0);

    // Para mejor clasificacion cambiamos los 0 de los outputs por -1
    change_all_values_for(data.data_output,data.num_out,data.num_case,0.0,-1.0);
    
    // Iniciamos los pesos
    nn_weight_randf(&nn);
    // Entrenamos la red
    int epoch = 150;
    nn_set_cost_output(&nn,COUT_GNUPLOT);
    train_network(nn,epoch,1,COST_TRAIN);
    
    //Podemos calcular la tasa de acierto discriminando si >0 o si <0
    double acurracy = single_binary_acurracy_rate(nn, data.data_input, num_in, data.data_output, 1, data.num_case);
    printf("Acurracy: %f\n",acurracy);
    return 0;
}
