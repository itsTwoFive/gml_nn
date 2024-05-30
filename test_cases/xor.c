#include <stdio.h>
#include <stdlib.h>
#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    
    // Estalecemos los tamanos de la red y creamos la Red Neuronal 
    int lay_count[] = {2,1};
    neural_net nn = nn_create(ACT_OPSIGMOID,2,lay_count,2);

    // Configuramos la tasa de aprendizaje
    nn_set_learning_rate(&nn,0.3);

    // Iniciamos los pesos
    nn_weight_randf(&nn);

    // Se crean los casos de entrenamiento y se cargan
    int num_in = 2, num_cases = 4;
    parser_result data = parse_data("../datasets/xor.csv",num_in);
    nn_set_training_data(nn,num_cases,data.data_input,data.data_output);

    // Normalizamos los datos usando Minmax
    for (int i = 0; i < num_in; i++)
    {
        minmax_normalization(data.data_input,i,data.num_case);  
    }

    // Para mejor clasificacion cambiamos los 0 de los outputs por -1
    change_all_values_for(data.data_output,data.num_out,data.num_case,0.0,-1.0);


    // Entrenamos la red
    int epoch = 80;
    nn_set_cost_output(&nn,COUT_GNUPLOT);
    train_network(nn,epoch,1,COST_TRAIN);

    for (int i = 0; i < data.num_case; i++)
    {
        printf("Case %i. %i %i = %f expected: %i\n",
        i,
        (int)data.data_input[i][0],
        (int)data.data_input[i][1],
        *mat_seek(*feed_forward(nn,data.data_input[i],2),0,0),
        (int)data.data_output[i][0]);
    }
    

    //Podemos calcular la tasa de acierto discriminando si >0 o si <0
    double acurracy = single_binary_acurracy_rate(nn, data.data_input, num_in, data.data_output, .5, num_cases);
    printf("Acurracy: %f\n",acurracy);
    return 0;
}
