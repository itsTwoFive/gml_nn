#include <stdio.h>
#include <stdlib.h>

#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    // Seleccionamos el fichero que contenga el set de datos
    char filename[]  = "../datasets/xor.csv";

    int num_output = 1;
    int num_input = get_number_atributes(filename)-num_output;

    // Lo leemos y guardamos en memoria los datos
    parser_result data = parse_data(filename,num_input);

    // Normalizamos los datos usando Minmax
    for (int i = 0; i < num_input; i++)
    {
        minmax_normalization(data.data_input,i,data.num_case);  
    }

    // Para mejor clasificacion cambiamos los 0 de los outputs por -1
    change_all_values_for(data.data_output,data.num_out,data.num_case,0.0,-1.0);


    // Estalecemos los tamanos de la red
    int lay_count[] = {2,1};

    // Creamos la Red Neuronal 
    neural_net nn = nn_create(ACT_OPSIGMOID,2,lay_count,2);

    // Configuramos la tasa de aprendizaje
    nn_set_learning_rate(&nn,0.3);

    // Dar valor a la semilla del generador de pesos iniciales e inicializar estos pesos
    nn_set_rand_seed(&nn,23241);
    nn_weight_randf(&nn);

    // Agregamos los datos de entrenamiento y prueba a la red
    nn_set_training_data(nn,4,data.data_input,data.data_output);
    // Entrenamos la red
    int epoch = 50;
    int print_each = 1;
    train_network(nn,epoch,print_each,COST_TRAIN);

    //Podemos calcular la tasa de acierto discriminando si >0 o si <0
    double acurracy = single_binary_acurracy_rate(nn,
        data.data_input,2,data.data_output,.5,4);

    printf("Acurracy: %f\n",acurracy);
    return 0;
}
