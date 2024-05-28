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

    nn_save(nn,"xor_model");
    return 0;
}
