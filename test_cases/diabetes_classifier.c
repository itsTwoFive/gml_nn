#include <stdio.h>
#include <stdlib.h>
#include "gml_nn.h"
#include "data_handler.h"


int main(void)
{
    // Seleccionamos el fichero que contenga el set de datos
    char filename[]  = "../datasets/Diabetes.csv";

    int num_output = 1;
    int num_input = get_number_atributes(filename)-num_output;

    // Lo leemos y guardamos en memoria los datos
    parser_result data = parse_data(filename,num_input);

    // Normalizamos los datos usando Minmax
    for (int i = 0; i < num_input; i++)
    {
        minmax_normalization(data.data_input,i,data.num_case);  
    }

    // Dividimos los datos en set de entrenamiento y pruebas
    double div_num = 75.0/100.0;
    parser_result * div_data = data_div(data,(int) data.num_case*div_num);
    parser_result train_data = div_data[0];
    parser_result test_data  = div_data[1];

    // Estalecemos los tamanos de la red
    int lay_count[] = {12,8,1};

    // Creamos la Red Neuronal 
    neural_net nn = nn_create(ACT_RELU,3,lay_count,num_input);

    // Configuramos tamano del minilote
    nn_set_batch_size(&nn,20);

    // Configuramos la tasa de aprendizaje
    nn_set_learning_rate(&nn,0.03);
    nn_set_decay_rate(&nn,0.2);

    // Dar valor a la semilla del generador de pesos iniciales e inicializar estos pesos
    nn_set_rand_seed(&nn,20201);
    nn_weight_randf(&nn);

    layer_set_act_func(nn,3,ACT_SIGMOID);

    // Agregamos los datos de entrenamiento y prueba a la red
    nn_set_training_data(nn,train_data.num_case,train_data.data_input,train_data.data_output);
    nn_set_testing_data(nn,test_data.num_case,test_data.data_input,test_data.data_output);

    // Entrenamos la red
    int epoch = 4000;
    int print_each = 10;
    nn_set_cost_output(&nn,COUT_GNUPLOT);
    train_network(nn,epoch,print_each,COST_BOTH);

    //Podemos calcular la tasa de acierto discriminando si >0 o si <0
    double acurracy = single_binary_acurracy_rate(nn,
        test_data.data_input,num_input,test_data.data_output,.5,test_data.num_case);

    printf("Acurracy: %f\n",acurracy);
    return 0;
}
