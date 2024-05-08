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

    // Para mejor clasificacion cambiamos los 0 de los outputs por -1
    change_all_values_for(data.data_output,data.num_out,data.num_case,0.0,-1.0);

    // Dividimos los datos en set de entrenamiento y pruebas
    double div_num = 75.0/100.0;
    parser_result * div_data = data_div(data,(int) data.num_case*div_num);
    parser_result train_data = div_data[0];
    parser_result test_data  = div_data[1];
    printf("%i\n",train_data.num_case);

    // Creamos la Red Neuronal 
    int lay_count[] = {15,8,1};
    neural_net nn = nn_create(ACT_LRELU,3,lay_count,num_input);

    // Configuramos diversos parametros
    nn_set_batch_size(&nn,20);
    nn_set_decay_rate(&nn,0.0);
    nn_set_lerning_rate(&nn,3e-4); //! learning

    // Dar valor a la semilla del generador de pesos iniciales e inicializar estos pesos
    // nn_set_rand_seed(&nn,0);

    nn_weight_randf(nn);

    // Podemos configurar tambien distintas Funciones de activacion para cada capa
    layer_set_act_func(nn,3,ACT_OPSIGMOID);

    // Entrenamos la red
    int epoch = 1e4;
    int print_each = 100;
    for (int i = 0; i < epoch; i++)
    {
        train_network(nn,train_data.num_case,train_data.data_input,train_data.data_output);
        if(i%print_each == 0){
            matrix * act_cost = cost(nn,train_data.num_case,train_data.data_input,train_data.data_output);
            matrix * act_cost2 = cost(nn,test_data.num_case,test_data.data_input,test_data.data_output);
            printf("EPOCH %i\nTrain: ",i);
            mat_print(*act_cost);
            mat_free(act_cost);
            printf("Test : ");
            mat_print(*act_cost2);
            mat_free(act_cost2); 
        }
    }

    //Podemos calcular la tasa de acierto 
    double acurracy = single_binary_acurracy_rate(nn,test_data.data_input,num_input,test_data.data_output,1,test_data.num_case);

    printf("Acurracy: %f\n",acurracy);
    return 0;
}
