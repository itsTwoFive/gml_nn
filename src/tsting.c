#include <stdlib.h>
#include <stdio.h>

#include "gml_nn.h"
#include "data_handler.h"

double funcion_de_prueba_act(double a){
    return a*2;
}

double funcion_de_prueba_err(double a, double b){
    printf("A\n");
    return a*b;
}

void neural_test(void){

    char filename[] = "../Datasets/AIDS_Classification.csv";

    int num_input = get_number_atributes(filename)-1;
    // printf("Numero de inputs = %i\n",num_input);
    // exit(0);

    parser_result datas = parse_data(filename,num_input);

    change_all_values_for(datas.data_output,datas.num_out,datas.num_case,0,-1);
    parser_result * div_datas = data_div(datas,(int)(datas.num_case*7/10));

    parser_result train_data = div_datas[0];
    parser_result test_data  = div_datas[1];

    int lay_arr[] = {32,16,8,1};

    neural_net nn = nn_create(ACT_LRELU,4,lay_arr,num_input);

    printf("Inputs = %i\n",nn.input_count);
    
    nn_weight_randf(nn);
    // layer_print(nn,3);

    //feed_forward(nn,datas.data_input[3],nn.input_count);
    //mat_print(*nn.layers[0]->out);

    nn_set_batch_size(&nn,30);

    // nn_set_decay_rate(&nn,0.0);
    // printf("Decay = %f\n",nn.decay_rate);
    nn_set_lerning_rate(&nn,0.0005);
    layer_set_act_func(nn,4,ACT_OPSIGMOID);
    int training_it = 10000;
    for (int i = 0; i < training_it; i++)
    {
        train_network(nn,train_data.num_case,train_data.data_input,train_data.data_output);
        if(i%100 == 0){
            // printf("Train %i: ",i);
            // matrix * act_cost = cost(nn,datas.num_case,train_data.data_input,train_data.data_output);
            // mat_print(*act_cost);
            printf("Test:  It %i: ",i);
            matrix * act_cost2 = cost(nn,test_data.num_case,test_data.data_input,test_data.data_output);
            mat_print(*act_cost2);
            // mat_free(act_cost);
            mat_free(act_cost2); 
        }
    }

    printf("Acurracy: %f\n",single_binary_acurracy_rate(nn,test_data.data_input,num_input,test_data.data_output,1,test_data.num_case));
    
    // for (int i = 0; i < datas.num_case; i++)
    // {
    //     printf("%f, %f expected: %f\n",datas.data_input[i][0],datas.data_input[i][1],datas.data_output[i][0]);
    //     mat_print(*feed_forward(nn,datas.data_input[i],2));
    // }
    

}

void parser_test(void){
    parser_result data = parse_data("../Datasets/Diabetes.csv",8);


    change_all_values_for(data.data_output,data.num_out,data.num_case,0,-1);

    for (int i = 0; i < (data.num_case); i++)
    {
        printf("%i, %f\n",i,data.data_output[i][0]);
    }
    printf("%i\n",data.num_case);
    

//    data_div(data,10);

    //printf("%i, %i, %i, %f, %f",train->num_case,train->num_in,train->num_out,train->data_input[200][4],data.data_input[200][4]);
}

int main(void)
{
    //  parser_test();
    neural_test();
    return 0;
}
