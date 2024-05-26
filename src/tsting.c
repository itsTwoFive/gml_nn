#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "data_handler.h"
#include "gml_nn.h"

double funcion_de_prueba_act(double a){
    return 1.0f/(1.0f+ exp(-a));
}

double funcion_de_prueba_err(double a, double b){
    return pow(a-b,2);
}

void neural_test(void){

    char filename[] = "../datasets/AIDS.csv";

    int num_input = get_number_atributes(filename)-1;
    // printf("Numero de inputs = %i\n",num_input);
    // exit(0);

    parser_result datas = parse_data(filename,num_input);

    // int class_number = 4;

    // double ** res_data = array_alloc(datas.num_case,class_number);
    // for (int i = 0; i < datas.num_case; i++)
    // {
    //     int num = (int) datas.data_output[i][0];
    //     res_data[i][num] = 1.0;
    // }
    // datas.data_output = res_data;
    // datas.num_out = class_number;

    change_all_values_for(datas.data_output,datas.num_out,datas.num_case,0,-1);

    parser_result * div_datas = data_div(datas,(int)(datas.num_case*7/10));

    parser_result train_data = div_datas[0];
    parser_result test_data  = div_datas[1];

    int lay_arr[] = {16,8,1};

    neural_net nn = nn_create(ACT_LRELU,3,lay_arr,num_input);

    printf("Inputs = %i\n",nn.input_count);
    
    nn_weight_randf(&nn);
    
    nn_set_batch_size(&nn,30);

    // nn_set_decay_rate(&nn,0.0);
    // printf("Decay = %f\n",nn.decay_rate);
    nn_set_lerning_rate(&nn,0.0001);
    layer_set_act_func(nn,3,ACT_OPSIGMOID);
    int training_it = 10000;
    for (int i = 0; i < training_it; i++)
    {
        train_network_epoch(nn,train_data.num_case,train_data.data_input,train_data.data_output);
        if(i%100 == 0){
            // printf("Train %i: ",i);
            // matrix * act_cost = cost(nn,train_data.num_case,train_data.data_input,train_data.data_output);
            // mat_print(*act_cost);
            printf("Test:  It %i: ",i);
            matrix * act_cost2 = cost(nn,test_data.num_case,test_data.data_input,test_data.data_output);
            mat_print(*act_cost2);
            // mat_free(act_cost);
            mat_free(act_cost2); 
        }
    }

    // plot_2d_data_for_binary(train_data.data_input,train_data.data_output,train_data.num_case,train_data.num_out);
    // show_areas_2d_plot(nn,train_data.num_out);

    printf("Acurracy: %f\n",single_binary_acurracy_rate(nn,test_data.data_input,num_input,test_data.data_output,1,test_data.num_case));
    
    // for (int i = 0; i < datas.num_case; i++)
    // {
    //     printf("%f, %f, %f expected: %f,%f\n",datas.data_input[i][0],datas.data_input[i][1],datas.data_input[i][2],datas.data_output[i][0],datas.data_output[i][1]);
    //     mat_print(*feed_forward(nn,datas.data_input[i],3));
    // }
}

void new_train_test(void){
    char filename[] = "../datasets/Diabetes.csv";

    int num_input = get_number_atributes(filename)-1;

    parser_result datas = parse_data(filename,num_input);

    change_all_values_for(datas.data_output,datas.num_out,datas.num_case,0,-1);

    parser_result * div_datas = data_div(datas,(int)(datas.num_case*7/10));

    parser_result train_data = div_datas[0];
    parser_result test_data  = div_datas[1];

    int lay_arr[] = {8,1};

    neural_net nn = nn_create(ACT_OPSIGMOID,2,lay_arr,num_input);

    // layer_set_alpha(nn,1,0.0);
    // layer_set_alpha(nn,2,0.0);
    
    // nn_set_rand_seed(&nn,25);
    nn_weight_randf(&nn);
    
    nn_set_batch_size(&nn,10);

    nn_set_lerning_rate(&nn,0.003);
    //  nn_set_decay_rate(&nn,0.2);
    
    // layer_custom_act_func(nn,1,funcion_de_prueba_act);
    // layer_custom_act_func(nn,2,funcion_de_prueba_act);
    // layer_custom_act_func(nn,3,funcion_de_prueba_act);
    
    // layer_set_act_func(nn,3,ACT_OPSIGMOID);
    // layer_set_act_func(nn,3,ACT_OPSIGMOID);

    nn_set_training_data(nn,train_data.num_case,train_data.data_input,train_data.data_output);
    nn_set_testing_data(nn,test_data.num_case,test_data.data_input,test_data.data_output);
    
    int training_it = 400;

    // nn_set_console_out(&nn,PRT_ONLYEPOCH);
    nn_set_cost_output(&nn,COUT_GNUPLOT);

    // nn_custom_err_func(&nn,&funcion_de_prueba_err);

    train_network(nn,training_it,1,COST_BOTH);
    printf("Acurracy: %f\n",single_binary_acurracy_rate(nn,test_data.data_input,num_input,test_data.data_output,0.8,test_data.num_case));
    
    nn_save(nn,"diabetesjuju");

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

void save_test(void){
    int lay_arr[] = {3,4,2};

    neural_net nn = nn_create(ACT_LRELU,3,lay_arr,2);
    
    layer_set_act_func(nn,3,ACT_OPSIGMOID);
    nn_save(nn,"nnprueba");
}

void load_test(void){

    neural_net nn = nn_load("diabetesjuju2");
    // layer_set_act_func(nn,3,ACT_HEAVISIDE);
    // nn_custom_err_func(&nn,&funcion_de_prueba_err);

    nn_save(nn,"diabetesjuju3");

    // char filename[] = "../datasets/Diabetes.csv";

    // int num_input = get_number_atributes(filename)-1;

    // parser_result datas = parse_data(filename,num_input);

    // // change_all_values_for(datas.data_output,datas.num_out,datas.num_case,0,-1);

    // parser_result * div_datas = data_div(datas,(int)(datas.num_case*7/10));

    // parser_result train_data = div_datas[0];
    // parser_result test_data  = div_datas[1];

    // printf("Acurracy: %f\n",single_binary_acurracy_rate(nn,test_data.data_input,num_input,test_data.data_output,1,test_data.num_case));
    
}

int main(void)
{
    //  parser_test();
    // neural_test();
    // save_test();
    // load_test();
    new_train_test();
    return 0;
}
