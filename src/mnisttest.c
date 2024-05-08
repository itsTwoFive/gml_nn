#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    char * dataset = "../datasets/mnist_train.csv";
    
    int input_count = 784;
    parser_result pre_data = parse_data(dataset,input_count);

    double ** res_data = array_alloc(pre_data.num_case,10);
    for (int i = 0; i < pre_data.num_case; i++)
    {
        int num = (int) pre_data.data_input[i][0];
        res_data[i][num] = 1.0;
    }
    parser_result data;

    data.num_case = pre_data.num_case;
    data.num_in = pre_data.num_out;
    data.num_out = 10;

    data.data_input = pre_data.data_output;
    data.data_output = res_data;

    change_all_values_for(data.data_output,data.num_out,data.num_case,0,-1);

    parser_result * div_datas = data_div(data,(int)(data.num_case*7/10));

    parser_result train_data = div_datas[0];
    parser_result test_data  = div_datas[1];

    int widths[] = {800,10};

    neural_net nn = nn_create(ACT_LRELU,2,widths,input_count);

    nn_weight_randf(nn);

    nn_set_batch_size(&nn,10);

    nn_set_lerning_rate(&nn,0.001);
    layer_set_act_func(nn,2,ACT_OPSIGMOID);
    int training_it = 100;

    parser_result rt_data = random_trim(test_data,40);

    for (int i = 0; i < training_it; i++)
    {   
        train_network(nn,train_data.num_case,train_data.data_input,train_data.data_output);
        if(i%1 == 0){
            // printf("Train %i: ",i);
            // matrix * act_cost = cost(nn,datas.num_case,train_data.data_input,train_data.data_output);
            // mat_print(*act_cost);
            printf("Test:  It %i: \n",i);
            matrix * act_cost2 = cost(nn,rt_data.num_case,rt_data.data_input,rt_data.data_output);
            mat_print(*act_cost2);
            // mat_free(act_cost);
            mat_free(act_cost2); 
        }
    }


    printf("Acurracy: %f\n",single_binary_acurracy_rate(nn,rt_data.data_input,input_count,rt_data.data_output,1,rt_data.num_case));

    return 0;
}