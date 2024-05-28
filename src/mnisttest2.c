#include "gml_nn.h"
#include "data_handler.h"

int main(void)
{
    char * dataset = "../datasets/mnist_test.csv";
    
    int input_count = 784;
    parser_result pre_data = parse_data(dataset,input_count);

    int num_classes;

    double ** res_data = from_integer_to_binary_classes(pre_data.data_input,pre_data.num_case,&num_classes);
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

    neural_net nn = nn_create(ACT_RELU,2,widths,input_count);

    nn_weight_randf(nn);

    nn_set_batch_size(&nn,10);

    nn_set_learning_rate(&nn,0.003);
    layer_set_act_func(nn,2,ACT_OPSIGMOID);

    // neural_net nn = nn_load("MNIST");

    int training_it = 100;

    parser_result rt_train = random_trim(train_data,1000);

    parser_result rt_data = random_trim(test_data,100);

    nn_set_training_data(nn,rt_train.num_case,rt_train.data_input,rt_train.data_output);
    nn_set_testing_data(nn,rt_data.num_case,rt_data.data_input,rt_data.data_output);

    nn_set_cost_output(&nn,COUT_GNUPLOT);

    train_network(nn,training_it,1,COST_BOTH);


    printf("Acurracy: %f\n",single_binary_acurracy_rate(nn,test_data.data_input,input_count,test_data.data_output,1,test_data.num_case));
    // nn_save(nn,"MNIST");

    return 0;
}
