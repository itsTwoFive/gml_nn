#include "gml_nn.h"
#include "data_handler.h"


int main(void)
{
    char * dataset = "../datasets/mnist_train.csv";
    
    int input_count = 784;
    parser_result pre_data = parse_data(dataset,1);

    int num_classes;

    for (int i = 0; i < input_count; i++)
    {
        minmax_normalization(pre_data.data_output,i,pre_data.num_case);  
    }

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

    neural_net nn = nn_load("MNIST");

    int trim_size = 100;
    parser_result rt_data = random_trim(test_data,trim_size);
    int success = 0;
    for (int i = 0; i < trim_size; i++)
    {  
        double *current_out = mat_toarray(*feed_forward(nn,rt_data.data_input[i],input_count));
        if (choose_best_class(current_out,10) == choose_best_class(rt_data.data_output[i],10)){
            success++;
        }
    }
    printf("Accuracy: %f\n",(double)success/(double)trim_size);
     nn_save(nn,"MNIST");

    return 0;
}
