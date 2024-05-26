#include "matrix.h"

#define ACT_NONE 0
#define ACT_SIGMOID 1
#define ACT_TANH 2
#define ACT_RELU 3
#define ACT_OPSIGMOID 4
#define ACT_LRELU 5
#define ACT_SOFTPLUS 6
#define ACT_HEAVISIDE 7
#define ACT_CUSTOM 9

#define ERR_SQRDIFF 1
#define ERR_HSQRDIFF 2
#define ERR_SIMPDIFF 3
#define ERR_CUSTOM 9

#define PRT_CONSOLE 0
#define PRT_NOCONSOLE 1
#define PRT_ONLYEPOCH 2

#define COUT_ONLY_CONSOLE 1
#define COUT_GNUPLOT 2
#define COUT_CSV 3

#define COST_NONE 0
#define COST_TRAIN 1
#define COST_TEST 2
#define COST_BOTH 3

typedef struct {
    int layer_width;
    int act_func;
    double (*c_act_func)(double);
    double alpha_rate;
    matrix *W;
    matrix *oW;
    matrix *dW;
    matrix *vW;
    matrix *cW;
    matrix *out;
} layer;

typedef struct{
    int num_cases_train;
    double **train_input;
    double **train_output;
    int num_cases_test;
    double **test_input;
    double **test_output;
} data;

typedef struct{
    int input_count;
    int err_func;
    double (*c_err_func)(double,double);
    int layer_count;
    double learning_rate;
    double decay_rate;
    double epsilon_rate;
    int rand_seed;
    int batch_size;
    int cost_output;
    int console_out;
    layer **layers;
    data *dataset;
}neural_net;

neural_net nn_create(int act_func, int layer_count, int layer_widths[], int input_count);

void layer_print(neural_net nn, int layer_num);

void nn_set_lerning_rate(neural_net *nn, double learning_rate);

void nn_set_decay_rate(neural_net *nn, double decay_rate);

void nn_set_epsilon(neural_net *nn, double epsilon_value);

void nn_set_batch_size(neural_net *nn, int size);

void layer_set_act_func(neural_net nn, int layer_pos, int act_func);

void layer_set_alpha(neural_net nn, int layer_pos, double alpha);

void layer_custom_act_func(neural_net nn, int layer_pos, double (*func)(double));

void nn_set_err_func(neural_net *nn, int err_func);

void nn_set_cost_output(neural_net *nn, int cost_out);

void nn_set_console_out(neural_net *nn, int console_out);

void nn_custom_err_func(neural_net *nn, double (*func)(double,double));

void nn_set_rand_seed(neural_net *nn, int seed);

void nn_weight_randf(neural_net nn);

matrix * cost(neural_net nn,int data_length, double ** data, double ** results);

void layer_forward(layer* lay, matrix* input);

matrix* feed_forward(neural_net nn, double data[], int data_size);

void train_network_epoch(neural_net nn, int data_length, double** data, double** results);

void train_network(neural_net nn, int epochs, int print_cost_each, int which_cost);

void nn_set_training_data(neural_net nn, int num_cases, double ** train_input, double** train_output);
 
void nn_set_testing_data(neural_net nn, int num_cases, double ** test_input, double** test_output);

double single_binary_acurracy_rate(neural_net nn, double ** input, int data_size,double ** expected_arr,double dist,int case_num);

int choose_class(double * outputs,int num_out,int target);


//! LOADER - SAVER

void nn_save(neural_net nn, char* name);

neural_net nn_load(char* filename);

//! VISUALIZER

void plot_2d_data_for_binary(double** data_in,double** data_out, int num_cases,int num_out);

void show_areas_2d_plot(neural_net nn,int num_out);
