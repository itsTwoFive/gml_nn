#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "gml_nn.h"

double sech(double x){
    return 1 / cosh(x);
}

double sigmoid(double x){
    return 1.0f/(1.0f+ exp(-x));
}

double dsigmoid(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

double op_sigmoid(double x){
    return 1.7159 * tanh((2.0/3.0) * x);
}

double dop_sigmoid(double x){
    double coef = sech(0.66667 *x);
    return (1.14391 * pow(coef,2));
}

double dtanh(double x){
    return pow(sech(x),2);
}

double relu(double x){
    if (x<0) return 0;
    return x;
}

double drelu(double x){
    if (x<0) return 0;
    return 1;
}

double lrelu(double x, double alpha){
    if (x<0) return x*alpha;
    return x;
}

double dlrelu(double x, double alpha){
    if (x<0) return alpha;
    return 1;
}

double softplus(double x){
    return log(1+exp(x));
}

double dsoftplus(double x){
    return sigmoid(x);
}

double sqrdiff(double given,double expected){
    return pow(given-expected,2);
}

double dsqrdiff(double given,double expected){
    return 2*(given-expected);
}

double hsqrdiff(double given, double expected){
    return sqrdiff(given,expected)/2;
}
double dhsqrdiff(double given, double expected){
    return given-expected;
}

double simpdiff(double given, double expected){
    return given-expected;
}

double dsimpdiff(){
    return 1;
}

double finite_diff_act(neural_net nn, layer lay, double x){
    double eps = nn.epsilon_rate;
    return (lay.c_act_func(x+eps) - lay.c_act_func(x))/eps;
}

double finite_diff_err(neural_net nn, double given, double expected){
    double eps = nn.epsilon_rate;
    return (nn.c_err_func(given+eps,expected) - nn.c_err_func(given,expected))/eps;
}

layer* layer_alloc(int layer_width, int input_count, int act_func){

    layer* new_layer = (layer*)malloc(sizeof(layer));

    new_layer->act_func = act_func;
    new_layer->layer_width = layer_width;

    new_layer->alpha_rate = 0.01;

    new_layer->W = mat_alloc(input_count+1,layer_width);
    new_layer->oW = mat_alloc(input_count+1,layer_width);
    new_layer->dW = mat_alloc(input_count+1,layer_width);
    new_layer->vW = mat_alloc(input_count+1,layer_width);
    new_layer->cW = mat_alloc(input_count+1,layer_width);
    new_layer->out = mat_alloc(1,layer_width+1);

    if (new_layer-> W == NULL || 
        new_layer->oW == NULL || 
        new_layer->dW == NULL ||
        new_layer->vW == NULL ||
        new_layer->cW == NULL || 
        new_layer->out == NULL) 
        {
        free(new_layer->W);
        free(new_layer->oW);
        free(new_layer->dW);
        free(new_layer->vW);
        free(new_layer->out);
        free(new_layer);
        return NULL; 
    }

    return new_layer;
}
void layer_free(layer* l) {
    if (l != NULL) {
        mat_free(l->W);
        mat_free(l->oW);
        mat_free(l->dW);
        mat_free(l->vW);
        mat_free(l->out);

        free(l);
    }
}

neural_net nn_create(int act_func, int layer_count, int layer_widths[],int input_count){
    layer **nn_layers = malloc(sizeof(layer*)*(layer_count +1));
    nn_layers[0] = layer_alloc(input_count,input_count,ACT_NONE);
    for (int i = 0; i < input_count; i++)
    {
        mat_set_number(*nn_layers[0]->W,i,i,1);
    }

    for (int i = 0; i < layer_count; i++)
    {
        nn_layers[i+1] = layer_alloc(layer_widths[i],nn_layers[i]->layer_width, act_func);
    }
    neural_net nn;
    
    nn.err_func = ERR_SQRDIFF;
    nn.layer_count = layer_count+1;
    nn.learning_rate = 0.1f;
    nn.decay_rate = 0.0f;
    nn.layers = nn_layers;
    nn.rand_seed = 0;
    nn.batch_size = 0;
    nn.input_count = input_count;
    nn.epsilon_rate = 1e-3;
    return nn;
}

void layer_print(neural_net nn,int layer_num){
    layer* cur_layer = nn.layers[layer_num];
    int num_neuronas = cur_layer->layer_width;
    matrix *pesos = cur_layer->W;
    int num_pesos_neurona = pesos->rows-1;
    printf("\n");
    for (int i = 0; i < num_neuronas; i++)
    {
        for (int j = 0; j < num_pesos_neurona; j++)
        {
            printf("w%i.%i %f\n",i,j,*mat_seek(*pesos,i,j));
        }
        printf("b%i   %f\n",i, *mat_seek(*pesos,i,num_pesos_neurona));
        printf("\n");
    }
    
}

void nn_set_lerning_rate(neural_net *nn, double learning_rate){
    nn->learning_rate = learning_rate;
}

void nn_set_decay_rate(neural_net *nn, double decay_rate){
    nn->decay_rate = decay_rate;
}

void nn_set_epsilon(neural_net *nn, double epsilon_value){
    nn->epsilon_rate = epsilon_value;
}

void nn_set_batch_size(neural_net *nn, int size){
    nn->batch_size = size;
}

void layer_set_act_func(neural_net nn, int layer_pos, int act_func){
    if (act_func != ACT_SIGMOID &&
        act_func != ACT_TANH &&
        act_func != ACT_RELU &&
        act_func != ACT_OPSIGMOID &&
        act_func != ACT_LRELU &&
        act_func != ACT_SOFTPLUS){
            perror("ERROR: La funcion de activacion no existe, si quiere usar una personalizada use nn_custom_act_func()\n");
        }
    else{
        nn.layers[layer_pos]->act_func = act_func;
    }
}

void layer_set_alpha(neural_net nn, int layer_pos, double alpha){
    nn.layers[layer_pos]->alpha_rate = alpha;
}

void layer_custom_act_func(neural_net nn, int layer_pos, double (*func)(double)){
    nn.layers[layer_pos]->act_func = ACT_CUSTOM;
    nn.layers[layer_pos]->c_act_func = func;
}

void nn_set_err_func(neural_net *nn, int err_func){
    if (err_func != ERR_SQRDIFF &&
        err_func != ERR_HSQRDIFF &&
        err_func != ERR_SIMPDIFF ){
            perror("ERROR: La funcion de error no existe, si quiere usar una personalizada use nn_custom_err_func()\n");
        }
    else{
        nn->err_func = err_func;
    }
}

void nn_custom_err_func(neural_net *nn, double (*func)(double,double)){
    nn->err_func = ERR_CUSTOM;
    nn->c_err_func = func;
}


void layer_forward(neural_net nn, layer* lay,matrix* input){
    matrix * multiply_sol = mat_alloc(input->rows,lay->W->cols);
    // mat_print(*input);
    // mat_print(*lay->W);
    mat_productf(*input,*lay->W,multiply_sol);
    int act_func = lay->act_func;
    for (int j = 0; j < lay->out->cols-1; j++)
    {
        double value = *mat_seek(*multiply_sol,0,j);
        double act_res = 0;
        if (act_func==ACT_SIGMOID){
            act_res =sigmoid(value);
        }
        else if(act_func==ACT_TANH){ 
            act_res = tanh(value);
        }
        else if(act_func==ACT_RELU){
            act_res = relu(value);
        }
        else if(act_func==ACT_OPSIGMOID){
            act_res = op_sigmoid(value);
        }
        else if(act_func==ACT_LRELU){
            act_res = lrelu(value,lay->alpha_rate);
        }
        else if(act_func==ACT_SOFTPLUS){
            act_res = softplus(value);
        }
        else if(act_func==ACT_CUSTOM){
            act_res = lay->c_act_func(value);
        }
        else if(act_func==ACT_NONE){
            act_res = value;
        }
        mat_set_number(*lay->out,0,j,act_res);
    }
    mat_set_number(*lay->out,0,lay->out->cols -1,1); // [x1,x2,x3,1]
    mat_free(multiply_sol);
}

matrix* input_transform(int data_size,double data[]){
    double *nuevo_array = (double*)malloc((data_size + 1) * sizeof(double));
    if (nuevo_array == NULL) {
        fprintf(stderr, "ERROR: No se pudo asignar memoria para el nuevo array. (Input Transform)\n");
        return NULL;
    }
    for (int i = 0; i < data_size; i++) {
        nuevo_array[i] = data[i];
    }

    nuevo_array[data_size] = 0;

    matrix* data_mat = mat_fromarray(data_size+1,nuevo_array);
    free(nuevo_array);
    return data_mat;
}

matrix* delete_bias(matrix* mat){
    matrix * res = mat_alloc(mat->rows,mat->cols -1);
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            mat_set_number(*res,i,j,*mat_seek(*mat,i,j)); //! AQUI PUEDE ROMPER
        }
    }
    return res;
}

matrix* feed_forward(neural_net nn,double data[],int data_size){
    matrix* data_mat = input_transform(data_size,data);
    layer_forward(nn,nn.layers[0],data_mat); //Capa 0 (Input)
    mat_free(data_mat);
    for (int layer_it = 1; layer_it < nn.layer_count; layer_it++)
    {
        layer_forward(nn,nn.layers[layer_it],nn.layers[layer_it-1]->out);
    }
    return delete_bias(nn.layers[nn.layer_count-1]->out);
    // return nn.layers[nn.layer_count-1]->out; // Devuelve el valor con el activo del bias
}

void clean_cvalues(neural_net nn){
    for (int layer = 1; layer < nn.layer_count; layer++)
    {
        mat_set_zeros(*nn.layers[layer]->cW);
    }
}

void nn_set_rand_seed(neural_net *nn, int seed){
    nn->rand_seed = seed;
}

void nn_weight_randf(neural_net nn){
    if (nn.rand_seed == 0){
        time_t t;
        time(&t);

        unsigned int seed = (unsigned int) t;
        nn.rand_seed = seed;
        printf("Semilla de generacion: %u\n",seed);
        srand(seed);
    }
    else{
        srand(nn.rand_seed);
    }
    for (int layer_it = 1; layer_it < nn.layer_count; layer_it++)
    {
        
        mat_randf(*nn.layers[layer_it]->W);
    }
}

double calc_error_diff(neural_net nn, double given, double expected){
    if (nn.err_func == ERR_SQRDIFF){
        return dsqrdiff(given,expected);
    }
    else if (nn.err_func == ERR_HSQRDIFF){
        return dhsqrdiff(given,expected);
    }
    else if (nn.err_func == ERR_SIMPDIFF){
        return dsimpdiff();
    }
    else if (nn.err_func == ERR_CUSTOM){
        return finite_diff_err(nn,given,expected);
    }
    else{
        perror("ERROR: Funcion de error no valida use nn_set_err_func() para cambiarla\n");
        exit(1);
    }
}

double calc_act_diff(neural_net nn, layer lay, double value){
    if (lay.act_func == ACT_SIGMOID){
        return dsigmoid(value);
    }
    else if (lay.act_func == ACT_TANH){
        return dtanh(value);
    }
    else if (lay.act_func == ACT_RELU){
        return drelu(value);
    }
    else if (lay.act_func == ACT_OPSIGMOID){
        return dop_sigmoid(value);
    }
    else if (lay.act_func == ACT_LRELU){
        return dlrelu(value,lay.alpha_rate);
    }
    else if (lay.act_func == ACT_SOFTPLUS){
        return dsoftplus(value);
    }
    else if (lay.act_func == ACT_CUSTOM){
        return finite_diff_act(nn,lay,value);
    }
    else{
        perror("ERROR: Funcion de activacion no valida use nn_set_act_func() para cambiarla\n");
        exit(1);
    }
}

void diff_outer_layer(neural_net nn,double* results){
    layer * o_layer = nn.layers[nn.layer_count-1];
    for (int neuron_it = 0; neuron_it < o_layer->W->cols; neuron_it++)
    {
        double output = *mat_seek(*o_layer->out,0,neuron_it);
        double error_diff = calc_error_diff(nn,output,results[neuron_it]);
        double activation_diff = calc_act_diff(nn,*o_layer,output);
        double common_diff = error_diff * activation_diff;
        mat_set_number(*o_layer->oW,0,neuron_it,common_diff);
        for (int weight_it = 0; weight_it < o_layer->W->rows; weight_it++)
        {
            mat_set_number(*o_layer->dW,neuron_it,weight_it,
                common_diff * *mat_seek(*nn.layers[nn.layer_count-2]->out,0,weight_it));
        }
    }
}

void diff_hidden_layer(neural_net nn,int layer_it){
    layer * c_layer = nn.layers[layer_it];
    layer * next_layer = nn.layers[layer_it +1];
    for (int neuron_it = 0; neuron_it < c_layer->W->cols; neuron_it++)
    {
        double output =  *mat_seek(*c_layer->out,0,neuron_it);
        double sum_oW = 0;
        for (int next_it = 0; next_it < next_layer->oW->cols; next_it++)
        {
            sum_oW += *mat_seek(*next_layer->oW,0,next_it) * *mat_seek(*next_layer->W,neuron_it,next_it);
        }
        double common_diff = calc_act_diff(nn,*c_layer,output) * sum_oW;
        mat_set_number(*c_layer->oW,0,neuron_it,common_diff);
        for (int weight_it = 0; weight_it < c_layer->W->rows; weight_it++)
        {
            mat_set_number(*c_layer->dW,weight_it,neuron_it,
                common_diff * *mat_seek(*nn.layers[layer_it-1]->out,0,weight_it));
        }
    }
}

double calc_cost(neural_net nn, double given, double expected){
    if (nn.err_func == ERR_SQRDIFF){
        return sqrdiff(given,expected);
    }
    else if (nn.err_func == ERR_HSQRDIFF){
        return hsqrdiff(given,expected);
    }
    else if (nn.err_func == ERR_SIMPDIFF){
        return simpdiff(given,expected);
    }
    else if (nn.err_func == ERR_CUSTOM){
        return nn.c_err_func(given,expected);
    }
    else{
        perror("ERROR: Funcion de error no valida use nn_set_err_func() para cambiarla\n");
        exit(1);
    }
}

matrix * single_cost(neural_net nn,matrix given,matrix expected){
    matrix * cost_mat = mat_alloc(1,given.cols);
    for (int i = 0; i < given.cols; i++)
    {
        mat_set_number(*cost_mat,0,i,calc_cost(nn,
            *mat_seek(given,0,i),
            *mat_seek(expected,0,i)));
    }
    return cost_mat;
}

//! CUIDADO Si se usa hay que liberar los recursos despues con mat_free
matrix * cost(neural_net nn,int data_length, double ** data, double ** results){
    int num_out = nn.layers[nn.layer_count-1]->out->cols-1;
    matrix * cost_mat = mat_alloc(1,num_out);
    for (int i = 0; i < data_length; i++)
    {
        matrix * res_mat = mat_fromarray(nn.layers[nn.layer_count-1]->layer_width,results[i]);
        matrix* sing_cost= single_cost(nn,*feed_forward(nn,data[i],nn.input_count) ,*res_mat);
        mat_sumf(*cost_mat, *sing_cost, cost_mat); 
        mat_free(res_mat);
        mat_free(sing_cost);
    }
    mat_dot_productf(*cost_mat,1/(double)data_length,cost_mat);
    
    return cost_mat;
    //mat_free(cost_mat);
}

void full_batch_train(neural_net nn,int data_length,double** data,double** results){
    for (int tdata_it = 0; tdata_it < data_length; tdata_it++)
    {
        feed_forward(nn,data[tdata_it],nn.layers[0]->layer_width); 
        diff_outer_layer(nn,results[tdata_it]); 
        layer * lay = nn.layers[nn.layer_count-1];
        mat_sumf(*lay->cW,*lay->dW,lay->cW);
        for (int layerpos = nn.layer_count-2; layerpos > 0; layerpos--)
        {
            diff_hidden_layer(nn,layerpos); 
            layer *c_lay = nn.layers[layerpos];
            mat_sumf(*c_lay->cW,*c_lay->dW,c_lay->cW);
        }
    }
    double coeficent = nn.learning_rate / (double) data_length;
    for (int layer_it = nn.layer_count-1; layer_it > 0; layer_it--)
    {
        layer* lay = nn.layers[layer_it];
        mat_dot_productf(*lay->cW,coeficent,lay->cW);
        mat_dot_productf(*lay->vW,nn.decay_rate,lay->vW);

        mat_sumf(*lay->vW,*lay->cW,lay->vW);
        mat_subsf(*lay->W,*lay->vW,lay->W);
    } 
}

void reduced_batch_train(neural_net nn,int data_length,double** data,double** results){
    int* order = (int*) malloc(sizeof(int)*nn.batch_size);
    for (int i = 0; i < nn.batch_size; i++)
    {
        order[i] = rand() % data_length;
    }

    for (int tdata_it = 0; tdata_it < nn.batch_size; tdata_it++)
    {
        feed_forward(nn,data[order[tdata_it]],nn.layers[0]->layer_width); 
        diff_outer_layer(nn,results[order[tdata_it]]); 
        layer * lay = nn.layers[nn.layer_count-1];
        mat_sumf(*lay->cW,*lay->dW,lay->cW);
        for (int layerpos = nn.layer_count-2; layerpos > 0; layerpos--)
        {
            diff_hidden_layer(nn,layerpos); 
            layer *c_lay = nn.layers[layerpos];
            mat_sumf(*c_lay->cW,*c_lay->dW,c_lay->cW);
        }
    }
    free(order);
    double coeficent = nn.learning_rate / (double) nn.batch_size;
    for (int layer_it = nn.layer_count-1; layer_it > 0; layer_it--)
    {
        layer* lay = nn.layers[layer_it];
        mat_dot_productf(*lay->cW,coeficent,lay->cW);
        mat_dot_productf(*lay->vW,nn.decay_rate,lay->vW);

        mat_sumf(*lay->vW,*lay->cW,lay->vW);
        mat_subsf(*lay->W,*lay->vW,lay->W);
    } 

}

void train_network(neural_net nn,int data_length,double** data,double** results){
    clean_cvalues(nn);

    if (nn.batch_size>0) {
        reduced_batch_train(nn,data_length,data,results);
    }
    else{
        full_batch_train(nn,data_length,data,results);
    } 
}

double single_binary_acurracy_rate(neural_net nn, double ** input, int data_size,double ** expected,double dist,int case_num){
    int success = 0;
    
    for (int i = 0; i < case_num; i++)
    {
        double value = *mat_seek(*feed_forward(nn,input[i],data_size),0,0);
        double a = fabs(expected[i][0]-value);
        if(a < dist ) success++;
    }
    
    return (double)success/case_num;
}