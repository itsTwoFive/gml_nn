#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>

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

double heaviside(double x){
    if(x>0) return 1;
    else return 0;
}

double dheaviside(){
    return 0;
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
    return fabs(given-expected);
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

    new_layer->alpha_rate = 0.1;

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

data *instance_new_data_space(){
    data *dataset = (data*) malloc(sizeof(data));
    dataset->num_cases_test = 0;
    dataset->num_cases_train = 0;
    return dataset;
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
    nn.cost_output = COUT_ONLY_CONSOLE;
    nn.console_out = PRT_CONSOLE;
    nn.dataset = (data*) instance_new_data_space();
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

void nn_set_cost_output(neural_net *nn, int cost_out){
    nn->cost_output = cost_out;
}

void nn_set_console_out(neural_net *nn, int console_out){
    nn->console_out = console_out;
}

void layer_set_act_func(neural_net nn, int layer_pos, int act_func){
    if (act_func == ACT_CUSTOM){
            printf("WARNING: Se esta usando como funcion de activacion una personalizada,\n\tasegurese de que ha sido establecida usando nn_custom_act_func()\n");
    }
    else if (act_func != ACT_SIGMOID &&
        act_func != ACT_TANH &&
        act_func != ACT_RELU &&
        act_func != ACT_OPSIGMOID &&
        act_func != ACT_LRELU &&
        act_func != ACT_SOFTPLUS &&
        act_func != ACT_HEAVISIDE){
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
    if (err_func == ERR_CUSTOM){
            printf("WARNING: Se esta usando como funcion de error una personalizada,\n\tasegurese de que ha sido establecida usando nn_custom_err_func()\n");
    }
    else if (err_func != ERR_SQRDIFF &&
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


void layer_forward(layer* lay,matrix* input){
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
        else if(act_func==ACT_HEAVISIDE){
            act_res = heaviside(value);
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
    layer_forward(nn.layers[0],data_mat); //Capa 0 (Input)
    mat_free(data_mat);
    for (int layer_it = 1; layer_it < nn.layer_count; layer_it++)
    {
        layer_forward(nn.layers[layer_it],nn.layers[layer_it-1]->out);
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

void nn_weight_randf(neural_net *nn){
    if (nn->rand_seed == 0){
        time_t t;
        time(&t);

        unsigned int seed = (unsigned int) t;
        nn->rand_seed = seed;
        printf("Semilla de generacion: %u\n",seed);
        srand(seed);
    }
    else{
        srand(nn->rand_seed);
    }
    for (int layer_it = 1; layer_it < nn->layer_count; layer_it++)
    {
        
        mat_randf(*nn->layers[layer_it]->W);
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
    else if (lay.act_func == ACT_HEAVISIDE){
        return dheaviside();
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

void train_network_epoch(neural_net nn,int data_length,double** data,double** results){
    clean_cvalues(nn);

    if (nn.batch_size>0) {
        reduced_batch_train(nn,data_length,data,results);
    }
    else{
        full_batch_train(nn,data_length,data,results);
    } 
}


void csv_mean_train_network(neural_net nn, int epochs, int print_cost_each, int which_cost){
    data *dataset = nn.dataset;
    FILE * fp = fopen("costs.csv","w");
    fprintf(fp,"epoch");
    fprintf(fp,",mean_train_cost");
    fprintf(fp,",mean_test_cost");
    fprintf(fp,"\n");
    
    for (int i = 0; i < epochs; i++)
    {
        FILE * tmp = tmpfile();
        if((i-1)%print_cost_each == 0 || i+1 == epochs || i == 0){
            if (nn.console_out != PRT_NOCONSOLE) fprintf(tmp,"EPOCH: %i\n",i);
            fprintf(fp,"%i",i+1);
            if (which_cost == COST_BOTH || which_cost == COST_TRAIN){
                matrix * act_cost = cost(nn,dataset->num_cases_train,dataset->train_input,dataset->train_output);
                if (nn.console_out == PRT_CONSOLE){
                    fprintf(tmp,"Train: ");
                    mat_fprint(*act_cost,tmp);  
                }
                double value = 0.0f;
                for (int i = 0; i < act_cost->cols; i++)
                {
                    value += *mat_seek(*act_cost,0,i);
                }
                fprintf(fp,",%f",value/act_cost->cols);
                mat_free(act_cost);
            }
            if (which_cost == COST_BOTH || which_cost == COST_TEST){
                matrix * act_cost2 = cost(nn,dataset->num_cases_test,dataset->test_input,dataset->test_output);
                if (nn.console_out == PRT_CONSOLE){
                    fprintf(tmp,"Test: ");
                    mat_fprint(*act_cost2,tmp);  
                } 
                double value = 0.0f;
                for (int i = 0; i < act_cost2->cols; i++)
                {
                    value += *mat_seek(*act_cost2,0,i);
                }
                fprintf(fp,",%f",value/act_cost2->cols);
                mat_free(act_cost2); 
            }
            rewind(tmp);
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), tmp) != NULL) {
                printf("%s", buffer);
            }
            fclose(tmp);
            fprintf(fp,"\n");
        }
        train_network_epoch(nn,dataset->num_cases_train,dataset->train_input,dataset->train_output);
    }
    fclose(fp);
}

void csv_train_network(neural_net nn, int epochs, int print_cost_each, int which_cost){
    data *dataset = nn.dataset;
    FILE * fp = fopen("costs.csv","w");
    fprintf(fp,"epoch");
    for (int i = 0; i < nn.layers[nn.layer_count-1]->layer_width; i++)
    {
        fprintf(fp,",train_cost%i",i);
    }
    for (int i = 0; i < nn.layers[nn.layer_count-1]->layer_width; i++)
    {
        fprintf(fp,",test_cost%i",i);
    }
    fprintf(fp,"\n");
    
    for (int i = 0; i < epochs; i++)
    {
        FILE * tmp = tmpfile();
        if((i-1)%print_cost_each == 0 || i+1 == epochs || i == 0){
            if (nn.console_out != PRT_NOCONSOLE) fprintf(tmp,"EPOCH: %i\n",i);
            fprintf(fp,"%i",i+1);
            if (which_cost == COST_BOTH || which_cost == COST_TRAIN){
                matrix * act_cost = cost(nn,dataset->num_cases_train,dataset->train_input,dataset->train_output);
                if (nn.console_out == PRT_CONSOLE){
                    fprintf(tmp,"Train: ");
                    mat_fprint(*act_cost,tmp);  
                }
                for (int i = 0; i < act_cost->cols; i++)
                {
                    fprintf(fp,",%f",*mat_seek(*act_cost,0,i));
                }
                mat_free(act_cost);
            }
            if (which_cost == COST_BOTH || which_cost == COST_TEST){
                matrix * act_cost2 = cost(nn,dataset->num_cases_test,dataset->test_input,dataset->test_output);
                if (nn.console_out == PRT_CONSOLE){
                    fprintf(tmp,"Test: ");
                    mat_fprint(*act_cost2,tmp);  
                } 
                for (int i = 0; i < act_cost2->cols; i++)
                {
                    fprintf(fp,",%f",*mat_seek(*act_cost2,0,i));
                }
                mat_free(act_cost2); 
            }
            rewind(tmp);
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), tmp) != NULL) {
                printf("%s", buffer);
            }
            fclose(tmp);
            fprintf(fp,"\n");
        }
        train_network_epoch(nn,dataset->num_cases_train,dataset->train_input,dataset->train_output);
    }
    fclose(fp);
}

void gnuplot_train_network(neural_net nn, int epochs, int print_cost_each, int which_cost){
    if (nn.layers[nn.layer_count-1]->layer_width > 1)
    {
        csv_mean_train_network(nn,epochs,print_cost_each,which_cost);
    }
    else csv_train_network(nn,epochs,print_cost_each,which_cost);

    char *datafile = "costs.csv";
    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp == NULL) {
        fprintf(stderr, "No se puede abrir Gnuplot.\n");
        exit(1);
    }

    
    fprintf(gp, "set datafile separator ','\n");
        fprintf(gp, "set xlabel 'Batch'\n");
        fprintf(gp, "set ylabel 'Cost'\n");

    if(which_cost == COST_BOTH){
        fprintf(gp, "set title 'Train vs Test Error'\n");
        fprintf(gp, "plot '%s' using 1:2 with lines linecolor rgb 'gray' title 'Train error', '%s' using 1:3 with lines linecolor rgb 'black' title 'Test error'\n", datafile, datafile);       
    }
    else if(which_cost == COST_TEST){
        fprintf(gp, "set title 'Test Error'\n");
        fprintf(gp, "plot '%s' using 1:2 with lines linecolor rgb 'black' title 'Test error'\n", datafile);
    }
    else if (which_cost == COST_TRAIN){
        fprintf(gp, "set title 'Train Error'\n");
        fprintf(gp, "plot '%s' using 1:2 with lines linecolor rgb 'gray' title 'Train error'\n", datafile);
    }
    
    
    pclose(gp);
}


void console_train_network(neural_net nn, int epochs, int print_cost_each, int which_cost){
    data *dataset = nn.dataset;
    for (int i = 0; i <= epochs; i++)
    {
        if((i-1)%print_cost_each == 0 || i == epochs || i == 0){
            if (nn.console_out != PRT_NOCONSOLE) printf("EPOCH: %i\n",i);
            if (which_cost == COST_BOTH || which_cost == COST_TRAIN){
                matrix * act_cost = cost(nn,dataset->num_cases_train,dataset->train_input,dataset->train_output);
                if (nn.console_out == PRT_CONSOLE){
                    printf("Train: ");
                    mat_print(*act_cost);  
                } 
                mat_free(act_cost);
            }
            if (which_cost == COST_BOTH || which_cost == COST_TEST){
                matrix * act_cost2 = cost(nn,dataset->num_cases_test,dataset->test_input,dataset->test_output);
                if (nn.console_out == PRT_CONSOLE){
                    printf("Test: ");
                    mat_print(*act_cost2);  
                } ;
                mat_free(act_cost2); 
            }
        }
        train_network_epoch(nn,dataset->num_cases_train,dataset->train_input,dataset->train_output);
    }
}

void train_network(neural_net nn, int epochs, int print_cost_each, int which_cost){
    if (nn.cost_output == COUT_ONLY_CONSOLE){
        console_train_network(nn,epochs,print_cost_each,which_cost);
    }
    else if (nn.cost_output == COUT_CSV){
        csv_train_network(nn,epochs,print_cost_each,which_cost);
    }
    else if (nn.cost_output == COUT_GNUPLOT){
        gnuplot_train_network(nn,epochs,print_cost_each,which_cost);
    }
    else{
        perror("ERROR: Tipo de salida de la funcion de coste no valido\nPara cambiarlo use el metodo nn_set_cost_output\n");
    }
}

void nn_set_training_data(neural_net nn, int num_cases, double ** train_input, double** train_output){
    data *dataset = nn.dataset;
    dataset->num_cases_train = num_cases;
    dataset->train_input = train_input;
    dataset->train_output = train_output;
}

void nn_set_testing_data(neural_net nn, int num_cases, double ** test_input, double** test_output){
    data *dataset = nn.dataset;
    dataset->num_cases_test = num_cases;
    dataset->test_input = test_input;
    dataset->test_output = test_output;
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

int choose_class(double * outputs,int num_out,int target){
    double min_dist = 10000;
    int current_best = 0;
    for (int i = 0; i < num_out; i++)
    {
        double dist = fabs(outputs[i] - target);
        if (dist < min_dist){
            min_dist = dist; 
            current_best = i;
        }
    }
    return current_best;
}

//! SAVER *****************************************************

FILE * create_file(char* name){
    char *filename = (char*) malloc(sizeof(char)* (strlen(name) + 10));

    strcpy(filename,"saved/");
    strcat(filename,name);
    strcat(filename,".nn");
    
    FILE * fp = fopen(filename,"w");
    free(filename);
    return fp;
}

void nn_save(neural_net nn, char* name){
    FILE *fp =create_file(name);
    fprintf(fp,"# File created using gml_nn. For more info https://github.com/itsTwoFive/gml_nn\n");
    fprintf(fp,"Input Number: %i\n",nn.input_count);
    fprintf(fp,"Learning Rate: %.8f\n",nn.learning_rate);
    fprintf(fp,"Decay Rate: %.8f\n",nn.decay_rate);
    fprintf(fp,"Epsilon: %.8f\n",nn.epsilon_rate);
    fprintf(fp,"Batch Size: %i\n",nn.batch_size);
    fprintf(fp,"Error Function: %i\n",nn.err_func);
    fprintf(fp,"Random Seed: %i\n",nn.rand_seed);
    fprintf(fp,"Number of Layers: %i\n",nn.layer_count-1);
    fprintf(fp,"Layer Widths: ");

    fprintf(fp,"%i",nn.layers[1]->layer_width);
    for (int i = 2; i < nn.layer_count; i++)
    {
        fprintf(fp,", %i",nn.layers[i]->layer_width);
    }
    fprintf(fp,"\n\n");

    for (int k = 1; k < nn.layer_count; k++)
    {
        layer lay = *nn.layers[k];
        fprintf(fp,"*Layer\n");
        fprintf(fp,"\tWidth: %i\n",lay.layer_width);
        fprintf(fp,"\tAlpha Value: %.8f\n",lay.alpha_rate);
        fprintf(fp,"\tActivation Function: %i\n",lay.act_func);
        fprintf(fp,"  -Weights");

        for (int i = 0; i < lay.W->rows; i++){
            fprintf(fp,"\n\t");
            for (int j = 0; j < lay.W->cols; j++){
                fprintf(fp,"%.10f ",*mat_seek(*lay.W,i,j));
            }
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

FILE * open_nn(char *filename){
    FILE *fp = fopen(filename,"r");
    if (fp == NULL){
        char *filepath = (char*) malloc(sizeof(char)* (strlen(filename) + 10));
        strcpy(filepath,"saved/");
        strcat(filepath,filename);
        strcat(filepath,".nn");

        fp = fopen(filepath,"r");
        if (fp == NULL){
            perror("ERROR: Could not find the neural network file.\n");
            perror("       Try writting the path or the name if its saved in 'saved' folder\n");
            exit(1);
        }
    }
    return fp;
}

double get_value(FILE * fp,char delim){
    char c;
    char num[100];
    memset(num,'\0',99);
    int count = 0;
    while ((c = fgetc(fp)) != EOF) {

        if (c == delim) {
            return (double)atof(num);
        }
        else if(c == ':'){
            memset(num,'\0',99);
            count =0;
        }
        else if(c == ' ' || c== '\t'){

        }
        else{
            num[count] = c;
            count++;
        }
    }
    return (double)atof(num);
    
}

neural_net nn_load(char* filename){
    FILE *fp = open_nn(filename);

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') {
            continue;
        }
        fseek(fp, -strlen(line), SEEK_CUR);
        break;
    }

    int input_num = (int)get_value(fp,'\n');
    double learning_rate = get_value(fp,'\n');
    double decay_rate = get_value(fp,'\n');
    double epsilon = get_value(fp,'\n');
    int batch_size = (int)get_value(fp,'\n');
    int error_function = (int)get_value(fp,'\n');
    int random_seed = (int)get_value(fp,'\n');
    int lay_num = (int)get_value(fp,'\n');
    int * layer_widths = (int*)malloc(sizeof(int)*lay_num);
    for (int i = 0; i < lay_num-1; i++)
    {
        layer_widths[i] = (int)get_value(fp,',');
    }
    layer_widths[lay_num-1] = (int)get_value(fp,'\n');

    neural_net nn = nn_create(ACT_NONE,lay_num,layer_widths,input_num);
    nn.learning_rate = learning_rate;
    nn.decay_rate = decay_rate;
    nn.epsilon_rate = epsilon;
    nn.batch_size = batch_size;
    nn_set_err_func(&nn,error_function);
    nn.rand_seed = random_seed;
    //printf("%i,%f,%f,%f,%i,%i,%i\n",nn.input_count,nn.learning_rate,nn.decay_rate,nn.epsilon_rate,nn.batch_size,nn.err_func,nn.rand_seed);
    for (int layer_it = 1; layer_it < lay_num+1; layer_it++)
    {
        get_value(fp,'*');
        get_value(fp,'\n');
        layer lay = *nn.layers[layer_it];
        lay.layer_width = (int)get_value(fp,'\n');
        layer_set_alpha(nn,layer_it,get_value(fp,'\n'));
        int act_func = (int)get_value(fp,'\n');
        layer_set_act_func(nn,layer_it,act_func);
        get_value(fp,'\t');
        for (int i = 0; i < lay.W->rows; i++)
        {
            for (int j = 0; j < lay.W->cols; j++)
            {
                double value = get_value(fp,' ');
                mat_set_number(*lay.W,i,j,value);
            }
        }
    }
    // fclose(fp);
    return nn;
}

//! VISUALIZER ************************************************

void plot_2d_data_for_binary(double** data_in,double** data_out, int num_cases,int num_out){
    
    FILE *pipe = popen("gnuplot -persist", "w"); // Abrir un pipe a Gnuplot

    // Enviar comandos a Gnuplot para plotear
    fprintf(pipe, "set title 'Input Data'\n");
    //fprintf(pipe, "set terminal x11\n");
    fprintf(pipe, "set terminal png\n");
    fprintf(pipe, "set output 'plot1.png'\n");
    fprintf(pipe, "set xrange [-4:4]\n");
    fprintf(pipe, "set yrange [-4:4]\n");
    fprintf(pipe, "plot '-' w circles lw 1.5");
    for (int i = 0; i < num_out-1; i++)
    {
        fprintf(pipe, ", '-' w circles lw 1.5");
    }
    fprintf(pipe, "\n");
     fflush(pipe);

    for (int j = 0; j < num_out; j++)
    {
        for (int i = 0; i < num_cases; i++)
        {
            if (data_out[i][j] > 0.5) fprintf(pipe, "%f %f\n", data_in[i][0], data_in[i][1]);
        }
        fprintf(pipe,"e\n");
        fflush(pipe);  
    }

    // Cerrar el pipe
    pclose(pipe);

}

void show_areas_2d_plot(neural_net nn,int num_out){
    double step = .05;
    
    FILE *pipe = popen("gnuplot -persist", "w"); // Abrir un pipe a Gnuplot

    fprintf(pipe, "set title 'Zones'\n");
    // fprintf(pipe, "set terminal x11\n");
    fprintf(pipe, "set terminal png\n");
    fprintf(pipe, "set output 'plot2.png'\n");
    fprintf(pipe, "plot '-' w p lw 1.5");
    for (int i = 0; i < num_out-1; i++)
    {
        fprintf(pipe, ",  '-' w p lw 1.5");
    }
    fprintf(pipe, "\n");
    fflush(pipe);

    for (int j = 0; j < num_out; j++)
    {
        for (double x = -4; x < 4; x +=step)
        {
            for (double y = -4; y < 4; y+=step)
            {
                double punto[] = {x,y};
                double * out =mat_toarray(*feed_forward(nn,punto,2));
                if (j == choose_class(out,num_out,1))
                {
                    fprintf(pipe, "%f %f\n", x, y);
                }
                free(out);
            }
        }
        fprintf(pipe,"e\n");
        fflush(pipe);  
    }
    // Cerrar el pipe
    pclose(pipe);
}