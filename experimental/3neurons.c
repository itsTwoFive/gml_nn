#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float eps = 1e-2;
float rate = 1e-2;

int training_iterations = 100000;
float minimun = 100;
int min_index = 0;

// float data[][3] = {
//     {0,0,1},
//     {1,0,3},
//     {0,1,-2},
//     {1,1,0},
//     {2,0,5},
//     {2,1,2},
//     {0,2,-5},
// };

float data[][3] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,0},
};

/// @brief Struct that contains the information of the weights and bias
/// incoming to a given neuron
typedef struct {
    float w1;
    float w2;
    float bias;
} Neuron;

Neuron neuron_list[3];
// 2 layers: 1st 2 neurons, 2nd: 1 neuron

Neuron best_config[3];

/// @brief Initialize the weights with random values
void loadNeurons(){
    for (size_t i = 0; i < 3; i++)
    {
        neuron_list[i].w1 = (float) rand()/RAND_MAX;
        neuron_list[i].w2 = (float) rand()/RAND_MAX;
        neuron_list[i].bias = (float) rand()/RAND_MAX;
    }
}

int data_length = sizeof(data)/sizeof(data[0]);

/// @brief Sigmoid function
/// @param x input value
/// @return value of the sigmoid at x
float sigmoid(float x){
    return 1.0f/(1.0f+ expf(-x));
}

/// @brief claculates the error of the whole network
/// @return returns the mean squared difference
float calc_error_neuron(){
    float total_error = 0;
    Neuron n1 = neuron_list[0];
    Neuron n2 = neuron_list[1];
    Neuron n3 = neuron_list[2];
    for (size_t i = 0; i < data_length; i++)
        {
            float x1 = data[i][0];
            float x2 = data[i][1];
            float expected_y = (data[i][2]);
            float given_v1 = sigmoid(n1.w1*x1 + n1.w2*x2 + n1.bias);
            float given_v2 = sigmoid(n2.w1*x1 + n2.w2*x2 + n2.bias);
            float given_y = sigmoid(n3.w1* given_v1 + n3.w2 *given_v2 +n3.bias);
            total_error += pow(expected_y - given_y,2);
            //printf("e%f, g%f\n",expected_y,given_y);
        }
    return total_error/ (float) data_length;
}

/// @brief Tune the value of the weights and the bias
void adjust_params(){
    for (size_t i = 0; i < training_iterations; i++){
        float actual_error = calc_error_neuron();
        for (size_t j = 0; j < 3; j++){
            Neuron *n;
            n = &neuron_list[j];
            float w1 = n->w1; 
            n->w1+=eps;
            float error_finite_diff_w1 = (calc_error_neuron()-actual_error)/eps;
            n->w1 = w1;
            float w2 = n->w2; 
            n->w2+=eps;
            float error_finite_diff_w2 = (calc_error_neuron()-actual_error)/eps;
            n->w2 = w2;
            float bias = n->bias; 
            n->bias+=eps;
            float error_finite_diff_b = (calc_error_neuron()-actual_error)/eps;
            n->bias = bias;
            n->w1 -= error_finite_diff_w1*rate;
            n->w2 -= error_finite_diff_w2*rate;
            n->bias -= error_finite_diff_b*rate;
        }
        printf("Train %lu, Error: %10f\n",i,actual_error);
        if (actual_error < minimun){
            minimun = actual_error;
            min_index = i;
            for (size_t i = 0; i < 3; i++)
            {
                best_config[i].w1 = neuron_list[i].w1;
                best_config[i].w2 = neuron_list[i].w2;
                best_config[i].bias = neuron_list[i].bias;
            }
            
        }
    }
}

/// @brief prints the values for the weights and the bias of a given neuron
/// @param index neuron's identifier
void printNeuron(int index){
    Neuron n = best_config[index];
    printf("Neuron %u >> w1: %f, w2: %f, bias: %f\n",index,n.w1,n.w2,n.bias);
}

/// @brief Feed forward of the network
/// @param x1 First input value
/// @param x2 Second input value
/// @return the result of the last neuron
float calculate(float x1, float x2){
    Neuron n1 = neuron_list[0];
    Neuron n2 = neuron_list[1];
    Neuron n3 = neuron_list[2];
    return sigmoid((n1.w1*x1 +n1.w2*x2 + n1.bias)*n3.w1 + (n2.w1*x1 +n2.w2*x2 + n2.bias) *n3.w2 +n3.bias);
}

int main(int argc, char const *argv[]){
    srand(33);
    loadNeurons();
    adjust_params();

    printf("-------------------\n");
    printf("Min Error in %u iterations = %f. Index = %u\n",training_iterations, minimun,min_index);

    for (size_t i = 0; i < 3; i++)
    {
        printNeuron(i);
    }
    
    printf("-------------------\n");
    for (int i = 0; i < data_length; i++)
    {
        float x1 = data[i][0];
        float x2 = data[i][1];
        printf("Calculated for %f and %f: %f, expected: %f\n",x1,x2,calculate(x1,x2),data[i][2]);
    }

    return 0;
}
