#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float eps = 1e-3;
float rate = 1e-1;

// AND logic gate
float data[][3] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,1}
};

int data_length = sizeof(data)/sizeof(data[0]);

/// @brief Sigmoid activation function
/// @param x input value
/// @return the result of sigmoid(x)
float sigmoid(float x){
    return 1.0f/(1.0f+ exp(-x));
}

/// @brief Calculates the error of the train epoch
/// @param w1 Value of the first weight of the neuron
/// @param w2 Value of the first weight of the neuron
/// @param b Value of the bias of the neuron
/// @return The squared mean error for the batch optimization training
float cost(float w1, float w2, float b){
    float total_cost = 0;
    for (size_t i = 0; i < data_length; i++)
        {
            float x1 = data[i][0];
            float x2 = data[i][1];
            float expected_y = data[i][2];
            float given_y = sigmoid(w1*x1 + w2*x2 + b);
            total_cost += pow(expected_y - given_y,2);
        }
    return total_cost/ (float) data_length;
}

int main(int argc, char const *argv[]){
    srand(258);

    // Initialize the neuron's parameters
    float w1 = (float) rand()/RAND_MAX * 4.0f - 2.0f;
    float w2 = (float) rand()/RAND_MAX *4.0f - 2.0f;
    float b = (float) rand()/RAND_MAX *4.0f - 2.0f;

    // Set the number of training iterations 
    int num_it =1000;
    for (size_t i = 0; i < num_it; i++)
    {
        printf("%zu. Coste de w1: %f, w2: %f y b: %f -> %f\n",i,w1,w2,b,cost(w1,w2,b));
        float cost_finite_diff_w1 = (cost(w1+eps,w2,b)-cost(w1,w2,b))/eps;
        float cost_finite_diff_w2 = (cost(w1,w2+eps,b)-cost(w1,w2,b))/eps;
        float cost_finite_diff_b = (cost(w1,w2,b+eps)-cost(w1,w2,b))/eps;
        w1 -= cost_finite_diff_w1*rate;
        w2 -= cost_finite_diff_w2*rate;
        b -= cost_finite_diff_b*rate;
    }
    
for (int i = 0; i < data_length; i++)
{
    float value = sigmoid(w1*data[i][0] + w2*data[i][1] + b);
    printf("x1: %f, x2: %f, expected: %f, given: %f\n",data[i][0],data[i][1],data[i][2],value);
}


    return 0;
}