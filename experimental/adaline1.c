#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/// @brief Input data
float data[][2] = {
    {0.0f,1.0f},
    {1.0f,3.0f},
    {2.0f,5.0f},
    {3.0f,7.0f},
    {-1.0f,-1.0f},
    {-2.0f,-3.0f},
    {-3.0f,-5.0f}
};

int data_length = sizeof(data)/sizeof(data[0]);

/// @brief Feeds the neural net in order to achive an output value
/// @param x Case input
/// @param w Weight of the input
/// @param b Bias of the neuron
/// @return Valueo of the neuron feed forward
float feedforward(float x,float w,float b){
    return x*w + b;
}

/// @brief Computes the error between the expected and the given output
/// @param w Weight of the input
/// @param b Bias of the neuron
/// @return Value of the error
float cost(float w, float b){
    float total_cost = 0;
    for (size_t i = 0; i < data_length; i++)
        {
            float x = data[i][0];
            float expected_y = data[i][1];
            float given_y = w*x + b;
            total_cost += pow(expected_y - given_y,2);
        }
    return total_cost/ (float) data_length;
}

/// @brief Computes the gradient for the bias
/// @param w Weight of the neuron
/// @param b Bias of the neuron
/// @return The value of the bias' gradient
float dcost_b(float w, float b,int i){
    float y = data[i][1];
    float a = feedforward(data[i][0],w,b);
    float diff = 2 * (a-y);
    return diff;
}

/// @brief Computes the gradient for the weight
/// @param w Weight of the neuron
/// @param b Bias of the neuron
/// @return The value of the weight's gradient
float dcost_w(float w, float b,int i){
    float y = data[i][1];
    float a = feedforward(data[i][0],w,b);
    float diff = 2 * (a-y);
    return diff;
}

int main()
{
    // Set a seed for random initialization
    srand(25);

    // Give random values for the weight and the bias
    float w = (float) rand()/RAND_MAX * 3.0f ;
    float b = (float) rand()/RAND_MAX * 3.0f ;

    // Choose a learning rate
    float rate = 1e-1;

    // Train for num_it epochs
    int num_it =100;
    for (size_t i = 0; i < num_it; i++)
    {
        // Select a training data to calc a gradient
        int stocastic_sel = rand() % data_length;
        printf("%zu. Cost %f. w: %f b: %f.\n",i,cost(w,b),w,b);

        // Tune values by gradient descent
        w -= dcost_w(w,b,stocastic_sel)*rate;
        b -= dcost_b(w,b,stocastic_sel)*rate;
    }

    // Check if answers are correct
    for (int test_case_num = 0; test_case_num < data_length; test_case_num++)
    {
        float x = data[test_case_num][0];
        printf("f(%f) = %f, expected = %f\n",x,feedforward(x,w,b),data[test_case_num][1]);
    }
    
    return 0;
}
