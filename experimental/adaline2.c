#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//y = 2x -3z +1
float data[][3] = {
    {0,0,1},
    {1,0,3},
    {0,1,-2},
    {1,1,0},
    {2,0,5},
    {2,1,2},
    {0,2,-5},
};

int data_length = sizeof(data)/sizeof(data[0]);

/// @brief Feeds the neural net in order to achive an output value
/// @param x1 Case input1
/// @param x2 Case input2
/// @param w1 Weight for the input1
/// @param w2 Weight for the input2
/// @param b Bias of the neuron
/// @return Valueo of the neuron feed forward
float feedforward(float x1,float x2,float w1,float w2,float b){
    return x1*w1 +x2*w2 + b;
}

/// @brief Computes the error between the expected and the given output
/// @param w1 Weight of the input1
/// @param w2 Weight of the input2
/// @param b Bias of the neuron
/// @return Value of the error
float cost(float w1, float w2, float b){
    float total_cost = 0;
    for (size_t i = 0; i < data_length; i++)
        {
            float x1 = data[i][0];
            float x2 = data[i][1];
            float expected_y = data[i][2];
            float given_y = feedforward(x1,x2,w1,w2,b);
            total_cost += pow(expected_y - given_y,2);
        }
    return total_cost/ (float) data_length;
}

/// @brief Computes the gradient for the bias
/// @param w1 Weight of the input1
/// @param w2 Weight of the input2
/// @param b Bias of the neuroon
/// @return The value of the bias' gradient
float dcost_b(float w1, float w2, float b,int i){
    float y = data[i][2];
    float a = feedforward(data[i][0],data[i][1],w1,w2,b);
    return 2 * (a-y);
}

/// @brief Computes the gradient for the weitght of the first input
/// @param w1 Weight of the input1
/// @param w2 Weight of the input2
/// @param b Bias of the neuroon
/// @return The value of the first weight's gradient
float dcost_w1(float w1,float w2, float b,int i){
    float y = data[i][2];
    float a = feedforward(data[i][0],data[i][1],w1,w2,b);
    return 2 * (a-y) * data[i][0]; 
}

/// @brief Computes the gradient for the weitght of the second input
/// @param w1 Weight of the input1
/// @param w2 Weight of the input2
/// @param b Bias of the neuroon
/// @return The value of the second weight's gradient
float dcost_w2(float w1,float w2, float b,int i){
    float y = data[i][2];
    float a = feedforward(data[i][0],data[i][1],w1,w2,b);
    return 2 * (a-y) * data[i][1]; 
}

int main()
{
    srand(25);

    float rate = 25e-2;

    float w1 = (float) rand()/RAND_MAX * 1.0f;
    float w2 = (float) rand()/RAND_MAX * 1.0f;
    float b = (float) rand()/RAND_MAX * 1.0f;

    int num_it =100;
    for (size_t i = 0; i < num_it; i++)
    {
        // Select a training data to calc a gradient
        int stocastic_sel = rand() % data_length;

        printf("%zu. Cost %f. w1: %f w2: %f b: %f\n",i,cost(w1,w2,b),w1,w2,b);
        //printf("w1: %f w2: %f b: %f\n",dcost_w1(w1,w2,b),dcost_w2(w1,w2,b),dcost_b(w1,w2,b));
        w1 -= dcost_w1(w1,w2,b,stocastic_sel)*rate;
        w2 -= dcost_w2(w1,w2,b,stocastic_sel)*rate;
        b -= dcost_b(w1,w2,b,stocastic_sel)*rate;
    }

    // Check if answers are correct
    for (int test_case_num = 0; test_case_num < data_length; test_case_num++)
    {
        float x1 = data[test_case_num][0];
        float x2 = data[test_case_num][1];
        printf("f(%f,%f) = %f, expected = %f\n",x1,x2,feedforward(x1,x2,w1,w2,b),data[test_case_num][2]);
    }
    return 0;
}


