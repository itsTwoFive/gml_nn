#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix2.h"

// float data[][3] = {
//     {0,0,0},
//     {1,0,1},
//     {0,1,1},
//     {1,1,0}
// };

float data[][3] = {
    {.6,.6,1},
    {.4,.6,1},
    {.4,.4,1},
    {.4,.5,1},
    {.5,.4,1},
    {.1,.1,0},
    {.1,.8,0},
    {0,1,0},
    {1,0,0},
    {.8,.9,0},
    {.1,.9,0},
    {.8,.1,0},
    {.5,.1,0},
    {.3,.15,0},
    {.4,.9,0},
    {.65,.85,0},
    {.9,.5,0},
    {.1,.5,0}
};

int data_length = sizeof(data)/sizeof(data[0]);

// Ax -> Resultado de capa "x"
// Wx -> Peso de la capa "x" a "x+1"
// dWx -> diferenciales de los pesos de la capa "x" 
// vWx -> velocidad de los pesos de la capa "x" (Uso de Momento)
matrix  *W1,  *W2, *A0, *A1, *A2;
matrix *dW1, *dW2;
matrix *oW1, *oW2;
matrix *vW1, *vW2;

float y;

// nx -> Numero de neuronas en la capa "x"
int num_inputs = 2;
int n1 = 3;
int n2 = 1;

float rate = 0.25f;   
float decay = 0.5f; // Factor de decadencia

/// @brief Sigmoid activation function
/// @param x input value
/// @return the result of sigmoid(x)
float sigmoid(float x){
    return 1.0f/(1.0f+ expf(-x));
}

/// @brief Derivate of the sigmoid activation function
/// @param x input value
/// @return the result of sigmoid'(x)
float dsigmoid(float x){
    return sigmoid(x)*(1-sigmoid(x));
}

/// @brief Initializes the weight and bias matrixes of a whole layer
/// @param previous_n Number of neurons in the previous layer
/// @param current_n Number of neurons in the layer to be initialized
/// @param weights Pointer to the Matrix which contains the layer's weights and biases
matrix * initCapa(int previous_n,int current_n){
    matrix * weights = mat_alloc(previous_n+1,current_n);
    mat_randf(*weights);
    return weights;
}

/// @brief Initializes all the layers of the network calling to the method initCapa
void initRandVariables(){
    srand(19242);
    W1 = initCapa(num_inputs,n1);
    W2 = initCapa(n1,n2);

    dW2 = mat_alloc(W2->rows,W2->cols);
    dW1 = mat_alloc(W1->rows,W1->cols);

    vW2 = mat_alloc(W2->rows,W2->cols);
    vW1 = mat_alloc(W1->rows,W1->cols);

    oW1 = mat_alloc(1,n1);
    oW2 = mat_alloc(1,n2);

    A1 = mat_alloc(1,n1 +1);
    A2 = mat_alloc(1,n2 +1);

}

/// @brief Initialize the input layer with the input values 
/// @param inputs Input values of the case
/// @param i_mat pointer to the input matrix
/// @return returns a matrix containing all the inputs/ this is the layer's output
matrix* initInputLayer(float inputs[],matrix* i_mat){
    for (int j = 0; j < num_inputs; j++){
        mat_set_number(*i_mat,0,j,inputs[j]);
    }
    mat_set_number(*i_mat,0,num_inputs,1);
}

/// @brief Feeds all the neurons in the layer in order to calculate it's output
/// @param input Output of the previous layer
/// @param weights The weights between the last layer and the current one and the respective biases
/// @return A matrix containing all the outputs
void layerForward(matrix input, matrix weights, matrix*output){
    matrix * multiply_sol = mat_alloc(input.rows,weights.cols);
    mat_productf(input,weights,multiply_sol);
    for (int j = 0; j < output->cols-1; j++)
    {
        float *value = mat_seek(*multiply_sol,0,j);
        mat_set_number(*output,0,j,sigmoid(*value));
    }
    mat_set_number(*output,0,output->cols -1,1); // [x1,x2,x3,1]
    mat_free(multiply_sol);
}

/// @brief Feeds the network and retrieves its output
/// @param feededData Initial input data
/// @return Value of the output of the neural network
float feedforwardNetwork(float feededData[]){
    if (A0 ==NULL) A0 = mat_alloc(1,num_inputs+1);
    initInputLayer(feededData,A0);
    layerForward(*A0,*W1,A1);
    layerForward(*A1,*W2,A2);

    return *mat_seek(*A2,0,0);
}

/// @brief Calculates the squared error for a given input
/// @param index The input values' index
/// @return The error of the feedforwrad operation 
float singleCost(int index){
    return pow(*mat_seek(*A2,0,0) - data[index][2],2);
}

/// @brief Calculates the mean squared error for the whole case batch
/// @return The mean of the btach errors
float cost(){
    float total_cost = 0;
    for (int i = 0; i < data_length; i++)
    {
        feedforwardNetwork(data[i]);
        
        total_cost += singleCost(i);
    }
    return total_cost/ (float) data_length;
}
/// @brief Calculates the gradient for the weights and biases of the outer layer
void diffOutLayer(){
    for (int j = 0; j < W2->cols; j++)
    {
        float output = *mat_seek(*A2,0,j);
        float common_diff = 2 * (output-y)*output*(1-output);
        mat_set_number(*oW2,0,j,common_diff);
        for (int i = 0; i < W2->rows; i++) // Aqui solo itera una vez, revisar para siguientes implementaciones
        {
           mat_set_number(*dW2,i,j,
                        common_diff * (*mat_seek(*A1,0,i)));
        }
    }
}

/// @brief alculates the gradient for the weights and biases of the hidden layer
void diffHiddenLayer(){
    for (int j = 0; j < W1->cols; j++)
    {
        float output = *mat_seek(*A1,0,j); // suma diff columna V 
        float sum_OxW = 0;
        for (int i = 0; i < oW2->cols; i++){
            sum_OxW += *mat_seek(*oW2,0,i) * *mat_seek(*W2,j,i);
        }
        float common_diff =output * (1-output) * sum_OxW;
        mat_set_number(*oW1,0,j,common_diff);
        for (int i = 0; i < W1->rows; i++)
        {
           mat_set_number(*dW1,i,j,
                        common_diff * (*mat_seek(*A0,0,i)));
        }
    }
}

/// @brief Method to adjust the weights and biases using the gradients and the learning rate
/// using momentum
void trainNetwork(){
    
    matrix *cW1, *cW2;

    cW1 = mat_alloc(W1->rows,W1->cols);
    cW2 = mat_alloc(W2->rows,W2->cols);

    for (int i = 0; i < data_length; i++)
    {
        y = data[i][2];
        feedforwardNetwork(data[i]);
        diffOutLayer();
        diffHiddenLayer();

        mat_sumf(*cW1,*dW1,cW1);
        mat_sumf(*cW2,*dW2,cW2);

    }
    float coeficent = rate / (float) data_length;

    mat_dot_productf(*cW1,coeficent,cW1);
    mat_dot_productf(*vW1,decay,vW1);
    mat_sumf(*vW1,*cW1,vW1);
    mat_subsf(*W1,*vW1,W1);

    mat_dot_productf(*cW2,coeficent,cW2);
    mat_dot_productf(*vW2,decay,vW2);
    mat_sumf(*vW2,*cW2,vW2);
    mat_subsf(*W2,*vW2,W2);

    mat_free(cW1);
    mat_free(cW2);

}

/// @brief Plots using gnuplot a graph showing all the training points
/// This function is extrated from future code in order to make the TFG's memory
/// @param data The dataset containing the training data
/// @param length length of the data
/// @param plotname Name of the plot to be printed
void plot2DDataForBinary(float data[data_length][3], int length, char * plotname){
    
    FILE *pipe = popen("gnuplot -persist", "w"); // Abrir un pipe a Gnuplot

    // Enviar comandos a Gnuplot para plotear
    fprintf(pipe, "set title 'Input Data'\n");
    //fprintf(pipe, "set terminal x11\n");
    fprintf(pipe, "set terminal png\n");
    fprintf(pipe, "set output 'plot1.png'\n");
    fprintf(pipe, "plot '-' w circles lc rgb 'blue' lw 1.5, ");
    fprintf(pipe, " '-' w circles lc rgb 'red' lw 1.5\n");
     fflush(pipe);

    for (int i = 0; i < length; i++)
    {
        if (data[i][2] > 0.5) fprintf(pipe, "%f %f\n", data[i][0], data[i][1]);
    }
    fprintf(pipe,"e\n");
    fflush(pipe);    

    for (int i = 0; i < length; i++)
    {
        if (data[i][2] < 0.5) fprintf(pipe, "%f %f\n", data[i][0], data[i][1]);
    }
    fprintf(pipe,"e\n");

    // Cerrar el pipe
    pclose(pipe);

}

/// @brief Creates a proces runing gnuplot and creates an area map for the results of the network
/// This function is extrated from future code in order to make the TFG's memory
void showAreas2DPlot(){
    float step = .01;
    
    FILE *pipe = popen("gnuplot -persist", "w"); // Abrir un pipe a Gnuplot

    fprintf(pipe, "set title 'Zones'\n");
    //fprintf(pipe, "set terminal x11\n");
    fprintf(pipe, "set terminal png\n");
    fprintf(pipe, "set output 'plot2.png'\n");
    fprintf(pipe, "plot '-' w p lc rgb 'blue' lw 1.5, ");
    fprintf(pipe, " '-' w p lc rgb 'red' lw 1.5\n");
    fflush(pipe);

    for (float x = 0; x < 1; x +=step)
    {
        for (float y = 0; y < 1; y+=step)
        {
            float punto[] = {x,y};
            if (feedforwardNetwork(punto)>0.5) fprintf(pipe, "%f %f\n", x, y);;
        }
    }

    fprintf(pipe,"e\n");
    fflush(pipe); 

    for (float x = 0; x < 1; x +=step)
    {
        for (float y = 0; y < 1; y+=step)
        {
            float punto[] = {x,y};
            if (feedforwardNetwork(punto)<0.5) fprintf(pipe, "%f %f\n", x, y);;
        }
    }
    fprintf(pipe,"e\n");

    // Cerrar el pipe
    pclose(pipe);
    
}

int main()
{
    initRandVariables();

    int num_it = 10000;
    
    
    printf("S. Coste %f.\n",cost());
    for (size_t i = 0; i < num_it; i++)
    {
        trainNetwork();
        if (i%(num_it/100) == 0) 
        printf("%zu. Coste %f.\n",i+1,cost());
        
    }
    for (size_t i = 0; i < data_length; i++)
    {
        float *td = data[i];

        printf("Test with values x1 = %f, x2 = %f, expected = %f, given = %f\n",td[0],td[1],td[2],feedforwardNetwork(td));

    }
    
    // showAreas2DPlot();
    // plot2DDataForBinary(data,data_length,"a");
    return 0;
}

