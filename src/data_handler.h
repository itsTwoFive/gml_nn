#include <stdio.h>


#define EPSILON 0.00001

typedef struct{
    double** data_input;
    double** data_output;
    char** atrib_names;
    int num_case;
    int num_in;
    int num_out;
}parser_result;

int get_number_lines(FILE* fp);

int get_number_atributes(char filename[]);

double ** array_alloc(int rows, int cols);

parser_result parse_data(char filename[],int num_inputs);

parser_result* data_div(parser_result in, int div_size);

void change_all_values_for(double ** data,int size_x, int size_y,float actual, float new);