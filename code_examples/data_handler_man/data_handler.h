// This matrix.h code version may differ from the finals version code founded in src

#include <stdio.h>


#define LOG_PARSER 0

#define EPSILON_DISTANCE 0.00001

typedef struct{
    double **data_input;
    double **data_output;
    char **atrib_names;
    int num_case;
    int num_in;
    int num_out;
}parser_result;

int get_number_cases(char filename[]);

int get_number_atributes(char filename[]);

double **array_alloc(int rows, int cols);

parser_result parse_data(char filename[], int num_inputs);

parser_result *data_div(parser_result in, int div_size);

parser_result random_trim(parser_result data, int size);

void change_all_values_for(double **data, int size_x, int size_y, double actual, double new);

double **from_integer_to_binary_classes(double **list, int num_case,int *num_classes);