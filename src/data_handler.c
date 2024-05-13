#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "data_handler.h"

#define LOG_PARSER 1

#define MAX_LEN 100

int get_number_lines(FILE* fp){
    char c;
    int num_filas = 0;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            num_filas++;
        }
    }
    return num_filas;
}

int get_number_cols(FILE* fp){
    int num_cols = 1;
    char c;
    char name[MAX_LEN];
    memset(name,'\0',MAX_LEN-1);
    int c_count = 0;
    if (LOG_PARSER) printf("PARSER: The dataset atributes are:\n");
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            if (LOG_PARSER){
                printf("PARSER: %s\n",name);
            }
            break;
        }
        else if (c== ','){
            num_cols++;
            c_count = 0;
            if (LOG_PARSER){
                printf("PARSER: %s\n",name);
            }
            memset(name,'\0',MAX_LEN-1);
        }
        else{
            name[c_count++] = c;
        }
    }

    return num_cols;
}

void save_atrib_names(parser_result *data,FILE *fp){
    rewind(fp);

    int num = 0;
    char c;
    char name[MAX_LEN];
    memset(name,'\0',MAX_LEN-1);
    int c_count = 0;

    int total_atrib = data->num_in +data->num_out;

    data->atrib_names = (char **) malloc(sizeof(char*)*total_atrib);
    for (int i = 0; i < total_atrib; i++)
    {
        data->atrib_names[i] = (char*) malloc(sizeof(char)*MAX_LEN);
    }

    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            strcpy(data->atrib_names[num],name);
            break;
        }
        else if (c== ','){
            c_count = 0;
            strcpy(data->atrib_names[num],name);
            num++;
            memset(name,'\0',MAX_LEN-1);
        }
        else{
            name[c_count++] = c;
        }
    }
}

int get_number_atributes(char filename[]){
    FILE * fp = fopen(filename,"r");

    if (fp == NULL) {
        fprintf(stderr,"ERROR: El programa fallo al abrir el fichero %s\n",filename);
        fprintf(stderr,"       compruebe si el fichero existe\n");
        exit(EXIT_FAILURE);
    }
    return get_number_cols(fp);
}

double get_nextf(FILE*fp){
    char c;
    char num[100];
    memset(num,'\0',99);
    int count = 0;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n' || c == ',') {
            return (double)atof(num);
        }
        else{
            num[count] = c;
            count++;
        }
    }
    return (double)atof(num);
}

double ** array_alloc(int rows, int cols){
    double** array = malloc(rows*sizeof(double*));
    if (array == NULL) {
        perror("Error al asignar memoria para el array de arrays");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++)
    {
        array[i] = (double*)malloc(cols * sizeof(double));
        if (array[i] == NULL) {
            perror("Error al asignar memoria para el array de doubles");
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            exit(EXIT_FAILURE);
        }
    }
    return array;
}

parser_result parse_data(char filename[],int num_inputs){
    FILE * fp = fopen(filename,"r");

    if (fp == NULL) {
        fprintf(stderr,"ERROR: El programa fallo al abrir el fichero %s\n",filename);
        fprintf(stderr,"       compruebe si el fichero existe\n");
        exit(EXIT_FAILURE);
    }
    int num_lines = get_number_lines(fp);
    fseek(fp,0,SEEK_SET);

    int num_atrib = get_number_cols(fp);

    parser_result res;
    res.num_case = num_lines;
    res.num_in = num_inputs; 
    res.num_out = num_atrib - num_inputs;

    save_atrib_names(&res,fp);

    res.data_input = array_alloc(num_lines,num_inputs);
    res.data_output = array_alloc(num_lines,res.num_out);
    

    for (int j = 0; j < num_lines; j++)
    {
        for (int i = 0; i < num_atrib; i++)
        {
            if(i <num_inputs){
                res.data_input[j][i] = get_nextf(fp);
            }
            else{
                res.data_output[j][i-num_inputs] = get_nextf(fp);
            }
        }
    }
    
    return res;
}

double ** hard_copy(double ** original, int rows, int cols){
    double** array = malloc(rows*sizeof(double*));
    if (array == NULL) {
        perror("Error al asignar memoria para el array de arrays");
        exit(1);
    }
    for (int i = 0; i < rows; i++)
    {
        array[i] = (double*)malloc(cols * sizeof(double));
        if (array[i] == NULL) {
            perror("Error al asignar memoria para el array de doubles");
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < cols; j++) {
                array[i][j] = original[i][j];
            }
    }
    return array;
}

void array_free(int** array, int filas) {
    for (int i = 0; i < filas; i++) {
        free(array[i]);
    }
    free(array);
}

parser_result parser_result_copy(parser_result old){
    parser_result new;
    new.num_case = old.num_case;
    new.num_in = old.num_in;
    new.num_out = old.num_out;

    new.data_input = hard_copy(old.data_input,old.num_case,old.num_in);
    new.data_output = hard_copy(old.data_output,old.num_case,old.num_out);

    return new;
}

int rand_num(int x, int y) {
    srand(time(NULL));
    
    int num = rand() % (y - x + 1) + x;
    
    return num;
}

void hard_copy_array(double* old, double* new,int size){
    for (int i = 0; i < size; i++)
    {
        new[i] = old[i];
    }
}

void fisher_yates_suffle(parser_result data){
    int tail_pointer = data.num_case-2;
    double * temp1 = malloc(sizeof(double)*data.num_in);
    double * temp2 = malloc(sizeof(double)*data.num_out);
    int pointer = 0;
    while (tail_pointer > 0)
    {
        pointer = (int)rand_num(0,(int)tail_pointer);

        hard_copy_array(data.data_input[pointer],temp1,data.num_in);
        hard_copy_array(data.data_output[pointer],temp2,data.num_out);
        
        hard_copy_array(data.data_input[tail_pointer],data.data_input[pointer],data.num_in);
        hard_copy_array(data.data_output[tail_pointer],data.data_output[pointer],data.num_out);

        hard_copy_array(temp1,data.data_input[tail_pointer],data.num_in);
        hard_copy_array(temp2,data.data_output[tail_pointer],data.num_out);

        tail_pointer--;
    }
    free(temp1);
    free(temp2);
}

parser_result* data_div(parser_result in, int div_size){
    parser_result *suffled = malloc(2*sizeof(parser_result));

    suffled[0].data_input = array_alloc(div_size,in.num_in);
    suffled[0].data_output = array_alloc(div_size,in.num_out);
    suffled[0].num_case = div_size;
    suffled[0].num_in = in.num_in;
    suffled[0].num_out = in.num_out;

    suffled[1].data_input = array_alloc(in.num_case - div_size,in.num_in);
    suffled[1].data_output = array_alloc(in.num_case - div_size,in.num_out);
    suffled[1].num_case = in.num_case - div_size;
    suffled[1].num_in = in.num_in;
    suffled[1].num_out = in.num_out;

    parser_result temp = parser_result_copy(in);

    fisher_yates_suffle(temp);
    
    for (int i = 0; i < div_size; i++)
    {
        hard_copy_array(temp.data_input[i],suffled[0].data_input[i],in.num_in);
        hard_copy_array(temp.data_output[i],suffled[0].data_output[i],in.num_out);
    }
    for (int i = 0; i < in.num_case-div_size; i++)
    {
        hard_copy_array(temp.data_input[div_size + i],suffled[1].data_input[i],in.num_in);
        hard_copy_array(temp.data_output[div_size + i],suffled[1].data_output[i],in.num_out);
    }
    
    // for (size_t i = 0; i < div_size; i++)
    // {
    //     printf("%f, %f, %f\n",suffled[0].data_input[i][0],suffled[0].data_input[i][1],suffled[0].data_output[i][0]);
    // }

    return suffled;
}

parser_result random_trim(parser_result data,int size){
    parser_result new_data;
    new_data.num_case = size;
    new_data.num_in = data.num_in;
    new_data.num_out = data.num_out;

    new_data.data_input = array_alloc(size,data.num_in);
    new_data.data_output = array_alloc(size,data.num_out);

    for (int i = 0; i < size; i++)
    {
        int r_index = rand() % data.num_case;
        new_data.data_input[i] = data.data_input[r_index];
        new_data.data_output[i] = data.data_output[r_index];
    }

    return new_data;
    
}

void change_all_values_for(double ** data,int size_x, int size_y,float actual, float new){
    for (int i = 0; i < size_y; i++)
    {
        for (int j = 0; j < size_x; j++)
        {
            if(fabs(data[i][j]-actual) < EPSILON){
                data[i][j] = new;
            }
        }
    }
}