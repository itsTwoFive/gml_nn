#define MIN_R -1.0f
#define MAX_R 1.0f

typedef struct {
    int cols;
    int rows;
    double *d;
}matrix;

matrix *mat_alloc(int rows, int cols);

void mat_free(matrix* mat);

double* mat_seek(matrix m,int i,int j);

void mat_set_number(matrix m, int i, int j, double value);

void mat_print(matrix m);

void mat_randf(matrix m);

void mat_sumf(matrix m1, matrix m2, matrix *result);

void mat_subsf(matrix m1, matrix m2, matrix *result);

void mat_productf(matrix m1, matrix m2, matrix *result);

void mat_dot_productf(matrix m, double coeficent, matrix *result);

double mat_column_sum(matrix m, int col);

void mat_set_zeros(matrix m);

matrix* mat_fromarray(int lenght,double array[]);