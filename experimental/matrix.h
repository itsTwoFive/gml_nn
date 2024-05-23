// This matrix.h code version differs from the finals version code founded in src

#define MIN_R 0.0f
#define MAX_R 1.0f

typedef struct {
    int cols;
    int rows;
    float *d;
}matrix;

matrix mat_alloc(int rows, int cols);

void mat_free(matrix* mat);

float* mat_seek(matrix m,int i,int j);

void mat_set_number(matrix m, int i, int j, float value);

void mat_print(matrix m);

void mat_randf(matrix m);

void mat_sumf(matrix m1, matrix m2, matrix *result);

void mat_subsf(matrix m1, matrix m2, matrix *result);

void mat_productf(matrix m1, matrix m2, matrix *result);

void mat_dot_productf(matrix m, float coeficent, matrix *result);

float mat_column_sum(matrix m, int col);

float mat_row_sum(matrix m, int row);

void mat_set_zeros(matrix m);