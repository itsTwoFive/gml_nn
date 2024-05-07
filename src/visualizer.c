#include "visualizer.h"

#include <stdio.h>
#include <stdlib.h>

void plot2DDataForBinary(double** data_in,double** data_out, int num_cases,int num_out){
    
    FILE *pipe = popen("gnuplot -persist", "w"); // Abrir un pipe a Gnuplot

    // Enviar comandos a Gnuplot para plotear
    fprintf(pipe, "set title 'Input Data'\n");
    //fprintf(pipe, "set terminal x11\n");
    fprintf(pipe, "set terminal png\n");
    fprintf(pipe, "set output 'plot1.png'\n");
    fprintf(pipe, "plot '-' w circles lw 1.5");
    for (int i = 0; i < num_out-1; i++)
    {
        fprintf(pipe, ", '-' w circles lw 1.5");
    }
    fprintf(pipe, "\n");
     fflush(pipe);

    for (int j = 0; j < num_out; j++)
    {
        for (int i = 0; i < num_cases; i++)
        {
            if (data_out[i][j] > 0.5) fprintf(pipe, "%f %f\n", data_in[i][0], data_in[i][1]);
        }
        fprintf(pipe,"e\n");
        fflush(pipe);  
    }

    // Cerrar el pipe
    pclose(pipe);

}

void showAreas2DPlot(neural_net nn,int num_out){
    double step = .05;
    
    FILE *pipe = popen("gnuplot -persist", "w"); // Abrir un pipe a Gnuplot

    fprintf(pipe, "set title 'Zones'\n");
    // fprintf(pipe, "set terminal x11\n");
    fprintf(pipe, "set terminal png\n");
    fprintf(pipe, "set output 'plot2.png'\n");
    fprintf(pipe, "plot '-' w p lw 1.5");
    for (int i = 0; i < num_out-1; i++)
    {
        fprintf(pipe, ",  '-' w p lw 1.5");
    }
    fprintf(pipe, "\n");
    fflush(pipe);

    for (int j = 0; j < num_out; j++)
    {
        for (double x = -2; x < 4; x +=step)
        {
            for (double y = -2; y < 4; y+=step)
            {
                double punto[] = {x,y};
                double * out =mat_toarray(*feed_forward(nn,punto,2));
                if (j == choose_class(out,num_out,1))
                {
                    fprintf(pipe, "%f %f\n", x, y);
                }
                free(out);
            }
        }
        fprintf(pipe,"e\n");
        fflush(pipe);  
    }

    // Cerrar el pipe
    pclose(pipe);
    
}