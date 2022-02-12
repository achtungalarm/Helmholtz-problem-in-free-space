#include<stdlib.h>
#include<stdio.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_complex_math.h>
#include<gsl/gsl_complex.h>
#include<gsl/gsl_sf.h>
#include<time.h>
#include<omp.h> 

#define pi M_PI

// Compute the complex signal of frequency w received at X, Y from a source located at X0, Y0
gsl_complex green(double w, double X, double Y, double X0, double Y0){
    double dX = X - X0, dY = Y - Y0;
    double R = sqrt(dX * dX + dY * dY);
    return gsl_complex_div(gsl_complex_exp(gsl_complex_rect(0., -w * R)), gsl_complex_rect(4 * pi * R, 0));
}

// Compute the complex signal of frequency w received ont the grid given by X x Y emitted by the source located at X0, Y0
void green_on_mesh(gsl_matrix_complex *G, double w, double *X, double *Y, double X0, double Y0, int N){
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            gsl_matrix_complex_set(G, i, j, green(w, X[i], Y[j], X0, Y0));
        }
    }
}

// Compute the complex signal of frequency w received ont the grid given by X x Y emitted by the sources located at X0, Y0
void green_plural_source(gsl_matrix_complex *G, double w, double *X, double *Y, double *X0, double *Y0, int N, int Ns){
    gsl_matrix_complex *temp = gsl_matrix_complex_alloc(N, N);
    for (int i=0; i<Ns; i++){
        green_on_mesh(temp, w, X, Y, X0[i], Y0[i], N);
        gsl_matrix_complex_add(G, temp);
    }
}

// Extrcat log(abs(G)), real part of G, imaginary part of G and phase of G in the dedicated matrix
void green_extract(gsl_matrix *GA, gsl_matrix *Gr, gsl_matrix *Gi, gsl_matrix *Gp, gsl_matrix_complex *G, int N, int Ns){
    gsl_complex temp;
    gsl_matrix_complex_scale(G, gsl_complex_rect(1/(double) Ns, 0.));
    int i, j;
    #pragma omp parallel for private(i, j, temp)
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            temp = gsl_matrix_complex_get(G, i, j);
            gsl_matrix_set(GA, i, j, gsl_complex_logabs(temp));
            gsl_matrix_set(Gr, i, j, GSL_REAL(temp));
            gsl_matrix_set(Gi, i, j, GSL_IMAG(temp));
            gsl_matrix_set(Gp, i, j, gsl_complex_arg(temp));
        }
    }
}

// Save the data of G matrix in ascii, such that gnuplot with matrix non-uniform can lpot it
void write_matrix(gsl_matrix *G, int N, FILE *f){
    fprintf(f, "%d ", N);
    for (int i=1; i<=N; i++){
        fprintf(f, "%d ", i);
    }
    fprintf(f, "\n");
    for (int j=0; j < N; j++){
        fprintf(f, "%d ", j+1);
        for (int i=0; i<N; i++){
            fprintf(f, "%.5lf ", gsl_matrix_get(G, i, j));
        }
        fprintf(f, "\n");
    }
}

// Function to spread sources given a special distribution
void source_coord(int c, double *X0, double *Y0, int Ns){
    double x0mi = -1, x0ma = 1, y0mi = -1, y0ma = 1;
    double Dx0, Dy0, dx0, dy0;
    switch(c){
        case 1: // Line
            Dy0 = y0ma - y0mi, dy0 = Dy0 / (double)(Ns-1);
            for (int i=0; i<Ns; i++){
                X0[i] = 0;
                Y0[i] = y0mi + i * dy0;
            }
            break;
        case 2: // Cross
            Dx0 = x0ma - x0mi, dx0 = 2*Dx0 / (double)(Ns-1);
            Dy0 = y0ma - y0mi, dy0 = 2*Dy0 / (double)(Ns-1);
            for (int i=0; i<Ns/2; i++){
                X0[i] = 0;
                Y0[i] = y0mi + i * dy0;
            }
            for (int i=0; i<Ns/2; i++){
                X0[Ns/2+i] = x0mi + i * dx0;
                Y0[Ns/2+i] = 0;
            }
            break;
        case 3: // Triangle
            double s = 0;
            double ds = 3. / (double) Ns;
            double temp = sqrt(15);
            for (int i=0; i<Ns; i++){
                if (s < 1){
                    X0[i] = -s;
                    Y0[i] = (1 - temp) * s + 1;
                }
                else if (s < 2){
                    X0[i] = 2 * s - 3;
                    Y0[i] = 2 - temp;
                }
                else {
                    X0[i] = 3 - s;
                    Y0[i] = 4 - 3 * temp + (temp - 1) * s;
                }
                s += ds;
            }
            break;
        case 4: // Arc of a circle
            double R = 5.;
            double theta_d = pi - GSL_REAL(gsl_complex_arcsin_real(1. / R));
            double theta_f = pi + GSL_REAL(gsl_complex_arcsin_real(1. / R));
            double dtheta = (theta_f - theta_d) / (double)(Ns-1);
            for (int i=0; i<Ns; i++){
                X0[i] = R * gsl_sf_cos(theta_d + i * dtheta)+sqrt(R * R - 1) ;
                Y0[i] = R * gsl_sf_sin(theta_d + i * dtheta);
            }
            break;
        case 5: // Arc of a parabola
            double F = 1.;
            double a = sqrt(F * F + 1.) / 2. - F / 2.;
            Dy0 = y0ma - y0mi, dy0 = Dy0 / (double)(Ns-1);
            for (int i=0; i<Ns; i++){
                Y0[i] = y0mi + i * dy0;
                X0[i] = a * (Y0[i] * Y0[i] - 1);
            }
            break;
    }
}

// Creation of the initial grid
void init_grid(double *X, double *Y, int N){
    double xmin = -10, xmax = 10, Dx = xmax - xmin, dx = Dx / (double)(N-1);
    double ymin = -10, ymax = 10, Dy = ymax - ymin, dy = Dy / (double)(N-1);
    for (int i=0; i<N; i++){
        X[i] = xmin + i * dx;
        Y[i] = ymin + i * dy;
    }
}

int main(int argc, char *argv[]){
    double w = 10.;
    int N = 512, Ns = 256;
    int cas = 3;

    gsl_matrix_complex *G = gsl_matrix_complex_alloc(N, N);
    gsl_matrix *GA = gsl_matrix_alloc(N, N); // log(abs(G))
    gsl_matrix *Gr = gsl_matrix_alloc(N, N); // Re(G)
    gsl_matrix *Gi = gsl_matrix_alloc(N, N); // Im(G)
    gsl_matrix *Gp = gsl_matrix_alloc(N, N); // arg(G)

    double *X, *Y, *X0, *Y0;

    X = malloc(sizeof(double) * N);     // abscissa of the grid
    Y = malloc(sizeof(double) * N);     // ordinate of the grid
    X0 = malloc(sizeof(double) * N);    // abscissa of the sources
    Y0 = malloc(sizeof(double) * N);    // ordinate of the sources

    init_grid(X, Y, N);
    source_coord(cas, X0, Y0, Ns);

    clock_t t;
    t = clock();
    green_plural_source(G, w, X, Y, X0, Y0, N, Ns); // Main computation
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("green_plural_source() took %.5lf seconds to execute (using C)\n", time_taken / (double) omp_get_num_procs());

    // Post-processing

    green_extract(GA, Gr, Gi, Gp, G, N, Ns);
    FILE *fa = fopen("abs.dat", "w");
    FILE *fr = fopen("real.dat", "w");
    FILE *fi = fopen("imag.dat", "w");
    FILE *fp = fopen("phase.dat", "w");
    write_matrix(GA, N, fa);
    write_matrix(Gr, N, fr);
    write_matrix(Gi, N, fi);
    write_matrix(Gp, N, fp);
    fclose(fa); fclose(fr); fclose(fi); fclose(fp);

    free(X); free(Y); free(X0); free(Y0);
    gsl_matrix_complex_free(G);
    gsl_matrix_free(GA); gsl_matrix_free(Gr); gsl_matrix_free(Gi); gsl_matrix_free(Gp);

    return 0;
}