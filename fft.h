// Fast fourier transform functions (taken from earlier HW assignment)
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019
#ifndef FFT_H
#define FFT_H

#include <iostream>
#include <fstream>


#include <cmath>
#include <complex>
using namespace std;
#define PI 3.141592653589793f
#define  I  Complex(0.0f, 1.0f)
typedef complex <double> Complex;

void makePhase(Complex *omega, int N);
void FT(Complex * Ftilde, Complex * F, Complex * omega, int N);
void FTinv(Complex * F, Complex * Ftilde, Complex * omega, int N);
void downsample(double *fine, double *coarse, int Nfine, int isodd);
void FFT(Complex * F, int N);
void FFT_(Complex * F, int N);
void FFTinv(Complex * F, int N);
void FFTinv_(Complex * F, int N);
void printComplexArray(FILE *fp, Complex * F, int N);
double zeroround(double a);
void fftshift(Complex * F, int N);

void FFT2D(Complex **F, int Ny, int Nx);
void FFTinv2D(Complex **F, int Ny, int Nx);
void transpose(Complex **F, Complex ** FT, int Ny, int Nx);
void printComplexArray2D(FILE* fp, Complex ** F, int N);

void FFTImageConvolution(Complex **image, int ny, int nx, Complex **kernel, int nky, int nkx);

#endif 
