// Utility functions for dealing with images
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019

#include "utilities.h"


// Converts image into greyscale (maintaining 3 color channels)
void Grayscale(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {
    //Calculating the grayscale in each pixel. 
    int val1;
    //The values of the 3 colours (R, B and G) are all the same  
    #pragma acc data copyin(pixels[0:nx*ny*nc]) copyin(ny) copyin(nx) copyin(nc) copy(output[0:nx*ny*nc]) create(val1)
    #pragma acc parallel loop 

    for(int i=0; i < ny; i++)
        {
            #pragma acc loop independent 
            for(int j=0; j < nx; j++)
            {
                val1 = pixels[yxc(i,j,0,nx,nc)];
                output[yxc(i,j,0,nx,nc)] = val1;
                output[yxc(i,j,1,nx,nc)] = val1;
                output[yxc(i,j,2,nx,nc)] = val1;
            }
        }

}

// Shrink input by an integer factor using a simple average pooling. Output must be allocated already.
// nx and ny are the sizes of the larger input image.
void shrink(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor) {
    // Loop over every pixel in the smaller output image, averaging over the nearest "factor" pixels
    // in each direction in the input image.
    // Note: if nx or ny is not evenly divisible by factor, this will leave the rightmost and/or
    // bottommost pixels unaveraged
    int nysml = ny/factor;
    int nxsml = nx/factor;
    uint32_t value = 0;
    #pragma acc data copyin(input[0:nx*ny*nc]) copyin(ny) copyin(nx) copyin(nc) copy(output[0:nysml*nxsml*nc]) create(value) copyin(factor) copyin(nxsml) copyin(nysml)
    {
    #pragma acc parallel loop 
    for (int ysml=0;ysml<nysml;++ysml) { // loop over columns in output
        //#pragma acc loop independent 
        for (int xsml=0;xsml<nxsml;++xsml) { // loop over rows in output
            //#pragma acc loop independent 
            for (int c=0;c<nc;++c) { // loop over color channels
                value = 0;
                for (int yf=0;yf<factor;++yf) { // loop over col pixels within pool
                    for (int xf=0;xf<factor;++xf) { // loop over row pixels within pool
                        value += input[yxc(ysml*factor+yf,xsml*factor+xf,c,nx,nc)];
                    }
                }
                output[yxc(ysml,xsml,c,nxsml,nc)] = value/(factor*factor);
            }
        }
    }
    }

    return;
}

// Enlarge the image by an integer factor by simply copying (maybe do interpolation at some point)
// Output must be allocated already.
void enlarge(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor) {
    // Loop over every pixel in the smaller input image and replicate into the larger image
    int nylrg = ny*factor;
    int nxlrg = nx*factor;
    //#pragma acc data copyin(input[0:nx*ny]) copyin(nx) copyin(nc) copy(output[0:nylrg*nxlrg]) copyin(factor) copyin(nxlrg) copyin(nylrg)
    //#pragma acc parallel loop 
    for (int y=0;y<nylrg;++y) { // loop over pixels in the large image
        //#pragma acc loop independent 
        for (int x=0;x<nxlrg;++x) {
        //#pragma acc loop independent 
            for (int c=0;c<nc;++c) { // loop over colors
                output[yxc(y,x,c,nxlrg,nc)] = input[yxc(y/factor,x/factor,c,nx,nc)];
            }
        }
    }

    return;
}
