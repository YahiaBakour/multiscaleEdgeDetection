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
    #pragma acc data copyin(pixels[0:nx*ny*nc]) copyout(output[0:nx*ny*nc]) 
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

    int **TMP1 = new int *[nysml];
    int **TMP2 = new int *[nysml];
    int **TMP3 = new int *[nysml];

    for (int i = 0; i < nysml; i++) {
        TMP1[i] = new int[nxsml];
        TMP2[i] = new int[nxsml];
        TMP3[i] = new int[nxsml];
    }
    #pragma acc data present(input[0:nx*ny*nc]) create(TMP1[0:nysml][0:nxsml])  create(TMP2[0:nysml][0:nxsml])  create(TMP3[0:nysml][0:nxsml])  copyout(output[0:nysml*nxsml*nc]) 
    {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < nysml; i++) {
        for (int j = 0; j < nxsml; j++) {
            TMP1[i][j] = 0;
            TMP2[i][j] = 0;
            TMP3[i][j] = 0;
        }
    }


    #pragma acc parallel loop 
    for (int ysml=0;ysml<nysml;++ysml) { // loop over columns in output
        #pragma acc loop independent 
        for (int xsml=0;xsml<nxsml;++xsml) { // loop over rows in output
            //#pragma acc loop independent
            //Make an array of factor*factor
            //populate array with data from input[]
            //reduce array to a sum
            //populate TMP1,2,3 
                for (int yf=0;yf<factor;++yf) { // loop over col pixels within pool
                    //#pragma acc loop independent 
                    for (int xf=0;xf<factor;++xf) { // loop over row pixels within pool
                        TMP1[ysml][xsml] += input[yxc(ysml*factor+yf,xsml*factor+xf,0,nx,nc)];
                        TMP2[ysml][xsml] += input[yxc(ysml*factor+yf,xsml*factor+xf,1,nx,nc)];
                        TMP3[ysml][xsml] += input[yxc(ysml*factor+yf,xsml*factor+xf,2,nx,nc)];
                    }
                }
        }
    }
    
    
    #pragma acc parallel loop 
    for (int ysml=0;ysml<nysml;++ysml) { // loop over columns in output
    #pragma acc loop independent 
        for (int xsml=0;xsml<nxsml;++xsml) { // loop over rows in output
            output[yxc(ysml,xsml,0,nxsml,nc)] = TMP1[ysml][xsml]/(factor*factor);
            output[yxc(ysml,xsml,1,nxsml,nc)] = TMP2[ysml][xsml]/(factor*factor);
            output[yxc(ysml,xsml,2,nxsml,nc)] = TMP3[ysml][xsml]/(factor*factor);
        }
    }
    }
    
    for (int i=0;i<nysml;++i) delete [] TMP1[i];
    for (int i=0;i<nysml;++i) delete [] TMP2[i];
    for (int i=0;i<nysml;++i) delete [] TMP3[i];
   delete[] TMP1;
   delete[] TMP2;
   delete[] TMP3;
    return;
}

// Enlarge the image by an integer factor by simply copying (maybe do interpolation at some point)
// Output must be allocated already.
void enlarge(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor) {
    // Loop over every pixel in the smaller input image and replicate into the larger image
    int nylrg = ny*factor;

    int nxlrg = nx*factor;
    #pragma acc data copyin(input[0:nx*ny]) copyout(output[0:nylrg*nxlrg]) 
    #pragma acc parallel loop 
    for (int y=0;y<nylrg;++y) { // loop over pixels in the large image
        #pragma acc loop independent 
        for (int x=0;x<nxlrg;++x) {
        #pragma acc loop independent 
            for (int c=0;c<nc;++c) { // loop over colors
                output[yxc(y,x,c,nxlrg,nc)] = input[yxc(y/factor,x/factor,c,nx,nc)];
            }
        }
    }

    return;
}
