// Code for multiscale edge detection using openMP (ACC) within MPI ranks
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019
// Image reading/writing code is courtesy of this open source library:
// https://github.com/nothings/stb

#include <iostream>
#include <chrono>
#include <complex>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "fft.h"
#include "utilities.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using std::cout;
using std::endl;
// Parameters
const uint8_t EDGE_THRESHOLD = 200; // only pixels with gradients larger than this marked as edges

// Forward declarations
void findMultiscaleEdgesFFT(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc);
void findMultiscaleEdges(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc);
void findEdges(uint8_t *input, uint8_t *output, int ny, int nx, int nc);
void findEdgesFFT(uint8_t *input, uint8_t *output, int ny, int nx, int nc);

// Main execution function
int main(int argc, char ** argv) {
    #pragma acc init
    if (argc != 2) {cout << "Usage: ./edgeDetect [imagefile.jpg]" << endl;return 0;}     // Check for correct usage

    // TO DO LIST:
    // Run with ACC only - be careful about unnecessary memcopy's
    // Run with MPI only (note kernels are arbitrarily sized, so need to be smart about the boundaries!)
    // Run with the FFT and multiply method



    // Print out call sequence that was used
    cout << "Call sequence: ";
    for (int i=0; i<argc; ++i) cout << argv[i] << " ";
    cout << endl;

    // Read in the image we will do edge detection on using stb library
    int nx, ny, nc;
    uint8_t * image = stbi_load(argv[1], &nx, &ny, &nc, NCOLORS); // NCOLORS forces NCOLORS channels per pixel
    cout << "Successfully read " << argv[1] << endl;
    cout << "(nx,ny,nChannels) = (" << nx << "," << ny << "," <<  nc << ")" << endl;


    // Convert image to greyscale for edge detection
    uint8_t * image_gray = new uint8_t [NCOLORS*nx*ny]; // same size as image but only one color channel
    for (long i=0;i<NCOLORS*nx*ny;++i) image_gray[i] = 0;
    cout << "Converting to grayscale...";
    Grayscale(image, image_gray, ny, nx, nc);
    cout << "Done" << endl << endl;

    // Allocate edgemap
    uint8_t * edges = new uint8_t [nx*ny]; // same size as image but only one color channel
    for (long i=0;i<nx*ny;++i) edges[i] = 0;


    //=================FFT SINGLE VERSION======================
    // Get the starting timestamp. 
    Time begin_time_f = std::chrono::steady_clock::now();


    // Run edge detection function
    cout << "Running edge detection with FFT...";
    findEdgesFFT(image_gray, edges, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time end_time_f = std::chrono::steady_clock::now(); 
    DeltaTime dt_f = end_time_f - begin_time_f; // Compute the difference.
    printf("FFT Edge detection runtime was %.10f seconds\n", dt_f.count());

    // Write out resulting edgemap
    stbi_write_jpg("edgesFFT.jpg", nx, ny, 1, edges, JPG_QUALITY);
    cout << "Wrote edgesFFT.jpg" << endl << endl;

/*
    //=================SERIAL SINGLE VERSION====================
    // Get the starting timestamp. 
    Time begin_time = std::chrono::steady_clock::now();


    // Run edge detection function
    cout << "Running edge detection serial...";
    findEdges(image_gray, edges, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time end_time = std::chrono::steady_clock::now(); 
    DeltaTime dt = end_time - begin_time; // Compute the difference.
    printf("Serial edge detection runtime was %.10f seconds\n", dt.count());

    // Write out resulting edgemap
    stbi_write_jpg("edges.jpg", nx, ny, 1, edges, JPG_QUALITY);
    cout << "Wrote edges.jpg" << endl << endl;
*/




    // =================================================================================================== //
    // MULTISCALE EDGE DETECTION
    int nlevels = 5;
    int levels [nlevels];
    levels[0]=1;
    levels[1]=2;
    levels[2]=4;
    levels[3]=6;
    levels[4]=8;

    // Allocate multiscale edgemaps
    uint8_t ** multiscaleEdges = new uint8_t * [nlevels];
    for (int l=0;l<nlevels;++l) multiscaleEdges[l] = new uint8_t [nx*ny/(levels[l]*levels[l])];

    // =============FFT MULTISCALE======================

    // Get the starting timestamp. 
    Time mbegin_time_f = std::chrono::steady_clock::now();

    // Run multiscale edge detection
    cout << "Running multiscale edge detection with FFT...";
    findMultiscaleEdgesFFT(image_gray, multiscaleEdges, levels, nlevels, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time mend_time_f = std::chrono::steady_clock::now(); 
    DeltaTime mdt_f = mend_time_f - mbegin_time_f; // Compute the difference.
    printf("Multiscale FFT Edge detection runtime was %.10f seconds\n", mdt_f.count());

    // Write out multiscale edgemap images
    uint8_t * enlargedEdges_f = new uint8_t [ny*nx];
    for (int i=0;i<ny*nx;++i) enlargedEdges_f[i] = 0;
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        enlarge(multiscaleEdges[l], enlargedEdges_f, ny/factor, nx/factor, 1, factor);
        char edgeOutfile [20]; 
        sprintf(edgeOutfile,"edges_FFT_%dx.jpg", factor);
        stbi_write_jpg(edgeOutfile, nx, ny, 1, enlargedEdges_f, JPG_QUALITY);
        cout << "Wrote " << edgeOutfile << endl;
    }
    cout << endl;


/*
    // ===============SERIAL MULTISCALE==================

    // Get the starting timestamp. 
    Time mbegin_time = std::chrono::steady_clock::now();

    // Run multiscale edge detection
    cout << "Running multiscale edge detection...";
    findMultiscaleEdges(image_gray, multiscaleEdges, levels, nlevels, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time mend_time = std::chrono::steady_clock::now(); 
    DeltaTime mdt = mend_time - mbegin_time; // Compute the difference.
    printf("Multiscale Edge detection runtime was %.10f seconds\n", mdt.count());

    // Write out multiscale edgemap images
    uint8_t * enlargedEdges = new uint8_t [ny*nx];
    for (int i=0;i<ny*nx;++i) enlargedEdges[i] = 0;
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        enlarge(multiscaleEdges[l], enlargedEdges, ny/factor, nx/factor, 1, factor);
        char edgeOutfile [20]; 
        sprintf(edgeOutfile,"edges_%dx.jpg", factor);
        stbi_write_jpg(edgeOutfile, nx, ny, 1, enlargedEdges, JPG_QUALITY);
        cout << "Wrote " << edgeOutfile << endl;
    }
    cout << endl;
*/



    // ==================================================================

    // Cleanup
    stbi_image_free(image);  
    delete [] edges; 
    delete [] image_gray;
    for (int i=0;i<nlevels;++i) delete [] multiscaleEdges[i];
    delete [] multiscaleEdges; 
    //delete [] enlargedEdges;
    delete [] enlargedEdges_f;
    return 0;
}



// Find edges at various coarser resolution levels. Output must be preallocated.
void findMultiscaleEdgesFFT(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc) {

    // Find edges at each of the downsampling levels in levels array and place into output
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        // Shrink image to smaller level
        uint8_t *small_img = new uint8_t [ny*nx*nc/(factor*factor)];
        shrink(input, small_img, ny, nx, nc, factor);

        // Detect edges of the shrunk image
        findEdgesFFT(small_img, output[l], ny/factor, nx/factor, nc);
        
        delete [] small_img;
    }

}

// Find edges at various coarser resolution levels. Output must be preallocated.
void findMultiscaleEdges(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc) {

    // Find edges at each of the downsampling levels in levels array and place into output
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        // Shrink image to smaller level
        uint8_t *small_img = new uint8_t [ny*nx*nc/(factor*factor)];
        shrink(input, small_img, ny, nx, nc, factor);

        // Detect edges of the shrunk image
        findEdges(small_img, output[l], ny/factor, nx/factor, nc);
        
        delete [] small_img;
    }

}


// Find the edges in the image at the current resolution using the input kernel (size nkx-by-nky), 
// Output must be preallocated and the same size as input.
void findEdgesFFT(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {

    const int ksize = 3; // this is about 30 sec regardless
    //const int ksize = 65; // this is about 30 sec regardless
    // Allocate kernel
    Complex **GX = new Complex * [ksize];
    for (int i=0;i<ksize;++i) GX[i] = new Complex [ksize];
    for (int j=0;j<ksize;++j) for (int i=0;i<ksize;++i) GX[j][i] = Complex(0,0);  
    Complex **GY = new Complex * [ksize];
    for (int i=0;i<ksize;++i) GY[i] = new Complex [ksize];
    for (int j=0;j<ksize;++j) for (int i=0;i<ksize;++i) GY[j][i] = Complex(0,0);  

    //Sobel Horizontal Mask     
    GX[0][0] = Complex(1,0); GX[0][1] = Complex(0,0); GX[0][2] = Complex(-1,0); 
    GX[1][0] = Complex(2,0); GX[1][1] = Complex(0,0); GX[1][2] = Complex(-2,0);  
    GX[2][0] = Complex(1,0); GX[2][1] = Complex(0,0); GX[2][2] = Complex(-1,0);

    //Sobel Vertical Mask   
    GY[0][0] =  Complex(1,0);  GY[0][1] = Complex(2,0); GY[0][2] =  Complex(1,0);    
    GY[1][0] =  Complex(0,0);  GY[1][1] = Complex(0,0); GY[1][2] =  Complex(0,0);    
    GY[2][0] =  Complex(-1,0); GY[2][1] =-Complex(2,0); GY[2][2] =  Complex(-1,0);

    cout << "Kernel size: " << ksize << endl;


    // Allocate full image results and copy in the input pixels since the FFT is done in place
    // Also put into complex.
    Complex **EDGESX = new Complex * [ny];
    for (int i=0;i<ny;++i) EDGESX[i] = new Complex [nx];
    for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) EDGESX[j][i] = pixels[j*nx*nc+i*nc] + 0.0 * 1i;
    Complex **EDGESY = new Complex * [ny];
    for (int i=0;i<ny;++i) EDGESY[i] = new Complex [nx];
    for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) EDGESY[j][i] = pixels[j*nx*nc+i*nc] + 0.0 * 1i;

    // x-direction convolution
    FFTImageConvolution(EDGESX, ny, nx, GX, 3, 3);

    // y-direction convolution
    FFTImageConvolution(EDGESY, ny, nx, GY, 3, 3);



    #pragma acc data copyout(output[0:nx*ny]) create(MAG) copyin(EDGE_THRESHOLD) present(nx) present(ny) copyin(TMPX[0:ny][0:nx]) copyin(TMPY[0:ny][0:nx])
    #pragma acc parallel loop 
    for(int i=0; i < ny; i++)
    {
        #pragma acc loop independent 
        for(int j=0; j < nx; j++)
        {
            //Gradient magnitude
            double MAG = sqrt(EDGESX[i][j].real()*EDGESX[i][j].real() + EDGESY[i][j].real()*EDGESY[i][j].real());
            //double MAG = pixels[i*nx*nc+j*nc];

            // Apply threshold to gradient  
            uint8_t MAGuint8;
            if (MAG > EDGE_THRESHOLD) MAGuint8 = 255; else MAGuint8 = 0;
            
            //setting the new pixel value
            output[yxc(i,j,0,nx,1)] = MAGuint8;
            
        }
    }
    
    for (int i=0;i<ny;++i) delete [] EDGESX[i];
    for (int i=0;i<ny;++i) delete [] EDGESY[i];
    for (int i=0;i<3;++i)  delete [] GX[i];
    for (int i=0;i<3;++i)  delete [] GY[i];
    delete [] EDGESX;
    delete [] EDGESY;
    delete [] GY;
    delete [] GX;

    return;
}



// Find the edges in the image at the current resolution using the input kernel (size nkx-by-nky), 
// Output must be preallocated and the same size as input.
void findEdges(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {
    

    const int ksize = 3;
    //const int ksize = 65; // 101: 65s, 75: 41s, 65: 30s. Comparable to FFT @ 65, but output is nearly meaningless at that level
    int **GX = new int * [ksize];
    for (int i=0;i<ksize;++i) GX[i] = new int [ksize];
    for (int j=0;j<ksize;++j) for (int i=0;i<ksize;++i) GX[j][i] = 0; 
    int **GY = new int * [ksize];
    for (int i=0;i<ksize;++i) GY[i] = new int [ksize];
    for (int j=0;j<ksize;++j) for (int i=0;i<ksize;++i) GY[j][i] = 0;

    //static int GX [ksize][ksize]; static int GY [ksize][ksize];

    //Sobel Horizontal Mask     
    GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1; 
    GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;  
    GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

    //Sobel Vertical Mask   
    GY[0][0] =  1; GY[0][1] = 2; GY[0][2] =   1;    
    GY[1][0] =  0; GY[1][1] = 0; GY[1][2] =   0;    
    GY[2][0] = -1; GY[2][1] =-2; GY[2][2] =  -1;

    cout << "Kernel size: " << ksize << endl;

    //Two arrays to store values for parallelization purposes
    int **TMPX = new int *[ny];
    int **TMPY = new int *[ny];
    for (int i = 0; i < ny; i++) {
        TMPY[i] = new int[nx];
        TMPX[i] = new int[nx];
    }
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            TMPY[i][j] = 0;
            TMPX[i][j] = 0;
        }
    }


    int valX,valY,MAG;
    #pragma acc data copyin(pixels[0:nx*ny*nc]) copyin(GX[0:3][0:3]) copyin(GY[0:3][0:3]) copy(TMPX[0:ny][0:nx]) copy(TMPY[0:ny][0:nx]) copyin(nx) copyin(ny) copyin(nc)
    {
    #pragma acc parallel loop
    for(int j=0; j < ny; j++)
    {
        #pragma acc loop independent 
        for(int i=0; i < nx; i++)
        {
            //setting the pixels around the border to 0, because the Sobel kernel cannot be allied to them
            if ((j<ksize/2)||(j>(ny-ksize/2))||(i<(ksize/2))||(i>(nx-ksize/2))) {TMPX[j][i] = 0; TMPY[j][i]= 0;}
            else
            {
                for (int kj=0;kj<ksize;++kj) {
                    for (int ki=0;ki<ksize;++ki) {
                        TMPY[j][i] +=  pixels[yxc(j+kj-ksize/2,i+ki-ksize/2,0,nx,nc)]* GY[kj][ki];
                        TMPX[j][i] +=  pixels[yxc(j+kj-ksize/2,i+ki-ksize/2,0,nx,nc)]* GX[kj][ki];
                    }
                }
            }
        }
    }
    }
    #pragma acc data copyout(output[0:nx*ny]) create(MAG) copyin(EDGE_THRESHOLD) copyin(nx) copyin(ny) copyin(TMPX[0:ny][0:nx]) copyin(TMPY[0:ny][0:nx])
    #pragma acc parallel loop 
    for(int i=0; i < ny; i++)
    {
        #pragma acc loop independent 
        for(int j=0; j < nx; j++)
        {
            //Gradient magnitude
            MAG = sqrt(TMPX[i][j]*TMPX[i][j] + TMPY[i][j]*TMPY[i][j]);

            // Apply threshold to gradient
            if (MAG > EDGE_THRESHOLD) MAG = 255; else MAG = 0;
            
            //setting the new pixel value
            output[yxc(i,j,0,nx,1)] = MAG;
            
        }
    }
    
    for (int i=0;i<ny;++i) delete [] TMPY[i];
    for (int i=0;i<ny;++i) delete [] TMPX[i];
    for (int i=0;i<3;++i)  delete [] GX[i];
    for (int i=0;i<3;++i)  delete [] GY[i];
    delete[] TMPY;
    delete[] TMPX;
    delete [] GY;
    delete [] GX;

    return;
}
