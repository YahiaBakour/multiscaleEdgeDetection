/* Code for multiscale edge detection using openMP (ACC) within MPI ranks
- Yahia Bakour and James Dunn, Boston University
- EC526 - Parallel Programming for High Performance Computing final project
- April/May 2019
- Image reading/writing code is courtesy of this open source library: https://github.com/nothings/stb
*/

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "utilities.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using std::cout;
using std::endl;

// Parameters
const uint8_t EDGE_THRESHOLD = 200; // only pixels with gradients larger than this marked as edges

// Forward declarations
void findMultiscaleEdges(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc);
void findEdges(uint8_t *input, uint8_t *output, int ny, int nx, int nc);

// Main execution function
int main(int argc, char ** argv) {
    #pragma acc init
    if (argc != 2) {cout << "Usage: ./edgeDetect [imagefile.jpg]" << endl;return 0;}     // Check for correct usage
    
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
    cout << "Done" << endl;

    // Allocate edgemap
    uint8_t * edges = new uint8_t [nx*ny]; // same size as image but only one color channel
    for (long i=0;i<nx*ny;++i) edges[i] = 0;

    // Get the starting timestamp. 
    Time begin_time = std::chrono::steady_clock::now();


    // Run edge detection function
    cout << "Running edge detection...";
    findEdges(image_gray, edges, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time end_time = std::chrono::steady_clock::now(); 
    DeltaTime dt = end_time - begin_time; // Compute the difference.
    printf("Edge detection runtime was %.10f seconds\n", dt.count());

    // Write out resulting edgemap
    stbi_write_jpg("edges.jpg", nx, ny, 1, edges, JPG_QUALITY);
    cout << "Wrote edges.jpg" << endl;




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
    for(int i = 0 ; i < nlevels; i++){

    }
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        enlarge(multiscaleEdges[l], enlargedEdges, ny/factor, nx/factor, 1, factor);
        char edgeOutfile [20]; 
        sprintf(edgeOutfile,"edges_%dx.jpg", factor);
        stbi_write_jpg(edgeOutfile, nx, ny, 1, enlargedEdges, JPG_QUALITY);
        cout << "Wrote " << edgeOutfile << endl;
    }
    
    // ==================================================================

    // Cleanup
    stbi_image_free(image);  
    delete [] edges; 
    delete [] image_gray;
    for (int i=0;i<nlevels;++i) delete [] multiscaleEdges[i];
    delete [] multiscaleEdges; 
    delete [] enlargedEdges;
    return 0;
}


// Find edges at various coarser resolution levels. Output must be preallocated.
void findMultiscaleEdges(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc) {
    #pragma acc data copyin(input[0:nx*ny*nc])
    {
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

}

// This was rewritten this way to allow for parallelization with ACC, Removed backwards dependencies.
// Find the edges in the image at the current resolution using the input kernel (size nkx-by-nky), 
// Output must be preallocated and the same size as input.
void findEdges(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {
    //Sobel Horizontal Mask     
    static int GX00 ,GX01,GX02,GX10 ,GX11,GX12,GX20 ,GX21,GX22,GY00,GY01,GY02,GY10,GY11,GY12,GY20,GY21,GY22;
    //Two arrays to store values for parallelization purposes
    int **TMPX = new int *[ny];
    int **TMPY = new int *[ny];

    for (int i = 0; i < ny; i++) {
        TMPY[i] = new int[nx];
        TMPX[i] = new int[nx];
    }


    //Sobel Horizontal Mask     
    GX00 = 1; GX01 = 0; GX02 = -1; 
    GX10 = 2; GX11 = 0; GX12 = -2;  
    GX20 = 1; GX21 = 0; GX22 = -1;

    //Sobel Vertical Mask   
    GY00 =  1; GY01 = 2; GY02 =   1;    
    GY10 =  0; GY11 = 0; GY12 =   0;    
    GY20 = -1; GY21 =-2; GY22 =  -1;

    int valX,valY,MAG;
    #pragma acc data copyin(pixels[0:nx*ny*nc]) create(TMPX[0:ny][0:nx]) create(TMPY[0:ny][0:nx]) copyout(output[0:nx*ny]) 
    {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            TMPY[i][j] = 0;
            TMPX[i][j] = 0;
        }
    }


    #pragma acc parallel loop
    for(int i=0; i < ny; i++)
    {
        #pragma acc loop independent 
        for(int j=0; j < nx; j++)
        {
            //setting the pixels around the border to 0, because the Sobel kernel cannot be allied to them
            if ((i==0)||(i==ny-1)||(j==0)||(j==nx-1)){TMPX[i][j] = 0; TMPY[i][j]= 0;}
            else
            {
                        TMPY[i][j] +=  pixels[yxc(i-1,j-1,0,nx,nc)]* GY00 +  pixels[yxc(i,j-1,0,nx,nc)]* GY10 +  pixels[yxc(i+1,j-1,0,nx,nc)]* GY20 + pixels[yxc(i-1,j,0,nx,nc)]* GY01 + pixels[yxc(i,j,0,nx,nc)]* GY11 +pixels[yxc(i+1,j,0,nx,nc)]* GY21 + pixels[yxc(i-1,j,0,nx,nc)]* GY02 + pixels[yxc(i,j,0,nx,nc)]* GY12 +  pixels[yxc(i+1,j,0,nx,nc)]* GY22;
                        TMPX[i][j] +=  pixels[yxc(i-1,j-1,0,nx,nc)]* GX00 +  pixels[yxc(i,j-1,0,nx,nc)]* GX10 +  pixels[yxc(i+1,j-1,0,nx,nc)]* GX20 + pixels[yxc(i-1,j,0,nx,nc)]* GX01 + pixels[yxc(i,j,0,nx,nc)]* GX11 +pixels[yxc(i+1,j,0,nx,nc)]* GX21 + pixels[yxc(i-1,j,0,nx,nc)]* GX02 + pixels[yxc(i,j,0,nx,nc)]* GX12 +  pixels[yxc(i+1,j,0,nx,nc)]* GX22;
            }
        }
    }
    
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
    }
    
    for (int i=0;i<ny;++i) delete [] TMPY[i];
    for (int i=0;i<ny;++i) delete [] TMPX[i];
   delete[] TMPY;
   delete[] TMPX;

    return;
}
