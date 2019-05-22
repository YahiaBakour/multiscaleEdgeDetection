/* Code for multiscale edge detection using openMP (ACC) within MPI ranks
- Yahia Bakour and James Dunn, Boston University
- EC526 - Parallel Programming for High Performance Computing final project
- April/May 2019
- Image reading/writing code is courtesy of this open source library: https://github.com/nothings/stb
*/


// TO COMPILE
// Just use make -k -f makeMPI

// TO RUN
// Get a multiple-core session on the scc with 
// >>qrsh -l h_rt=00:30:00 -pe omp 1 -P paralg -l gpus=1.0 -l gpu_c=6.0
// or something similar.
// Run with mpirun -np 4 ./edgeDetectMPI flowers.jpg
// Where -np # represents the number of processors AND SCALES to run multiscale edge detection on
// Any input image may be used, flowers.jpg is just one example.

#include <iostream>
#include <chrono>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "utilities.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Useful globals
int world_size; // number of processes
int my_rank; // my process number


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


    // MPI initializations.  Note that each MPI rank (processor) here is used
    // merely to calculate a different scale of edge detection, making the implementation
    // somewhat trivial.  Nonetheless, it should still provide a significant speedup.
    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Get the rank (i.e. the thread number)
    cout << "My rank is " << my_rank+1 << " out of " << world_size << endl;


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



    // =================================================================================================== //
    // MULTISCALE EDGE DETECTION
    int nlevels = 1;
    int levels [5] = {1,2,4,6,8};
    levels[0]=levels[my_rank];


    // Allocate multiscale edgemaps
    uint8_t ** multiscaleEdges = new uint8_t * [nlevels];
    for (int l=0;l<nlevels;++l) multiscaleEdges[l] = new uint8_t [nx*ny/(levels[l]*levels[l])];

    // Get the starting timestamp. 
    Time mbegin_time = std::chrono::steady_clock::now();

    // Run multiscale edge detection
    cout << "Running multiscale edge detection...";
    findMultiscaleEdges(image_gray, multiscaleEdges, levels, nlevels, ny, nx, nc);
    cout << "Done" << endl;

    // Wait for all threads to return so we get valid timing
    MPI_Barrier(MPI_COMM_WORLD);


    // Get the end timestamp
    Time mend_time = std::chrono::steady_clock::now(); 
    DeltaTime mdt = mend_time - mbegin_time; // Compute the difference.
    printf("Multiscale Edge detection runtime (rank %d, scale %d) was %.10f seconds\n", my_rank, levels[0], mdt.count());

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
    delete [] image_gray;
    for (int i=0;i<nlevels;++i) delete [] multiscaleEdges[i];
    delete [] multiscaleEdges; 
    delete [] enlargedEdges;


    // Clean up
    MPI_Finalize();

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
    static int GX [3][3]; static int GY [3][3];

    //Two arrays to store values for parallelization purposes
    int **TMPX = new int *[ny];
    int **TMPY = new int *[ny];

    for (int i = 0; i < ny; i++) {
        TMPY[i] = new int[nx];
        TMPX[i] = new int[nx];
    }


    //Sobel Horizontal Mask     
    GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1; 
    GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;  
    GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

    //Sobel Vertical Mask   
    GY[0][0] =  1; GY[0][1] = 2; GY[0][2] =   1;    
    GY[1][0] =  0; GY[1][1] = 0; GY[1][2] =   0;    
    GY[2][0] = -1; GY[2][1] =-2; GY[2][2] =  -1;

    int valX,valY,MAG;
    #pragma acc data copyin(pixels[0:nx*ny*nc]) copyin(GX[0:3][0:3]) copyin(GY[0:3][0:3]) create(TMPX[0:ny][0:nx]) create(TMPY[0:ny][0:nx]) copyout(output[0:nx*ny]) 
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
                        TMPY[i][j] +=  pixels[yxc(i-1,j-1,0,nx,nc)]* GY[0][0] +  pixels[yxc(i,j-1,0,nx,nc)]* GY[1][0] +  pixels[yxc(i+1,j-1,0,nx,nc)]* GY[2][0] + pixels[yxc(i-1,j,0,nx,nc)]* GY[0][1] + pixels[yxc(i,j,0,nx,nc)]* GY[1][1] +pixels[yxc(i+1,j,0,nx,nc)]* GY[2][1] + pixels[yxc(i-1,j,0,nx,nc)]* GY[0][2] + pixels[yxc(i,j,0,nx,nc)]* GY[1][2] +  pixels[yxc(i+1,j,0,nx,nc)]* GY[2][2];
                        TMPX[i][j] +=  pixels[yxc(i-1,j-1,0,nx,nc)]* GX[0][0] +  pixels[yxc(i,j-1,0,nx,nc)]* GX[1][0] +  pixels[yxc(i+1,j-1,0,nx,nc)]* GX[2][0] + pixels[yxc(i-1,j,0,nx,nc)]* GX[0][1] + pixels[yxc(i,j,0,nx,nc)]* GX[1][1] +pixels[yxc(i+1,j,0,nx,nc)]* GX[2][1] + pixels[yxc(i-1,j,0,nx,nc)]* GX[0][2] + pixels[yxc(i,j,0,nx,nc)]* GX[1][2] +  pixels[yxc(i+1,j,0,nx,nc)]* GX[2][2];
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
