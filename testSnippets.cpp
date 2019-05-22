
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // A FEW EXAMPLES SHOWING HOW TO ACCESS AND CHANGE INDIVIDUAL PIXEL VALUES
    // Make the top row of pixels pure yellow
/*     for (int x=0; x<nx; ++x) {
        image[nc*x + 0] = 255; //Red
        image[nc*x + 1] = 255; //Green 
// +2 -- Blue
    } */

    // Notice that we are y-major here, meaning that pixel 1 is (0,0) but pixel 2 is (1,0)
    // Accordingly, it is much faster to access a row of pixels at once than a column of pixels
    // Make a few rows of pixels pure white
/*     for (int y=100; y<110; ++y){
        for (int x=0; x<nx; ++x) {
            for (int c=0; c<nc; ++c) {
                //image[nx*nc*y + nc*x + c] = 255;
                image[yxc(y,x,c,nx,nc)] = 255;
            }
        }
    }
 */
/* 
     stbi_write_png("testOutput.png", nx, ny, nc, image, nx*3);
    cout << "Wrote testOutput.png" << endl;
    stbi_write_jpg("testOutput.jpg", nx, ny, nc, image, JPG_QUALITY);
    cout << "Wrote testOutput.jpg" << endl;
    stbi_write_bmp("testOutput.bmp", nx, ny, nc, image);
    cout << "Wrote testOutput.png" << endl; 
 */
    // Test shrink function
    int factor = 4;
    uint8_t * imagesmall = new uint8_t [ny*nx*nc/(factor*factor)];
    shrink(image,imagesmall,ny,nx,nc,factor);
    stbi_write_jpg("outputSmall4x.png", nx/factor, ny/factor, nc, imagesmall, JPG_QUALITY);
    delete [] imagesmall;

    // Test enlarge function
    factor = 2;
    uint8_t * imagelarge = new uint8_t [ny*nx*nc*factor*factor];
    enlarge(image,imagelarge,ny,nx,nc,factor);
    stbi_write_jpg("outputLarge2x.png", nx*factor, ny*factor, nc, imagelarge, JPG_QUALITY);
    delete [] imagelarge;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // END CODE THAT TESTS HOW TO READ/CHANGE/WRITE IMAGES
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

