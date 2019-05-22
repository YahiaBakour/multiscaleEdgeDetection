# Parallizeing Multiscale Edge Detection with OpenACC, OpenMPI, and FFT
<p> By: <b>Yahia Bakour and James Dunn</b></p>

 
 <p float="left" align="center">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/architecture-cliffside-cold-789380.jpg" width="345.6" height="230.4">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/mountain_edge_1.jpg" width="345.6" height="230.4">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/mountain_edge_2.jpg" width="345.6" height="230.4">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/mountain_edge_4.jpg" width="345.6" height="230.4">
</p>
<p><a href="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/EC526%20Report_%20Multiscale%20Edge%20Detection.pdf">
Link to Report</a></p>
<p><a href="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/FINAL%20PRESENTATION_%20MultiScale%20Edge%20Detection.pptx">
Link to Slides</a></p>

## Abstract
Multiscale edge detection is a computer vision technique that finds pixels in an image that have sharp gradients at differing physical scales.  The resulting multiscale edge-maps are useful for tasks such as object segmentation and image alignment/registration.  At the core of all edge detection algorithms is a convolution of the input image with a kernel approximating the spatial derivative (gradient) of the image brightness.  This convolution, along with other loops in the program running the edge detection, are prime candidates for a parallel implementation within a GPU or across multiple cores or GPUs.  

We present a parallelized implementation of the multiscale edge detection algorithm in c++ using openACC.  A separate open MPI-based parallel implementation is also presented. The parallel implementations are compared with both a serial implementation and an implementation that uses the Fourier convolution theorem.  The openACC implementation on a GPU was the fastest, at 13.9x the speed of the serial implementation on a large 17.9 Mpix image.

Code was tested on the Boston University shared computing cluster (scc1.bu.edu).

## Introduction

The term “edge detection” applies to a variety of mathematical methods that aim to identify pixels in an image where the brightness changes sharply with respect to its neighboring pixels.  Edge detection is a fundamental tool in computer vision and image processing used for segmentation and registration/alignment. We use the Sobel edge detection algorithm [1] to generate edge maps, but the methods in this report apply to other edge detection algorithms as well. The Sobel method convolves the original image by two 3x3 kernels to approximate the brightness derivatives in the horizontal and vertical directions.

The multiscale edge detection procedure simply runs the edge detection algorithm at multiple scales.  Specifically, it takes the original image, shrinks it by some factor(s), then runs the same edge detection algorithm on the shrunk image(s).  The result is a stack of edge-maps that show the edges in the image at different physical scales.  The images in Figures 1 through 8 show one potential application of the multiscale edge detection technique.

The non-dependent loops in the multiscale edge detection process make it a prime candidate for a parallel implementation, both on a GPU via openACC, and across cores via MPI.

We first implement a serial version of the edge detection algorithm in c++ and use it for multiscale edge detection. We then examine two different parallelization techniques with the aim of speeding up the program: an OpenACC implementation run on a Tesla GPU, and an MPI implementation with each edgemap scale run on a separate CPU core.  Finally, we compare a serial implementation that uses the Fourier convolution theorem to execute the convolutions in the edge detection algorithm.

In all our implementations, the output edgemaps themselves are identical or nearly identical, but the runtime is markedly different.  We use runtime on the target system, the Boston University Shared Computing Cluster (SCC) as our primary performance metric.


## Prerequisites
> * GCC
> * openMPI
> * openACC
> * We used a shared computing cluser with Tesla GPUs, but you will require a gpu to run this.


## Get started

First clone this repository to your shared computing cluster folder or to your local machine assuming you have the pre-requisites listed above. 

To run the serial version run:
```
module load gcc/7.2.0
make
./edgeDetect flowers.jpg
make -k clean
```

To run the FFT version run:
```
module load gcc/7.2.0
make
./edgeDetectFFT flowers.jpg
make -k clean
```

To run the MPI version run:
```
module load mpich
make -k -f makeMPI
mpirun -np 5 ./edgeDetectMPI flowers.jpg
make -k -f makeMPI -k clean\
```

To run the ACC version run:
```
module load pgi/18.4
make -k -f makeACC
./edgeDetectACC flowers.jpg
make -k -f makeACC -k clean
```

## Serial Implementation

The serial implementation of our multiscale edge detection algorithm proceeds as follows.  We first shrink the input image by a factor F[2,16], we run the Sobel Edge Detection algorithm on the shrunk image, then we enlarge the image by a factor F, then repeat for larger and larger factors F. The purpose of this is to gain more and more detail from the image on previously overlooked edges. 

Algorithm for Multiscale Edge Detection:
Require: I : Input Image, Output: Array containing output images, F: Array of factors to run multiscale edge detection on.

<b>Equation 1: Pseudocode for the multiscale edge detection algorithm. </b>
```
Multiscale(I,Output,F):
-For I in range(Len(F)):
--OutSmallImage <- Size of output image with factor
--ShrinkImage(Input, OutSmallImage, F[I]) 
--SobelEdgeDetection(OutSmallImage,OutEdgemap)
--EnlargeImage(OutEdgemap, Output[I], F[I])
```

Sobel Edge Detection is done by convolving the following 2 3x3 kernels with the original image :
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/kernelsforsobeloperator.png" >

Where:
* A is the original input image 
* Gx and Gy are the gradients found from the convolution. 

We then find the Magnitude of the gradient from it’s components and decide whether it is above or below a certain threshold. If we determine that it is above the edge threshold, then we count it as an edge. Shrinking the input image is done using simple average pooling to determine what the new pixel should be from the previous pixels. Enlarging the image is done by simple copying of the previous pixel to the new pixels.</p>

## OpenACC Implementation

<p>Very quickly we realized that the serial version of our code had many backwards compatibility issues and we had to rewrite the code for our grayscale, enlarge, shrink, and sobel edge detection functions to be parallelizable by openACC. This led to a speedup of slightly below 5x. Then we realized that a lot of the time, the arrays necessary for GPU accelerated multiscale edge detection were already on the GPU and didn’t have to be copied in and out repetitively. We rewrote our multiscale edge detection function to only copyin the input array (Image) once and then constantly refer to it when needed using openACC’s present clause. We managed to gain a speedup of x7 on the small image from the serial version after modifying it to remove all backwards dependencies and ensuring to copyin all static arrays during the multiscale edge detection. We also turned our 2 kernels from 3x3 matrices to 18 single int variables which helped shave off a few more seconds as we didn’t have to perform a copyin of the 2 kernels everytime the function is run.</p>

<p>One of the main issues we faced was decoupling dependent loops to gain independence from one another, after the loops were intrinsically parallel we would insert the <i><b>#pragma acc loop independent</b></i> directive to tell the compiler that the loops are independent from one another. We also tested using the <i><b>#pragma acc loop collapse(n)</b></i>  directive that collapses the first n loops into one loop but found better results with the first approach.</p>

<p>We also realized that a simple naive ACC Implementation of this project would have only gained a speedup of x2 but after we repeatedly analyzed the code for speed bottlenecks, we were able to maximize our usage of the GPU through the use of OpenACC best practices and accelerate our code. </p>


## OpenMPI Implementation

<p>Open MPI provides a smooth and compact way to simultaneously execute a program on multiple “ranks”.  Each rank can be a separate core, CPU, GPU, or even a separate physical computer.  Open MPI also provides a standard for communication between the ranks that is made invisible to the programmer. </p>

<p>The most obvious and clear-cut way to speed up multiscale edge detection with MPI is to simply run each of the edge scales on its own rank because the scales are completely independent from one another.  A less clear-cut way to use MPI would be to split up the image into blocks and assign each block to a different rank.  Less obvious still would be to use MPI to split up the edge scales across GPUs, and then let openACC parallelize on each GPU.</p>

<p>We implement the first of these methods - simply running each scale in the multiscale edge detection algorithm on its own core of a CPU.  Using MPI and ACC simultaneously to run across multiple GPUs was explored but abandoned for lack of time.</p>

## FFT Implementation

<p>The Fast Fourier Transform’s (FFT) primary use is to quickly convert data from time space to frequency space and back again with order Nlog2(N) operations.  By comparison, the standard (slow) Discrete Fourier Transform (DFT) uses order N2 operations.  The FFT thus provides the potential for significant decreases in runtime, particularly for large arrays.  Unfortunately, the FFT is a recursive algorithm, so it is not amenable to parallelization.</p>

<p>One of the other uses of the FFT is to perform fast convolutions, which could make the convolution step of our edge detection algorithm faster.  This is made possible by use of the Fourier Convolution Theorem [3].  The Fourier Convolution Theorem states that the inverse Fourier transform of the product of two Fourier transformed functions is mathematically equivalent to convolution.  In discrete pseudocode: <b> A⊗B=invFFT(FFT(A) * FFT(B)) </b> </p>
<p>The most intuitive way to understand this is that the shift operation in frequency space is performed with multiplication.  Mathematically, a shift of x by  is performed with:
<b>e^(ik(x+lamba))=e^(ikx)*e(ik*lambda)</b></p>

<p>Convolution is merely a series of shifts and multiplications, so this simple equation brings light to why the Fourier convolution theorem works.</p>

<p>Naïvely, one would expect that simply taking the serial edge detection code and replacing the corresponding 4x nested for loops with a FFT-based convolution would offer an N4 to 2N2log2(N) speedup.  Upon careful consideration however, this arithmetic does not accurately describe the convolutions that are performed in edge detection.  </p>

<p> Let the image that we wish to convolve by be NxN pixels, and let the edge detection kernel be the usual 3x3 Sobel kernel.  One direction of the Sobel convolution (x or y), thus takes only O(9N2) operations. </p>

<p>The Fourier convolution theorem requires the two arrays being convolved to be the same size, because the product in the theorem is a pointwise product.  Consequently, in order to convolve two arrays that are not the same size, in our case an NxN image and a 3x3 kernel, the smaller of the two arrays must be zero-padded to the size of the larger array before performing the FFT.  The FFT-based convolution must act on two N2 images instead of one N2 image and one 32 image. Thus, instead of getting the desired 9N2 to 2log2(3) Nlog2(N) speedup that we might have expected, we instead get 9N2 vs 2N2log2(N).  This actually means the FFT convolution will be a slower algorithm for any N>32. </p>

In our implementation, the FFT convolution method actually does somewhat worse than 2N2log2(N) due to the type conversions from the pixels’ native uint_8 to the 64-bit complex required by the FFT and the associated additional memory allocations.  Our experiments on 2.4 MP and larger images (see results section) confirm this.


## Results

Trials using each of the implementations defined above were run 10 times on the BU SCC to gather statistics.  Four images with sizes from 2.4 Mpix to 17.9 Mpix were used to show how runtime scales with problem size.

The serial implementation was compiled with g++ using only the “-std=c++11” flag. No compiler optimizations were used. The open ACC implementation was run on a single Tesla M2070 GPU cards with 6 GB of Memory.  Open ACC compilation was performed with “pgc++ -Minfo=accel -ta=tesla ...” and run with 1 CPU and 1 Tesla GPU in an interactive terminal via “qrsh -pe omp 1 -P paralg -l gpus=1.0 -l gpu_c=6.0”. Open MPI compilation was perfomred with mpicxx, and run with 4 cores (“-np 4”) using the same qrsh command to get an interactive terminal.

<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/resulting_runtimes.png" width="50%" height="50%">

| Implementation  | Runtime 士 σ (s) | |
| ------------- | ------------- |------------- |
| Image Size | <b>1920x1232</b>  | <b>5184x3456</b>  |
| Serial | 0.567 士0.00078  | 4.34 士 0.0075 | 
| ACC on Tesla | 0.081 士 0.0012 | 0.312 士 0.00366 | 
| MPI-4 Cores | 0.358 士  0.004| 2.711 士 0.009 | 
| FFT Convolutions | 91.23 士 0.14| ----- | 


The FFT implementation for 3x3 Sobel kernels is 161x slower than the serial implementation, for reasons mentioned earlier in section V.  To demonstrate at what size kernel the FFT would start to outperform the serial and parallel (non-FFT) implementations, we can artificially enlarge the edge detection convolution kernel.  This slows the serial implementation down, but has a minimal effect of the FFT implementation.  We found that enlarging the kernel from 3x3 to 47x47 made the serial implementation have comparable runtime to the FFT implementation.

| Implementation  | Kernel Size | Runtime 士 σ (s) |
| ------------- | ------------- |------------- |
| Serial | 47x47 | 97.47 士 0.32  |
| FFT convolutions| 47x47  | 98.28 士 0.36 | 

<p>We managed to get a GPU powered speedup of <b>x7.5</b> on the small image and <b>x13.9</b> on the large image</p> 

## Output of running the program on flowers.jpg

<p> This shows the result of running the multiscale edge detection program on flowers.jpg with the following scales taken into account: <b>x1, x2, x4, x6, x8</b></p>

<p float="left" align="center">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/flowers.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_1x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_2x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_4x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_6x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_8x.jpg" width="192" height="123">
</p>


## Conclusions

<p> We have successfully parallelized all of the parallelizable loops in a multiscale edge detection algorithm implemented in c++.  The best runtime improvements were shown using  openACC on a Tesla GPU.  </p>

<p> By an apples-to-apples runtime comparison against a serial version of the same code on the same system (Boston University SCC), we found an improvement of 7.036x for a small image (2.4 Mpix), and 13.9x for a large image (17.9 Mpix) using ACC.  Parallelization across CPUs with MPI yielded the approximately the expected speedup of 1.6x on the small image and 1.6x on the large image  by letting each scale run on its own core. The full-scale edge detection took the most time to run, and so we were able to execute all scales of edge detection in the same time as the full scale.</p>

<p> To complement the parallelization effort, we took the convolution part of the serial code and implemented it using an FFT with the Fourier convolution theorem.  The resulting 161x slower runtime shows that the FFT is at a clear runtime disadvantage for the small 3x3 pixel kernels used for Sobel edge detection.  We ran with larger kernels to determine what kernel size would be needed for the FFT to be a more efficient implementation.  We found that the FFT implementation was more efficient than the serial implementation for kernels larger than 47x47 pixels for the images we calculated performance on. </p>

<p> Future work would include implementing the MPI parallelization more efficiently by having each rank run a slice of the image at all scales, rather than each rank running a single scale.  This is straightforward, but it involves communication between ranks at the boundaries and the associated additional code.  Additionally, the pgc++ compiler (ACC) could be linked to from MPI.  This would allow us to use MPI to split up the task between multiple GPUs, rather than across multiple CPU cores, resulting in the best of both ACC and MPI.  We attempted the ACC+MPI fusion methods in [7] and [8], but neither was configured properly to work on the SCC.  Further effort in this area would likely result in additional speed gains. </p> 



## References

* “Sobel operator” , https://en.wikipedia.org/wiki/Sobel_operator
* “Purple Petaled Flower Field”, https://www.pexels.com/photo/purple-petaled-flower-field-1131407/
* “Convolution Theorem”, https://en.wikipedia.org/wiki/Convolution_theorem
* “OpenACC Tutorial - Data movement”, https://docs.computecanada.ca/wiki/OpenACC_Tutorial_-_Data_movement#C.2B.2B_Classes
* “Advanced OpenACC” ,  http://icl.cs.utk.edu/classes/cosc462/2017/pdf/OpenACC_3.pdf
* “Architecture-cliffside-cold”, https://images.pexels.com/photos/789380/pexels-photo-789380.jpeg?cs=srgb&dl=architecture-cliffside-cold-789380.jpg&fm=jpg
* “Multiprocessor Programming: TechWeb : Boston University”, http://www.bu.edu/tech/support/research/software-and-programming/programming/multiprocessor/#MPI 
* “OpenACC + MPI”, http://www.speedup.ch/workshops/w43_2014/tutorial/html/openacc_mpi_1.html
* "Image Reading Code", https://github.com/nothings/stb, accessed 4/10/19



