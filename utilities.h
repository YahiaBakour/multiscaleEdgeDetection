#ifndef UTILITIES_H
#define UTILITIES_H
// Utility functions for dealing with images
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019

#include <cstdint>

// Hardcoded parameters
#define JPG_QUALITY 100 // 0 to 100, 100 being best quality and largest file
#define NCOLORS 3 // use 3 colors (RGB)

// Preprocessor directives
#define Time std::chrono::time_point<std::chrono::steady_clock>
#define DeltaTime std::chrono::duration<double>

// Forward declarations
void Grayscale(uint8_t *input, uint8_t *output, int ny, int nx, int nc);
void shrink(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor);
void enlarge(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor);
inline int yxc(int y, int x, int c, int nx, int nc) { return nx*nc*y + nc*x + c; } // converts 3-d indices into 1d index


#endif
