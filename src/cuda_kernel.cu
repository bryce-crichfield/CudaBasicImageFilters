#include "../inc/cuda_kernel.cuh"
#include "../inc/cuda_wrapper.h"
#include "../inc/gaussian.hpp"
#include "../inc/morphology.hpp"
#include <math.h>


#define BLACK   0
#define WHITE   255
#define HIST_SIZE 256


/* ============================== Gaussian Blur ============================= */
__global__ void gaussian_blur(
    unsigned char* input, 
    unsigned char* output, 
    float* gaussian, GaussianArgs* G,
    const int width, const int height
)
{   
    // Get X and Y value of image and 1D index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;
    // Exit if we are out of bounds
    if(y >= height || x >= width) { return; }
    // If we are in padding bounds, convolve
    if(x >= G->RADIUS && y >= G->RADIUS && 
        x < (width- G->RADIUS) && y < (height- G->RADIUS)) 
    { 
        float sum = 0.0f;
        for(int dy= y - G->RADIUS, g_y= 0; dy < y + G->RADIUS; dy++, g_y++) {
            for(int dx= x - G->RADIUS, g_x= 0; dx < x + G->RADIUS; dx++, g_x++) {
                unsigned char pixel = input[dy * width + dx];
                float gauss = gaussian[g_y * G->DIAMETER + g_x];
                sum += pixel * gauss;
            }
        }
        unsigned char result = 0;
        if(sum > 255) { result = 255; }
        else if(sum < 1) { result = 0; }
        else result = (unsigned char) sum;
        output[tid] = result;
    } else {
    // Otherwise do a simple copy (leaves edge unblurred)
        output[tid] = input[tid];
    }
}
/* ========================================================================== */



/* ============================ Otsu Thresholding =========================== */
__global__ void otsu_count(
    unsigned char* input,
    unsigned int* output,
    const int width, const int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;
    if(x < width && y < height) {     
        atomicAdd(&output[input[tid]], 1);
    } else { 
        return;
    }
}
// -----------------------------------------
__global__ void otsu_variance(
    unsigned int* count_in,
    float* vari_out,
    const int total
)
{

    // calc prob sum and histogram
    __shared__ unsigned int prob_sum;
    __shared__ float hist[HIST_SIZE];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    prob_sum += tid * count_in[tid];
    hist[tid] = (float) count_in[tid] / total;
    __syncthreads();

    // calc variances
    float p1 = 0, p2 = 0;
    float m1 = 0, m2 = 0;
    float s1 = 0;

    for(int t = 0; t <= tid % HIST_SIZE; t++) {
        p1 += hist[t];
        s1 += t * hist[t];
    }

    p2 = 1 - p1;
    m1 = (float) s1 / (float) p1;
    m2 = (float) (prob_sum - s1) / (float) p2;
    vari_out[tid] = p1 * p2 * pow((m1 - m2), 2);
}
// -----------------------------------------
__global__ void otsu_binarize(
    unsigned char* image_in,
    unsigned char* image_out,
    int threshold,
    const int width, const int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;
    if(image_in[tid] < threshold) {
        image_out[tid] = 0;
    } else {
        image_out[tid] = 255;
    }
}
/* ========================================================================== */



/* ========================= Morphological Transform ======================== */
__device__ void morph_dilate(
    unsigned char* input,
    unsigned char* output, 
    MorphSize* M,
    bool* element,
    const int x, const int y,
    const int tid,
    const int width, const int height
)
{
    bool fill = false;
    for(int mx = x - M->RADIUS, cx = 0; mx < x + M->RADIUS; mx++, cx++) {
        for(int my = y - M->RADIUS, cy = 0; my < y + M->RADIUS; my++, cy++) {
            bool image = input[my * width + mx] == BLACK;
            bool morph = element[cy * M->DIAMETER + cx] == true;
            if(image && morph) {
                fill = true;
                break;
            }
        }
    }
    if(fill) output[tid] = BLACK; 
}
// -----------------------------------------
__device__ void morph_erode(
    unsigned char* input,
    unsigned char* output, 
    MorphSize* M,
    bool* element,
    const int x, const int y,
    const int tid,
    const int width, const int height
)
{
    bool erase = false;
    for(int mx = x - M->RADIUS, cx = 0; mx < x + M->RADIUS; mx++, cx++) {
        for(int my = y - M->RADIUS, cy = 0; my < y + M->RADIUS; my++, cy++) {
            bool image = input[my * width + mx] == WHITE;
            bool morph = element[cy * M->DIAMETER + cx] == true;
            if(image && morph) {
                erase = true;
                break;
            }
        }
    }
    if(erase) output[tid] = WHITE;
}
// -----------------------------------------
__global__ void morph_filter(
    unsigned char* input, 
    unsigned char* output,
    MorphSize* M,
    bool* element,
    bool isOpen,
    const int width, const int height
)
{
    // Get X and Y value of image and 1D index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;
    // Exit if we are out of bounds
    if(y >= height || x >= width) { return; }

    if(x >= M->RADIUS && y >= M->RADIUS && 
        x < (width-M->RADIUS) && y < (height-M->RADIUS)) 
    {
        output[tid] = input[tid];
        if(isOpen) {
            morph_erode(
                input, output, M, element,
                x, y, tid,
                width, height
            );    
            morph_dilate(
                input, output, M, element,
                x, y, tid,
                width, height
            );
        } else {
            morph_dilate(
                input, output, M, element,
                x, y, tid,
                width, height
            );
            morph_erode(
                input, output, M, element,
                x, y, tid,
                width, height
            );  
        }
    } else {
        output[tid] = input[tid];
    }  
}
/* ========================================================================== */


/* ============================= Boundary Trace ============================= */
__global__ void boundary_trace(
    unsigned char* input,
    unsigned char* output,
    const int width, const int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;
    bool pixel = input[tid] == BLACK;
    bool onEdge = x == 0 || y == 0 || x == width - 1 || y == height - 1;
    if(onEdge && pixel) {
        output[tid] = BLACK;
        return;
    } else {
        bool top = input[(y - 1) * width + x] == BLACK;
        bool bot = input[(y + 1) * width + x] == BLACK;
        bool lef = input[y * width + (x - 1)] == BLACK;
        bool rig = input[y * width + (x + 1)] == BLACK;
        bool active_neighbor = top || bot || lef || rig;
        bool inactive_neigbor = !top || !bot || !lef || !rig;
        if(active_neighbor && inactive_neigbor) {
            output[tid] = BLACK;
            return;
        } else {
            output[tid] = WHITE;
        }
    }

}
/* ========================================================================== */



/* =============================== Bitwise Or =============================== */
__global__ void bitwise_or(
    unsigned char* input,
    unsigned char* mask,
    unsigned char* output,
    const int width, const int height
) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;

    bool a = input[tid] == BLACK;
    bool b = mask[tid] == BLACK;
    if(a && !b || !a && b) {
        output[tid] = WHITE;
    } else if(a && b || !a && !b) {
        output[tid] = BLACK;
    }
    else {
        output[tid] = BLACK;
    }
}
/* ========================================================================== */



/* =============================== Bitwise Not ============================== */
__global__ void bitwise_not(
    unsigned char* input,
    unsigned char* output,
    const int width, const int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * width + x;

    if(input[tid] == BLACK) 
        output[tid] = WHITE;
    else 
        output[tid] = BLACK;
}
/* ========================================================================== */