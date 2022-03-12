#pragma once
#include "cuda_wrapper.h"
#include "gaussian.hpp"
#include "morphology.hpp"

/* ============================== Gaussian Blur ============================= */
__global__ void gaussian_blur(
    unsigned char* input, 
    unsigned char* output, 
    float* gaussian, GaussianArgs* G,
    const int width, const int height
);
/* ========================================================================== */



/* ============================ Otsu Thresholding =========================== */
__global__ void otsu_count(
    unsigned char* input,
    unsigned int* output,
    const int width, const int height
);
// -----------------------------------------
__global__ void otsu_variance(
    unsigned int* count_in,
    float* vari_out,
    const int total
);
// -----------------------------------------
__global__ void otsu_binarize(
    unsigned char* image_in,
    unsigned char* image_out,
    int threshold,
    const int width, const int height
);
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
);
// -----------------------------------------
__device__ void morph_erode(
    unsigned char* input,
    unsigned char* output, 
    MorphSize* M,
    bool* element,
    const int x, const int y,
    const int tid,
    const int width, const int height
);
// -----------------------------------------
__global__ void morph_filter(
    unsigned char* input, 
    unsigned char* output,
    MorphSize* M,
    bool* element,
    bool isOpen,
    const int width, const int height
);
/* ========================================================================== */



/* ============================ Boundary Tracing ============================ */
__global__ void boundary_trace(
    unsigned char* input,
    unsigned char* output,
    const int width, const int height
);
/* ========================================================================== */



/* =============================== Bitwise Or =============================== */
__global__ void bitwise_or(
    unsigned char* input,
    unsigned char* mask,
    unsigned char* output,
    const int width, const int height
);
/* ========================================================================== */



/* =============================== Bitwise Not =============================== */
__global__ void bitwise_not(
    unsigned char* input,
    unsigned char* output,
    const int width, const int height
);
/* ========================================================================== */