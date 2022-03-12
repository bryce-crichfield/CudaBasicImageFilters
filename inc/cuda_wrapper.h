#pragma once
#include <opencv2/opencv.hpp>
#include "gaussian.hpp"
#include "morphology.hpp"

/* ================================ CUDAImage =============================== */
typedef struct {
    unsigned char* d_in;
    unsigned char* d_out;
    unsigned char* d_mask;
    size_t width;
    size_t height;
    size_t bytes;
} CUDAImage;
CUDAImage* cuda_image_create(cv::Mat* mat);
void cuda_image_mat(CUDAImage* image, cv::Mat* mat);
void cuda_image_swap(CUDAImage* image);
void cuda_image_destroy(CUDAImage* image);
cv::Mat loadImage(int level);


/* ============================ CUDAImage Filters =========================== */
void run_gaussian_blur(CUDAImage* image, GaussianArgs* G);
void run_otsu_threshold(CUDAImage* image);
void run_morph_filter(CUDAImage* image, MorphSize* M, int order, int e_type, int iterations);
void run_boundary_trace(CUDAImage* image);
void run_floodfill(CUDAImage* image);
void run_bitwise_or(CUDAImage* image);
void run_bitwise_not(CUDAImage* image);