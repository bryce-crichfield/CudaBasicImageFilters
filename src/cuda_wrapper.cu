#include <opencv2/opencv.hpp>
#include <openslide/openslide.h>
#include "../inc/cuda_wrapper.h"
#include "../inc/cuda_kernel.cuh"
#include "../inc/gaussian.hpp"
#include "../inc/morphology.hpp"
#include <stdio.h>
#include <vector>
using namespace cv;


/* ========================================================================== */
/*                                  CUDAImage                                 */
/* ========================================================================== */
CUDAImage* cuda_image_create(cv::Mat* mat)
{
    unsigned char* input = mat->data;
    size_t width = mat->cols; size_t height = mat->rows;
    CUDAImage* image = (CUDAImage*)malloc(sizeof(CUDAImage));
    int byte_size = width * height * sizeof(unsigned char);
    unsigned char* d_in;
    unsigned char* d_out;
    unsigned char* d_mask;
    assert(cudaMalloc((void**) &d_in, byte_size) 
    == cudaSuccess);
    assert(cudaMalloc((void**) &d_out, byte_size) 
    == cudaSuccess);
    assert(cudaMalloc((void**) &d_mask, byte_size) 
    == cudaSuccess);
    assert(cudaMemcpy(d_in, input, byte_size, cudaMemcpyHostToDevice) 
    == cudaSuccess);
    image->d_in = d_in;
    image->d_out = d_out;
    image->d_mask = d_mask;
    image->width = width;
    image->height = height;
    image->bytes = image->width*image->height * sizeof(unsigned char);
    return image;
}

void cuda_image_mat(CUDAImage* image, cv::Mat* mat)
{   
    int size = image->width*image->height;
    int byte_size = size * sizeof(unsigned char);
    unsigned char* data = new unsigned char[size];
    assert(cudaMemcpy(data, image->d_in, byte_size, cudaMemcpyDeviceToHost) 
    == cudaSuccess);
    for(int x = 0; x < image->width; x++) {
        for(int y = 0; y < image->height; y++) {
            mat->at<uchar>(x, y) = data[x * image->width + y];
        }
    }
    free(data);
}

void cuda_image_swap(CUDAImage* image) 
{
    unsigned char* temp = image->d_in;
    image->d_in = image->d_out;
    image->d_out = temp;
}

void cuda_image_destroy(CUDAImage* image) 
{
    cudaFree(image->d_in);
    cudaFree(image->d_out);
    cudaFree(image->d_mask);
}

Mat loadImage(int level)
{
    const char* path = "rsc/slide.svs";
    openslide_t* osr = openslide_open(path);
    int64_t w; int64_t h;
    openslide_get_level_dimensions(osr, level, &w, &h);
    uint32_t* buf = new uint32_t[w * h];
    openslide_read_region(osr, buf, 0, 0, level ,w, h);
    openslide_close(osr);
    auto mat = Mat(h, w, CV_8UC4, (void*) buf);
    cvtColor(mat, mat, COLOR_BGR2GRAY);
    return mat;
}


/* ========================================================================== */
/*                            Gaussian Blur Wrapper                           */
/* ========================================================================== */
void run_gaussian_blur(CUDAImage* image, GaussianArgs* G)
{
    // initialize gaussian kernel
    float* gaussian = G->create_kernel();
    float* d_gaussian;
    GaussianArgs* d_G;
    assert(cudaMalloc((void**) &d_gaussian, G->BYTES)
    == cudaSuccess);
    assert(cudaMalloc((void**) &d_G, sizeof(GaussianArgs))
    == cudaSuccess);
    assert(cudaMemcpy(d_gaussian, gaussian, G->BYTES, cudaMemcpyHostToDevice) 
    == cudaSuccess);
    assert(cudaMemcpy(d_G, G, sizeof(GaussianArgs), cudaMemcpyHostToDevice) 
    == cudaSuccess);

    // launch kernel
    const dim3 BlockDim(16,16);
    dim3 GridDim;
    GridDim.x = (image->width + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (image->height + BlockDim.y - 1) / BlockDim.y;
    gaussian_blur<<<GridDim, BlockDim>>>
    (image->d_in, image->d_out, d_gaussian, d_G, image->width, image->height);
    assert(cudaPeekAtLastError() 
    == cudaSuccess);
    cuda_image_swap(image);

    // clean up cuda variables
    cudaFree(d_gaussian);
    cudaFree(d_G);
    free(gaussian);
}



/* ========================================================================== */
/*                            Otsu Theshold Wrapper                           */
/* ========================================================================== */
void run_otsu_threshold(CUDAImage* image)
{
    // initialize counts array in cuda
    unsigned int* d_count;
    int out_bytes = 256 * sizeof(unsigned int);
    assert(cudaMalloc((void**) &d_count, out_bytes) 
    == cudaSuccess);
    float* d_vari;
    int vari_bytes = 256 * sizeof(float);
    cudaMalloc((void**) &d_vari, vari_bytes);

    // calculate histogram with counts array
    const dim3 BlockDim(16,16);
    dim3 GridDim;
    GridDim.x = (image->width + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (image->height + BlockDim.y - 1) / BlockDim.y;
    otsu_count<<<GridDim, BlockDim>>>
    (image->d_in, d_count, image->width, image->height);
    assert(cudaPeekAtLastError() 
    == cudaSuccess);

    // calculate threshold with variances
    float* vari = new float[256];
    otsu_variance<<<256, 1>>>(d_count, d_vari, image->width*image->height);
    assert(cudaPeekAtLastError() 
    == cudaSuccess);
    assert(cudaMemcpy(vari ,d_vari, vari_bytes, cudaMemcpyDeviceToHost) 
    == cudaSuccess);
    double max_var = 0;
    int threshold = 0;
    for(int i = 0; i < 255; i++) {
        if(vari[i] > max_var) {
            threshold = i;
            max_var = vari[i];
        }
    }

    // perform binarization
    threshold -= 40;
    otsu_binarize<<<GridDim, BlockDim>>>
    (image->d_in, image->d_out, threshold, image->width, image->height);
    assert(cudaPeekAtLastError() 
    == cudaSuccess);
    cuda_image_swap(image);

    // clean up cuda variables
    cudaFree(d_count);
    cudaFree(d_vari);
    free(vari);
}



/* ========================================================================== */
/*                  Morphological Opening And Closing Wrapper                 */
/* ========================================================================== */
void run_morph_filter(CUDAImage* image, MorphSize* M, int order, int e_type, int iterations)
{
    // determine and create structuring element
    bool* element;
    switch (e_type)
    {
    case ELEMENT::CROSS:
        element = M->element_cross();
        break;
     case ELEMENT::CIRCLE:
        element = M->element_circle();
        break;
     case ELEMENT::RECTANGLE:
        element = M->element_recta();
        break;
    default:
        printf("RUNTIME ERROR: Provided element type is not correct (run_morph_filter)!");
        break;
    }  

    // initialize cuda variables
    int bytes = image->width * image->height * sizeof(unsigned char);
    bool* d_element;
    assert(cudaMalloc((void**) &d_element, M->SIZE * sizeof(bool))
    ==cudaSuccess);
    assert(cudaMemcpy(d_element, element, M->SIZE * sizeof(bool), cudaMemcpyHostToDevice)
    ==cudaSuccess);
    MorphSize* d_M;
    assert(cudaMalloc((void**) &d_M, sizeof(MorphSize))
    ==cudaSuccess);
    assert(cudaMemcpy(d_M, M, sizeof(MorphSize), cudaMemcpyHostToDevice)
    ==cudaSuccess);

    // launch kernel iterations
    const dim3 BlockDim(16,16);
    dim3 GridDim;
    GridDim.x = (image->width + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (image->height + BlockDim.y - 1) / BlockDim.y;
    for(int i = 0; i < iterations; i++) {
        switch (order)
        {
        case 0:
            morph_filter<<<GridDim, BlockDim>>>
            (image->d_in, image->d_out, d_M, d_element, true, image->width, image->height);
            cuda_image_swap(image);
            morph_filter<<<GridDim, BlockDim>>>
            (image->d_in, image->d_out, d_M, d_element, false, image->width, image->height);
            break;
        case 1:
            morph_filter<<<GridDim, BlockDim>>>
            (image->d_in, image->d_out, d_M, d_element, false, image->width, image->height);
            cuda_image_swap(image);
            morph_filter<<<GridDim, BlockDim>>>
            (image->d_in, image->d_out, d_M, d_element, true, image->width, image->height);
            break;
        default:
            printf("RUNTIME ERROR: Invalid Order for run_morph_filter\n");
            exit(-1);
            break;
        }
        cuda_image_swap(image);
    }
    assert(cudaPeekAtLastError() 
    == cudaSuccess);

    // clean up cuda variables
    cudaFree(d_M);
    cudaFree(d_element);
    free(element);
}



/* ========================================================================== */
/*                               Boundary Trace                               */
/* ========================================================================== */
void run_boundary_trace(CUDAImage* image)
{
    const dim3 BlockDim(16,16);
    dim3 GridDim;
    GridDim.x = (image->width + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (image->height + BlockDim.y - 1) / BlockDim.y;
    boundary_trace<<<GridDim, BlockDim>>>
    (image->d_in, image->d_mask, image->width, image->height);
}



/* ========================================================================== */
/*                                  Floodfill                                 */
/* ========================================================================== */
void run_floodfill(CUDAImage* image) 
{
    unsigned char* mask = new unsigned char[image->width*image->height];
    assert(cudaMemcpy(mask, image->d_in, image->bytes, cudaMemcpyDeviceToHost)
    == cudaSuccess);
    Mat maskMat = cv::Mat(image->height, image->width, CV_8UC1, mask);
    cv::floodFill(maskMat, cv::Point(0, 0), Scalar(0));
    assert(cudaMemcpy(image->d_mask, maskMat.data, image->bytes, cudaMemcpyHostToDevice)
    == cudaSuccess);
}



/* ========================================================================== */
/*                                 Bitwise Or                                 */
/* ========================================================================== */
void run_bitwise_or(CUDAImage* image)
{
    const dim3 BlockDim(16,16);
    dim3 GridDim;
    GridDim.x = (image->width + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (image->height + BlockDim.y - 1) / BlockDim.y;
    bitwise_or<<<GridDim, BlockDim>>>
    (image->d_in, image->d_mask, image->d_out, image->width, image->height);
    cuda_image_swap(image);
}



/* ========================================================================== */
/*                                 Bitwise Not                                */
/* ========================================================================== */
void run_bitwise_not(CUDAImage* image)
{
    const dim3 BlockDim(16,16);
    dim3 GridDim;
    GridDim.x = (image->width + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (image->height + BlockDim.y - 1) / BlockDim.y;
    bitwise_not<<<GridDim, BlockDim>>>
    (image->d_mask, image->d_out, image->width, image->height);
    image->d_mask = image->d_out;
    // cuda_image_swap(image);
}