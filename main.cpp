#include <opencv2/opencv.hpp>
#include <chrono>
#include <openslide/openslide.h>
#include "inc/cuda_wrapper.h"
#include "inc/gaussian.hpp"
#include "inc/morphology.hpp"
#include "iostream"

using namespace cv;
using namespace std;
using namespace std::chrono;

#define SOURCE "rsc/2.png"
#define DESTINATION "out/Final-.png"

int main()
{
    // init
    auto mat = loadImage(2);
    CUDAImage* img = cuda_image_create(&mat);
    auto start = high_resolution_clock::now();
    // filter
    run_gaussian_blur(img,  new GaussianArgs(32, 1.5));
    run_otsu_threshold(img);
    run_morph_filter(img, new MorphSize(2), OPEN_CLOSE, ELEMENT::CROSS, 5);
    // run_boundary_trace(img);
    run_floodfill(img);
    run_bitwise_not(img);
    run_bitwise_or(img);
    // save, clean, exit
    auto duration = duration_cast<milliseconds>
        (high_resolution_clock::now() - start);
    cuda_image_mat(img, &mat);
    cuda_image_destroy(img);
    imwrite(DESTINATION, mat);
    cout << "\n Duration: " << duration.count() << " ms\n" << endl;
    return 0;
}