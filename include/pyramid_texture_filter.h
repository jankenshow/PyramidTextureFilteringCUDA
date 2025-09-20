#pragma once

#include "image.h"
#include "pyramid.cuh"
#include "bilateral_filter.cuh"
#include <memory>

class PyramidTextureFilter {
public:
    PyramidTextureFilter(float sigma_s = 5.0f, float sigma_r = 0.05f, 
                        int nlev = 11, float scale = 0.8f);
    ~PyramidTextureFilter();

    ImagePtr apply(const ImagePtr& input);

private:
    float sigma_s_;
    float sigma_r_;
    int nlev_;
    float scale_;

    PyramidPtr pyramid_;
    BilateralFilter bilateral_filter_;
};