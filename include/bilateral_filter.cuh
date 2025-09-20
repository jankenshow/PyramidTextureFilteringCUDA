#pragma once

#include "image.h"

class BilateralFilter {
public:
    BilateralFilter();
    ~BilateralFilter();

    void apply(const ImagePtr& input, const ImagePtr& guidance, ImagePtr& output,
               int window_size, float sigma_space, float sigma_range);

private:
    float* d_spatial_weights_;
    int max_window_size_;

    void computeSpatialWeights(int window_size, float sigma_space);
};