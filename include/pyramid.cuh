#pragma once

#include "image.h"
#include <vector>
#include <memory>

class Pyramid {
public:
    Pyramid(int nlev, float scale, float sigma = 1.0f);
    ~Pyramid();

    void build(const ImagePtr& input);
    
    const std::vector<ImagePtr>& getGaussianPyramid() const { return gaussian_pyramid_; }
    const std::vector<ImagePtr>& getLaplacianPyramid() const { return laplacian_pyramid_; }

private:
    int nlev_;
    float scale_;
    float sigma_;
    int kernel_size_;
    float* d_kernel_;
    
    std::vector<ImagePtr> gaussian_pyramid_;
    std::vector<ImagePtr> laplacian_pyramid_;

    void createGaussianKernel();
    void downsample(const ImagePtr& input, ImagePtr& output);
    void upsample(const ImagePtr& input, const ImagePtr& reference, ImagePtr& output);
};

using PyramidPtr = std::shared_ptr<Pyramid>;