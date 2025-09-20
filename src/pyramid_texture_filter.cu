#include "pyramid_texture_filter.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cmath>

__global__ void add_images_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void resize_kernel(const float* input, float* output,
                            int input_width, int input_height, int channels,
                            int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x1 = (int)floor(src_x);
    int y1 = (int)floor(src_y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    float fx = src_x - x1;
    float fy = src_y - y1;
    
    x1 = max(0, min(input_width - 1, x1));
    y1 = max(0, min(input_height - 1, y1));
    x2 = max(0, min(input_width - 1, x2));
    y2 = max(0, min(input_height - 1, y2));
    
    for (int c = 0; c < channels; c++) {
        float v1 = input[(y1 * input_width + x1) * channels + c];
        float v2 = input[(y1 * input_width + x2) * channels + c];
        float v3 = input[(y2 * input_width + x1) * channels + c];
        float v4 = input[(y2 * input_width + x2) * channels + c];
        
        float i1 = v1 * (1 - fx) + v2 * fx;
        float i2 = v3 * (1 - fx) + v4 * fx;
        float value = i1 * (1 - fy) + i2 * fy;
        
        output[(y * output_width + x) * channels + c] = value;
    }
}

PyramidTextureFilter::PyramidTextureFilter(float sigma_s, float sigma_r, int nlev, float scale)
    : sigma_s_(sigma_s), sigma_r_(sigma_r), nlev_(nlev), scale_(scale) {
    pyramid_ = std::make_shared<Pyramid>(nlev_, scale_);
}

PyramidTextureFilter::~PyramidTextureFilter() {
}

ImagePtr PyramidTextureFilter::apply(const ImagePtr& input) {
    if (!input) {
        throw std::runtime_error("Invalid input image");
    }
    
    std::cout << "Building pyramid..." << std::endl;
    pyramid_->build(input);
    
    const auto& G = pyramid_->getGaussianPyramid();
    const auto& L = pyramid_->getLaplacianPyramid();
    
    std::cout << "Gaussian pyramid: " << G.size() << ", Laplacian pyramid: " << L.size() << std::endl;
    
    // Start from the coarsest level
    auto result = std::make_shared<Image>(
        G.back()->getWidth(), 
        G.back()->getHeight(), 
        G.back()->getChannels()
    );
    
    // Copy coarsest level
    cudaMemcpy(result->getData(), G.back()->getData(), 
               G.back()->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Process each level from coarse to fine
    for (int level = G.size() - 2; level >= 0; level--) {
        std::cout << "Processing level " << level << std::endl;
        
        // Calculate adaptive parameters
        float adaptive_sigma_s = sigma_s_ * pow(scale_, level);
        int w1 = (int)ceil(adaptive_sigma_s * 0.5f + 1);
        int w2 = (int)ceil(adaptive_sigma_s * 2.0f + 1);
        
        // Create temporary images
        auto result_up = std::make_shared<Image>(
            G[level]->getWidth(), 
            G[level]->getHeight(), 
            G[level]->getChannels()
        );
        
        auto result_hat = std::make_shared<Image>(
            G[level]->getWidth(), 
            G[level]->getHeight(), 
            G[level]->getChannels()
        );
        
        auto result_lap = std::make_shared<Image>(
            G[level]->getWidth(), 
            G[level]->getHeight(), 
            G[level]->getChannels()
        );
        
        auto result_out = std::make_shared<Image>(
            G[level]->getWidth(), 
            G[level]->getHeight(), 
            G[level]->getChannels()
        );
        
        auto result_refine = std::make_shared<Image>(
            G[level]->getWidth(), 
            G[level]->getHeight(), 
            G[level]->getChannels()
        );
        
        // Upsample current result
        std::cout << "Upsampling from " << result->getWidth() << "x" << result->getHeight() 
                  << " to " << G[level]->getWidth() << "x" << G[level]->getHeight() << std::endl;
        
        dim3 block(16, 16);
        dim3 grid(
            (G[level]->getWidth() + block.x - 1) / block.x,
            (G[level]->getHeight() + block.y - 1) / block.y
        );
        
        resize_kernel<<<grid, block>>>(
            result->getData(), result_up->getData(),
            result->getWidth(), result->getHeight(), result->getChannels(),
            G[level]->getWidth(), G[level]->getHeight()
        );
        cudaDeviceSynchronize();
        
        // First bilateral filtering
        bilateral_filter_.apply(result_up, G[level], result_hat, w1, adaptive_sigma_s, sigma_r_);
        
        // Add Laplacian detail
        dim3 add_grid((G[level]->getSize() + 255) / 256);
        dim3 add_block(256);
        
        add_images_kernel<<<add_grid, add_block>>>(
            result_hat->getData(), L[level]->getData(), result_lap->getData(),
            G[level]->getSize()
        );
        cudaDeviceSynchronize();
        
        // Second bilateral filtering
        bilateral_filter_.apply(result_lap, result_hat, result_out, w2, adaptive_sigma_s, sigma_r_);
        
        // Final enhancement
        bilateral_filter_.apply(result_out, result_out, result_refine, w2, adaptive_sigma_s, sigma_r_);
        
        result = result_refine;
    }
    
    return result;
}