#include "bilateral_filter.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

__global__ void bilateral_filter_kernel(const float* input, const float* guidance, float* output,
                                       const float* spatial_weights, int window_size,
                                       int width, int height, int channels, float sigma_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = window_size;
    float sigma_range_sq_2 = 2.0f * sigma_range * sigma_range;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        int center_idx = (y * width + x) * channels + c;
        float center_guidance[3];
        
        // Get guidance center value
        if (channels == 3) {
            center_guidance[0] = guidance[(y * width + x) * channels + 0];
            center_guidance[1] = guidance[(y * width + x) * channels + 1];
            center_guidance[2] = guidance[(y * width + x) * channels + 2];
        } else {
            center_guidance[0] = guidance[center_idx];
        }
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                // Handle boundary conditions with reflection
                nx = max(0, min(width - 1, nx));
                ny = max(0, min(height - 1, ny));
                
                int neighbor_idx = (ny * width + nx) * channels + c;
                
                // Spatial weight
                float spatial_weight = spatial_weights[(dy + radius) * (2 * radius + 1) + (dx + radius)];
                
                // Range weight
                float range_weight = 1.0f;
                if (channels == 3) {
                    float diff_r = guidance[(ny * width + nx) * channels + 0] - center_guidance[0];
                    float diff_g = guidance[(ny * width + nx) * channels + 1] - center_guidance[1];
                    float diff_b = guidance[(ny * width + nx) * channels + 2] - center_guidance[2];
                    float range_dist_sq = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
                    range_weight = expf(-range_dist_sq / sigma_range_sq_2);
                } else {
                    float diff = guidance[neighbor_idx] - center_guidance[0];
                    range_weight = expf(-(diff * diff) / sigma_range_sq_2);
                }
                
                float total_weight = spatial_weight * range_weight;
                sum += input[neighbor_idx] * total_weight;
                weight_sum += total_weight;
            }
        }
        
        output[center_idx] = (weight_sum > 0) ? sum / weight_sum : input[center_idx];
        output[center_idx] = fmaxf(0.0f, fminf(1.0f, output[center_idx]));
    }
}

BilateralFilter::BilateralFilter() : d_spatial_weights_(nullptr), max_window_size_(0) {
}

BilateralFilter::~BilateralFilter() {
    if (d_spatial_weights_) {
        cudaFree(d_spatial_weights_);
    }
}

void BilateralFilter::computeSpatialWeights(int window_size, float sigma_space) {
    int radius = window_size;
    int weight_size = (2 * radius + 1) * (2 * radius + 1);
    
    // Reallocate if needed
    if (weight_size > max_window_size_) {
        if (d_spatial_weights_) {
            cudaFree(d_spatial_weights_);
        }
        cudaMalloc(&d_spatial_weights_, weight_size * sizeof(float));
        max_window_size_ = weight_size;
    }
    
    // Compute weights on CPU
    std::vector<float> weights(weight_size);
    float sigma_space_sq_2 = 2.0f * sigma_space * sigma_space;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float spatial_dist_sq = dx * dx + dy * dy;
            float weight = exp(-spatial_dist_sq / sigma_space_sq_2);
            weights[(dy + radius) * (2 * radius + 1) + (dx + radius)] = weight;
        }
    }
    
    // Copy to GPU
    cudaMemcpy(d_spatial_weights_, weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
}

void BilateralFilter::apply(const ImagePtr& input, const ImagePtr& guidance, ImagePtr& output,
                          int window_size, float sigma_space, float sigma_range) {
    if (!input || !guidance || !output) {
        throw std::runtime_error("Invalid input images");
    }
    
    if (input->getWidth() != guidance->getWidth() || 
        input->getHeight() != guidance->getHeight() ||
        input->getWidth() != output->getWidth() || 
        input->getHeight() != output->getHeight()) {
        throw std::runtime_error("Image dimensions must match");
    }
    
    // Compute spatial weights
    computeSpatialWeights(window_size, sigma_space);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(
        (input->getWidth() + block.x - 1) / block.x,
        (input->getHeight() + block.y - 1) / block.y
    );
    
    bilateral_filter_kernel<<<grid, block>>>(
        input->getData(), guidance->getData(), output->getData(),
        d_spatial_weights_, window_size,
        input->getWidth(), input->getHeight(), input->getChannels(),
        sigma_range
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed");
    }
}