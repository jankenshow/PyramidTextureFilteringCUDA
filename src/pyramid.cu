#include "pyramid.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

__global__ void downsample_kernel(const float* input, float* output, 
                                const float* kernel, int kernel_size,
                                int input_width, int input_height, int channels,
                                int output_width, int output_height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= output_width || y >= output_height || c >= channels) return;
    
    int kernel_radius = kernel_size / 2;
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Map output coordinates to input coordinates
    float src_x = x / scale;
    float src_y = y / scale;
    
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int src_px = (int)(src_x + kx);
            int src_py = (int)(src_y + ky);
            
            // Handle boundary conditions with reflection
            src_px = max(0, min(input_width - 1, src_px));
            src_py = max(0, min(input_height - 1, src_py));
            
            float weight = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
            int src_idx = (src_py * input_width + src_px) * channels + c;
            
            sum += input[src_idx] * weight;
            weight_sum += weight;
        }
    }
    
    int output_idx = (y * output_width + x) * channels + c;
    output[output_idx] = sum / weight_sum;
}

__global__ void upsample_kernel(const float* input, float* output, 
                              const float* kernel, int kernel_size,
                              int input_width, int input_height, int channels,
                              int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= output_width || y >= output_height || c >= channels) return;
    
    int kernel_radius = kernel_size / 2;
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Map output coordinates to input coordinates
    float src_x = (float)x * input_width / output_width;
    float src_y = (float)y * input_height / output_height;
    
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int src_px = (int)(src_x + kx);
            int src_py = (int)(src_y + ky);
            
            // Handle boundary conditions with reflection
            src_px = max(0, min(input_width - 1, src_px));
            src_py = max(0, min(input_height - 1, src_py));
            
            float weight = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
            int src_idx = (src_py * input_width + src_px) * channels + c;
            
            sum += input[src_idx] * weight;
            weight_sum += weight;
        }
    }
    
    int output_idx = (y * output_width + x) * channels + c;
    output[output_idx] = sum / weight_sum;
}

__global__ void subtract_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

Pyramid::Pyramid(int nlev, float scale, float sigma) 
    : nlev_(nlev), scale_(scale), sigma_(sigma), kernel_size_(5), d_kernel_(nullptr) {
    createGaussianKernel();
}

Pyramid::~Pyramid() {
    if (d_kernel_) {
        cudaFree(d_kernel_);
    }
}

void Pyramid::createGaussianKernel() {
    std::vector<float> kernel_1d(kernel_size_);
    int radius = kernel_size_ / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < kernel_size_; i++) {
        float x = i - radius;
        kernel_1d[i] = exp(-(x * x) / (2 * sigma_ * sigma_));
        sum += kernel_1d[i];
    }
    
    // Normalize
    for (int i = 0; i < kernel_size_; i++) {
        kernel_1d[i] /= sum;
    }
    
    // Create 2D kernel
    std::vector<float> kernel_2d(kernel_size_ * kernel_size_);
    for (int y = 0; y < kernel_size_; y++) {
        for (int x = 0; x < kernel_size_; x++) {
            kernel_2d[y * kernel_size_ + x] = kernel_1d[y] * kernel_1d[x];
        }
    }
    
    // Copy to GPU
    size_t kernel_size = kernel_size_ * kernel_size_ * sizeof(float);
    cudaMalloc(&d_kernel_, kernel_size);
    cudaMemcpy(d_kernel_, kernel_2d.data(), kernel_size, cudaMemcpyHostToDevice);
}

void Pyramid::build(const ImagePtr& input) {
    gaussian_pyramid_.clear();
    laplacian_pyramid_.clear();
    
    gaussian_pyramid_.resize(nlev_);
    laplacian_pyramid_.resize(nlev_ - 1);
    
    // First level is the input image
    gaussian_pyramid_[0] = input;
    
    for (int level = 0; level < nlev_ - 1; level++) {
        // Calculate dimensions for next level
        int curr_width = gaussian_pyramid_[level]->getWidth();
        int curr_height = gaussian_pyramid_[level]->getHeight();
        int channels = gaussian_pyramid_[level]->getChannels();
        
        int next_width = (int)(curr_width * scale_);
        int next_height = (int)(curr_height * scale_);
        
        // Create next Gaussian level
        gaussian_pyramid_[level + 1] = std::make_shared<Image>(next_width, next_height, channels);
        downsample(gaussian_pyramid_[level], gaussian_pyramid_[level + 1]);
        
        // Create Laplacian level
        laplacian_pyramid_[level] = std::make_shared<Image>(curr_width, curr_height, channels);
        auto upsampled = std::make_shared<Image>(curr_width, curr_height, channels);
        upsample(gaussian_pyramid_[level + 1], gaussian_pyramid_[level], upsampled);
        
        // Compute Laplacian = Gaussian - Upsampled
        dim3 block(16, 16, 1);
        dim3 grid((curr_width * curr_height * channels + block.x - 1) / block.x);
        
        subtract_kernel<<<grid, block>>>(
            gaussian_pyramid_[level]->getData(),
            upsampled->getData(),
            laplacian_pyramid_[level]->getData(),
            curr_width * curr_height * channels
        );
        cudaDeviceSynchronize();
    }
}

void Pyramid::downsample(const ImagePtr& input, ImagePtr& output) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (output->getWidth() + block.x - 1) / block.x,
        (output->getHeight() + block.y - 1) / block.y,
        (output->getChannels() + block.z - 1) / block.z
    );
    
    downsample_kernel<<<grid, block>>>(
        input->getData(), output->getData(),
        d_kernel_, kernel_size_,
        input->getWidth(), input->getHeight(), input->getChannels(),
        output->getWidth(), output->getHeight(), scale_
    );
    cudaDeviceSynchronize();
}

void Pyramid::upsample(const ImagePtr& input, const ImagePtr& reference, ImagePtr& output) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (reference->getWidth() + block.x - 1) / block.x,
        (reference->getHeight() + block.y - 1) / block.y,
        (reference->getChannels() + block.z - 1) / block.z
    );
    
    upsample_kernel<<<grid, block>>>(
        input->getData(), output->getData(),
        d_kernel_, kernel_size_,
        input->getWidth(), input->getHeight(), input->getChannels(),
        reference->getWidth(), reference->getHeight()
    );
    cudaDeviceSynchronize();
}