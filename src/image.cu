#include "image.h"
#include <cuda_runtime.h>
#include <stdexcept>

Image::Image(int width, int height, int channels) 
    : width_(width), height_(height), channels_(channels) {
    size_t size = getSize() * sizeof(float);
    cudaError_t err = cudaMalloc(&data_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for image");
    }
}

Image::Image(const cv::Mat& mat) {
    cv::Mat float_mat;
    if (mat.type() == CV_8UC3) {
        mat.convertTo(float_mat, CV_32FC3, 1.0/255.0);
    } else if (mat.type() == CV_8UC1) {
        mat.convertTo(float_mat, CV_32FC1, 1.0/255.0);
    } else {
        float_mat = mat.clone();
    }
    
    width_ = float_mat.cols;
    height_ = float_mat.rows;
    channels_ = float_mat.channels();
    
    size_t size = getSize() * sizeof(float);
    cudaError_t err = cudaMalloc(&data_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for image");
    }
    
    copyFromHost(reinterpret_cast<const float*>(float_mat.data));
}

Image::~Image() {
    if (data_) {
        cudaFree(data_);
    }
}

void Image::copyFromHost(const float* host_data) {
    size_t size = getSize() * sizeof(float);
    cudaError_t err = cudaMemcpy(data_, host_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from host to device");
    }
}

void Image::copyToHost(float* host_data) const {
    size_t size = getSize() * sizeof(float);
    cudaError_t err = cudaMemcpy(host_data, data_, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from device to host");
    }
}

void Image::copyFromDevice(const float* device_data) {
    size_t size = getSize() * sizeof(float);
    cudaError_t err = cudaMemcpy(data_, device_data, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from device to device");
    }
}

void Image::copyToDevice(float* device_data) const {
    size_t size = getSize() * sizeof(float);
    cudaError_t err = cudaMemcpy(device_data, data_, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from device to device");
    }
}

cv::Mat Image::toCvMat() const {
    std::vector<float> host_data(getSize());
    copyToHost(host_data.data());
    
    cv::Mat result;
    if (channels_ == 3) {
        cv::Mat float_mat(height_, width_, CV_32FC3, host_data.data());
        float_mat.convertTo(result, CV_8UC3, 255.0);
    } else if (channels_ == 1) {
        cv::Mat float_mat(height_, width_, CV_32FC1, host_data.data());
        float_mat.convertTo(result, CV_8UC1, 255.0);
    }
    
    return result;
}

void Image::fromCvMat(const cv::Mat& mat) {
    cv::Mat float_mat;
    if (mat.type() == CV_8UC3) {
        mat.convertTo(float_mat, CV_32FC3, 1.0/255.0);
    } else if (mat.type() == CV_8UC1) {
        mat.convertTo(float_mat, CV_32FC1, 1.0/255.0);
    } else {
        float_mat = mat.clone();
    }
    
    if (width_ != float_mat.cols || height_ != float_mat.rows || channels_ != float_mat.channels()) {
        throw std::runtime_error("Image dimensions do not match");
    }
    
    copyFromHost(reinterpret_cast<const float*>(float_mat.data));
}