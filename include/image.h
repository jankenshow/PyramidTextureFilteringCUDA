#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class Image {
public:
    Image(int width, int height, int channels);
    Image(const cv::Mat& mat);
    ~Image();

    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getChannels() const { return channels_; }
    float* getData() const { return data_; }
    size_t getSize() const { return width_ * height_ * channels_; }

    void copyFromHost(const float* host_data);
    void copyToHost(float* host_data) const;
    void copyFromDevice(const float* device_data);
    void copyToDevice(float* device_data) const;

    cv::Mat toCvMat() const;
    void fromCvMat(const cv::Mat& mat);

private:
    int width_;
    int height_;
    int channels_;
    float* data_;
};

using ImagePtr = std::shared_ptr<Image>;