#include <iostream>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "pyramid_texture_filter.h"

int main() {
    try {
        // I/O settings
        std::string input_path = "../data/";
        std::string output_path = "../output/";
        
        // Create output directory
        std::filesystem::create_directories(output_path);
        
        // Parameters
        std::string img_name = "08.png";
        float sigma_s = 7.0f;
        float sigma_r = 0.08f;
        int nlev = 11;
        float scale = 0.8f;
        
        std::cout << "Loading image: " << input_path + img_name << std::endl;
        
        // Read image
        cv::Mat input_mat = cv::imread(input_path + img_name);
        if (input_mat.empty()) {
            std::cerr << "Error: Could not read image " << input_path + img_name << std::endl;
            return -1;
        }
        
        // Convert BGR to RGB
        cv::cvtColor(input_mat, input_mat, cv::COLOR_BGR2RGB);
        std::cout << "Image size: " << input_mat.cols << "x" << input_mat.rows << std::endl;
        
        // Create image object
        auto input_image = std::make_shared<Image>(input_mat);
        
        // Create filter
        std::cout << "Creating pyramid texture filter..." << std::endl;
        std::cout << "Parameters: sigma_s=" << sigma_s << ", sigma_r=" << sigma_r 
                  << ", nlev=" << nlev << ", scale=" << scale << std::endl;
        
        PyramidTextureFilter filter(sigma_s, sigma_r, nlev, scale);
        
        // Apply filtering
        std::cout << "Applying filter..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto result_image = filter.apply(input_image);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
        
        // Convert result to OpenCV format
        cv::Mat result_mat = result_image->toCvMat();
        
        // Convert RGB back to BGR for saving
        cv::cvtColor(result_mat, result_mat, cv::COLOR_RGB2BGR);
        
        // Save result
        std::string output_filename = output_path + "cuda_result_" + img_name;
        bool success = cv::imwrite(output_filename, result_mat);
        
        if (success) {
            std::cout << "Result saved to: " << output_filename << std::endl;
        } else {
            std::cerr << "Error: Could not save result image" << std::endl;
            return -1;
        }
        
        std::cout << "Processing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}