#include <iostream>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <chrono> 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/cuda.hpp"

//#include "crow.h"

using namespace std;
using namespace std::chrono; 
using namespace cv;
using namespace cuda;

CascadeClassifier eyes_cascade;


int main(int argc, char **argv) {
    std::cout << "Hello from main.." << std::endl;

    printCudaDeviceInfo(0);
    
    string image_file;
    if  (argc == 1)
    {
        // cout << "Please pass image filename to read as cmd argument" << endl;
        // return -1;
        image_file = "../resources/image.jpg";
    }
    else if (argc == 2)
    {
        image_file = argv[1];
    }

    //// 2
    /// flag value 1 to imread means -- cv2.IMREAD_COLOR:
    /// It specifies to load a color image. 
    /// Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
    cv::Mat frame = cv::imread(image_file.c_str(), 1);

    if (frame.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file.c_str());
        return -1;
    }
    std::cout << "cv::imread Test: image reading done successfully.. " << image_file << std::endl;

    dnn::Net net;
    std::string model_path = "../resources/effi-model.pb";
    net = cv::dnn::readNet(model_path);
    std::cout << "dnn test: net loaded successfully..." << std::endl;

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // ---------------------------------------------------------------
    // test
    cv::Mat roi = frame.clone();

    cv::Mat blob;
    // cv::cuda::GpuMat blob;
    cv::dnn::blobFromImage(roi, blob, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);


    // // Sets the input to the network
    // net.setInput(blob);

    // auto start = high_resolution_clock::now();

    // // Runs the forward pass to get output of the output layers
    // std::vector<Mat> outs;
    // net.forward(outs); //, getOutputsNames(net)
    // // std::cout << "outs[0]" << outs[0] << std::endl;

    // // create 1d matrix from vectors of matrix
    // cv::Mat out = outs[0];

    // // get top prediction and array index from opencv output matrix
    // // https://docs.opencv.org/3.4/d9/d8d/samples_2dnn_2classification_8cpp-example.html
    // Point classIdPoint;
    // double confidence;
    // minMaxLoc(out, 0, &confidence, 0, &classIdPoint);
    // int classId = classIdPoint.x;
    // cout << "predicted classId: " << classId << endl;
    // cout << "confidence: " << confidence << endl;
    // // ## -----------------------------------------------------------

    // // After function call 
    // auto stop = high_resolution_clock::now();
    // // Subtract stop and start timepoints and 
    // // cast it to required unit. Predefined units 
    // // are nanoseconds, microseconds, milliseconds, 
    // // seconds, minutes, hours. Use duration_cast() 
    // // function. 
    // auto duration = duration_cast<microseconds>(stop - start); 

    // // To get the value of duration use the count() 
    // // member function on the duration object 
    // cout << "duration.count() microseconds(10e-6):" << duration.count()  << endl; 
    // cout << "----------------------------------------------------------" << endl;

    std::vector<Mat> outs;
    auto outputNames = net.getUnconnectedOutLayersNames();

	// warmup
	for(int i = 0; i < 3; i++)
	{
		net.setInput(blob);
		net.forward(outs, outputNames);
	}

	// benckmark
	auto start = std::chrono::steady_clock::now();
    int loopCount = 1000;
	for(int i = 0; i < loopCount; i++)
	{
		net.setInput(blob);
		net.forward(outs, outputNames);
	}
	auto end = std::chrono::steady_clock::now();

	std::chrono::milliseconds timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>((end - start) / loopCount);
	std::cout << "Time per inference: " << timeTaken.count() << " ms" << std::endl;
    std::cout << "FPS: " << 1000.0 / (timeTaken.count()) << std::endl;
}