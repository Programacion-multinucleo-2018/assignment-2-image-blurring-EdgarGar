#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include "common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

//Funcion de imagen pixelBluePixelr
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int step) {
    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    // Matriz
    //Valoresn en X y Y
    if((xIndex < width) && (yIndex < height)) {
        //Location of pixel in input and output
        const int tid = yIndex * step + (3 * xIndex);
        int pixelBlue = 0;
        int pixelGreen = 0;
        int pixelRed = 0;
        int tm = 0;

        // Pixeles vecinos
        for(int filX=-2; filX<3; filX++) {
            for(int filY=-2; filY<3; filY++) {
                int tid = (yIndex+filY) * step + (3 * (xIndex+filX));

                // Bordes
                if((xIndex+filX)%width>1 && (yIndex+filY)%height>1) {
                    pixelBlue += input[tid];
                    pixelGreen += input[tid+1];
                    pixelRed += input[tid+2];
                    tm++;
                }
            }
        }
        // Promedio
        output[tid] = static_cast<unsigned char>(pixelBlue/tm);
        output[tid+1] = static_cast<unsigned char>(pixelGreen/tm);
        output[tid+2] = static_cast<unsigned char>(pixelRed/tm);
    }
}

void blur(const cv::Mat& input, cv::Mat& output)
{

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_output, grayBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]) {
    // Lectura de la imagen
    string imagePath;
    if (argc < 2)
      imagePath = "imagenMedia.jpg";
    else
      imagePath = argv[1];

    //Lee la imagen desde dentro
    cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

    //Crear una salida de la imagen
    cv::Mat output(input.rows, input.cols, CV_8UC3);

    //Llamada de la funcion
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    blur(input, output);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Timepo en compilar
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("blur elapsed %f ms\n", duration_ms.count());

    //Allow the windows to resize
    namedWindow("Input", cv::WINDOW_NORMAL);
    namedWindow("Output", cv::WINDOW_NORMAL);

    //Show the input and output
    imshow("Input", input);
    imshow("Output", output);

    //Wait for key press
    cv::waitKey();

    return 0;
}
