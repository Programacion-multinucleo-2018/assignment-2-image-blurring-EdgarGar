#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Funcion de imagen bluePixelPixelr
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int step) {
    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    // Matriz
    //Valoresn en X y Y
    if((xIndex<width) && (yIndex<height)) {
        //Location of pixel in input and output
        const int tid = yIndex * step + (3 * xIndex);
        int bluePixel = 0;
        int greenixel = 0;
        int redPixel = 0;
        int Average = 0;

        // Pixeles vecinos
        for(int filX=-2; filX<3; filX++) {
            for(int filY=-2; filY<3; filY++) {
                int tid = (yIndex+filY) * step + (3 * (xIndex+filX));

                // Bordes
                if((xIndex+filX)%width>1 && (yIndex+filY)%height>1) {
                    bluePixel += input[tid];
                    greenixel += input[tid+1];
                    redPixel += input[tid+2];
                    Average++;
                }
            }
        }
        // Promedio
        output[tid] = static_cast<unsigned char>(bluePixel/Average);
        output[tid+1] = static_cast<unsigned char>(greenixel/Average);
        output[tid+2] = static_cast<unsigned char>(redPixel/Average);
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
	const dim3 block(1024, 1024);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("grid.x %d, grid.y %d, block.x %d block.y %d \n", grid.x, grid.y, block.x, block.y);


	// Launch the color conversion kernel
  auto start_cpu =  std::chrono::high_resolution_clock::now();
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));
  auto end_cpu =  std::chrono::high_resolution_clock::now();

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]) {
    // Read input image
    string imagePath;
    if (argc < 2)
      imagePath = "spiderman.jpg"
    else
      imagePath = argv[1];

    //Lee la imagen desde dentro
    cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

    //Crear una salida de la imagen
    cv::Mat output(input.rows, input.cols, CV_8UC3);

    //Llamada de la funcion
    blur(input, output);


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
