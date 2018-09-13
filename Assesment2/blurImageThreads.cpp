//Algunas partes del codigo fueron obtenidas de clase
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

// Funcion de imagen bluePixelr
void blur(const cv::Mat& input, cv::Mat& output) {
    cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

    int i;
    int j;
    int filX;
    int filY;

    #pragma omp parallel for private (i, y, filX, filY) shared(input, output)

    // Matriz
    //Valoresn en x
    for(i = 0; i < input.rows; i++) {
      //Valores en Y
        for(j = 0; j < input.cols; j++) {
            int bluePixel = 0;
            int greenPixel = 0;
            int redPixel = 0;
            int average = 0;

            // Pixeles vecinos
            for(filX=-2; filX<3; filX++) {
                for(filY=-2; filY<3; filY++) {
                    int idyn = i+filX;
                    int idyn = j+filY;

                    // Bordes
                    if((idyn>0 && idyn < input.cols) && (idyn>0 && idyn < input.rows)) {
                        bluePixel += input.at<cv::Vec3b>(idyn, idyn)[0];
                        greenPixel += input.at<cv::Vec3b>(idyn, idyn)[1];
                        redPixel += input.at<cv::Vec3b>(idyn, idyn)[2];
                        average++;
                    }
                }
            }
            // Promedio
            output.at<cv::Vec3b>(i, j)[0] = bluePixel/average;
            output.at<cv::Vec3b>(i, j)[1] = greenPixel/average;
            output.at<cv::Vec3b>(i, j)[2] = redPixel/average;
        }
    }
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
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    blur(input, output);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("Blur elapsed %f ms\n", duration_ms.average());

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
