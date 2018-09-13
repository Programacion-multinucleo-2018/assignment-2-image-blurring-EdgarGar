//Algunas partes del codigo fueron obtenidas de clase
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

// Funcion de imagen pixelBluePixelr
void blur(const cv::Mat& input, cv::Mat& output) {
    cout << "Input image step: " << input.step << " rows: " << input.rows << " colums: " << input.cols << endl;

    // Matriz
    //Valoresn en x
   for(int i = 0; i < input.rows; i++) {
     //Valores en Y
       for(int j = 0; j < input.cols; j++) {
           int pixelBlue = 0;
           int pixelGreen = 0;
           int pixelRed = 0;
           int tm = 0;

           // Pixeles Vecions
           for(int filX=-2; filX<3; filX++) {
               for(int filY=-2; filY<3; filY++) {
                   int idX = i+filX;
                   int idY = j+filY;

                   //Bordes
                   if((idY>0 && idY < input.cols) && (idX>0 && idX < input.rows)) {
                       pixelBlue += input.at<cv::Vec3b>(idX, idY)[0];
                       pixelGreen += input.at<cv::Vec3b>(idX, idY)[1];
                       pixelRed += input.at<cv::Vec3b>(idX, idY)[2];
                       tm++;
                   }
               }
           }
           // Promedio
           output.at<cv::Vec3b>(i, j)[0] = floor(pixelBlue/tm);
           output.at<cv::Vec3b>(i, j)[1] = floor(pixelGreen/tm);
           output.at<cv::Vec3b>(i, j)[2] = floor(pixelRed/tm);
       }
   }
}

int main(int argc, char *argv[]) {
    // Lectura de la Imagen
    string imagePath;
    if (argc < 2)
      imagePath = "spiderman.jpg";
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

    //Tiempo Promedio
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
