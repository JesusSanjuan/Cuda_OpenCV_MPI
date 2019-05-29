

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc.hpp>



// GPU constant memory to hold our kernels (extremely fast access time)
__constant__ float convolutionKernelStore[256];

/**
* Convolution function for cuda.  Destination is expected to have the same width/height as source, but there will be a border
* of floor(kWidth/2) pixels left and right and floor(kHeight/2) pixels top and bottom
*
* @param source      Source image host pinned memory pointer
* @param width       Source image width
* @param height      Source image height
* @param paddingX    source image padding along x
* @param paddingY    source image padding along y
* @param kOffset     offset into kernel store constant memory
* @param kWidth      kernel width
* @param kHeight     kernel height
* @param destination Destination image host pinned memory pointer
*/
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
	//Distribucion de indices para la localizacion de los pixeles
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float sum = 0.0;
	int   pWidth = kWidth / 2;
	int   pHeight = kHeight / 2;
	//Solo ejecutamos para pixeles validos
	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight; j++)
		{
			for (int i = -pWidth; i <= pWidth; i++)
			{
				//obteniendo el peso para la locacion
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				float w = convolutionKernelStore[(kj * kWidth) + ki + kOffset];
				sum += w * float(source[((y + j) * width) + (x + i)]);
			}
		}
	}
	//Promedio de la suma
	destination[(y * width) + x] = (unsigned char)sum;
}

//utilizacion del teorema de pitagoras a lo largo del vector en el gpu
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float af = float(a[idx]);
	float bf = float(b[idx]);

	c[idx] = (unsigned char)sqrtf(af*af + bf * bf);
}

//creacion de un buffer de imagenes, regresando al host, pasando del dispositivo al host de puntero a puntero
unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{
	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

extern "C" int main(int argc, char** argv)
{
	//abrir la camara web
	cv::VideoCapture camera(0);
	cv::Mat          frame;
	if (!camera.isOpened())
		return -1;

	//creando las ventanas de captura
	cv::namedWindow("Entrada");
	cv::namedWindow("Escala de grises");
	cv::namedWindow("Blur");
	cv::namedWindow("Sobel");

	//Creando los eventos de temporizacion de cuda
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Creando el kernel gausiano (sum = 159)
	const float gaussianKernel5x5[25] =
	{
		2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f,
		4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
		2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f,
	};

	cudaMemcpyToSymbol(convolutionKernelStore,gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);

	const size_t gaussianKernel5x5Offset = 0;

	// Gradientes de sobel
	const float sobelGradientX[9] =
	{
		-1.f, 0.f, 1.f,
		-2.f, 0.f, 2.f,
		-1.f, 0.f, 1.f,
	};
	const float sobelGradientY[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};

	cudaMemcpyToSymbol(convolutionKernelStore,sobelGradientX, sizeof(sobelGradientX),sizeof(gaussianKernel5x5));

	cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientY, sizeof(sobelGradientY),sizeof(gaussianKernel5x5) + sizeof(sobelGradientX));

	const size_t sobelGradientXOffset =	sizeof(gaussianKernel5x5) / sizeof(float);

	const size_t sobelGradientYOffset =	sizeof(sobelGradientX) / sizeof(float) + sobelGradientXOffset;

	//Creamos las imagenes compartidas- una inicial y una para el resultado
	camera >> frame;
	unsigned char *sourceDataDevice, *blurredDataDevice, *edgesDataDevice;
	cv::Mat source(frame.size(), CV_8U,createImageBuffer(frame.size().width * frame.size().height,&sourceDataDevice));

	cv::Mat blurred(frame.size(), CV_8U,createImageBuffer(frame.size().width * frame.size().height,&blurredDataDevice));

	cv::Mat edges(frame.size(), CV_8U,createImageBuffer(frame.size().width * frame.size().height,&edgesDataDevice));

	//Creando dos imagenes temporales en el GPU (para mantener los gradientes de sobel)
	unsigned char *deviceGradientX, *deviceGradientY;
	cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
	cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);

	//Ciclo de captura de imagenes
	while (1)
	{
		//captura de la imagen y almacenamiento a gris
		camera >> frame;
		cv::cvtColor(frame, source, CV_BGR2GRAY);

		//creacion del evento de inicio
		cudaEventRecord(start);
		{
			//configuracion de los parametros de lanzamiento del kernel de convolucion
			dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
			dim3 cthreads(16, 16);

			//configuracion de los parametros de lanzamiento del kernel de pitagoras
			dim3 pblocks(frame.size().width * frame.size().height / 256);
			dim3 pthreads(256, 1);

			//Lanzamiento del kernel para ejecucion del filtro de Gauss
			convolve << <cblocks, cthreads >> > (sourceDataDevice, frame.size().width,frame.size().height, 0, 0, gaussianKernel5x5Offset, 5, 5, blurredDataDevice);
			//Lanzamiento del gradiente de sobel
			//primero obtenemos cada uno de los gradientes X y Y 
			//de kernels de 5x5 y posteriormente obtenemos el producto
			convolve <<<cblocks, cthreads >>> (blurredDataDevice, frame.size().width,frame.size().height, 2, 2, sobelGradientXOffset, 3, 3, deviceGradientX);
			convolve <<<cblocks, cthreads >>> (blurredDataDevice, frame.size().width,frame.size().height, 2, 2, sobelGradientYOffset, 3, 3, deviceGradientY);
			pythagoras <<<pblocks, pthreads >>> (deviceGradientX, deviceGradientY, edgesDataDevice);

			cudaThreadSynchronize();
		}
		cudaEventRecord(stop);

		//Mostrando el tiempo transcurrido
		float ms = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		std::cout << "Tiempo transcurrido GPU : " << ms << " milisegundos" << std::endl;

		
		// mostramos los resultados de los procesamientos
		cv::imshow("Entrada", frame);
		cv::imshow("Escala de grises", source);
		cv::imshow("Blur", blurred);
		cv::imshow("Sobel", edges);

		if (cv::waitKey(1) == 27) break;
	}
	// Limpieza de variables y fin de la ejecución
	cudaFreeHost(source.data);
	cudaFreeHost(blurred.data);
	cudaFreeHost(edges.data);
	cudaFree(deviceGradientX);
	cudaFree(deviceGradientY);

	return 0;
}