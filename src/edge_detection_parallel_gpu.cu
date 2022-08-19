/**
 * @file edge_detection_parallel_gpu.cu
 * @author jz544 & cc2396
 * @brief 
 * @version 1.0
 * @date 2022-05-01
 * 
 * compile: nvcc -o edge_detection_parallel_gpu edge_detection_parallel_gpu.cu -lm
 * run: ./edge_detection_parallel_gpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
#include <locale>
// #include <cublas_v2.h>
// #include <cusolverDn.h>
#include <math.h>

#define IMG_PATH "./lena.pgm"
#define SAVE_PATH "./"
#define NUM_IMGS 100

typedef struct {
	char version[3]; 
	int width;
	int height;
	int maxGrayLevel;
	int **imageData;
	int **gx;
	int **gy;
} pgm;

void read_image(char* dir, pgm* image);   // used for reading image
void read_comments(FILE *input_image);    // used for reading image
int isspace(int argument); 			      // used for reading image
void padding(pgm* image);                 // padding 
void init_pgm_image(pgm* out, pgm image); // init pgm image for output
void min_max_normalization(pgm* image, int** matrix); // min max normalization
int convolution(pgm* image, int kernel[3][3], int row, int col);
void write_pgm_file(pgm* image, char* dir, int** matrix);

__global__ void sobel_edge_filter_cuda(int* dataIn, int* dataOut, int imgHeight,  int imgWidth) {
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;
    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth +  xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth  + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) *  imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) *  imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

int main(int argc, char* argv[]) {
    pgm image, out_image;   // define the input image and output image
    int num_of_imgs;
    dim3 grid(10,10);
    dim3 block(30,30);
    // number of threads = 30 * 30 * 30 * 30 = 900 * 900 = 810000
	if (argc == 1) {
		printf("[INFO] no arguments, use default number of images\n");
		printf("[INFO] number of images: %s\n", argv[1]);
		num_of_imgs = NUM_IMGS;
	}
	else {
		printf("[INFO] get num of images from command line\n");
		printf("[INFO] number of images: %s\n", argv[1]);
		num_of_imgs = atoi(argv[1]);
	}

    // read image 
    printf("[INFO] test image from %s\n", IMG_PATH);
    read_image(IMG_PATH, &image);
    padding(&image);
    init_pgm_image(&out_image, image);
    // allocate memory on device for input and output image'
    int *device_input_img, *device_output_img;
    int *host_flatten_input_img, *host_flatten_output_img;
    // convert input image into 1D array
    host_flatten_input_img = (int*)malloc(sizeof(int) * image.width * image.height);
    host_flatten_output_img = (int*)malloc(sizeof(int) * out_image.width * out_image.height);
    for (int i = 0; i < image.height; i++) {
        for (int j = 0; j < image.width; j++) {
            host_flatten_input_img[i * image.width + j] = image.imageData[i][j];
        }
    }
    cudaMalloc((void**)&device_input_img, image.width * image.height * sizeof(int));
    cudaMalloc((void**)&device_output_img, out_image.width * out_image.height * sizeof(int));
    struct timeval start, end;   // start and stop timer
    float el_time;               // elapsed time
    gettimeofday(&start, NULL);  // start counting time
    // kernel
	for (int i = 0; i < num_of_imgs; i++) {
        sobel_edge_filter_cuda<<<grid, block>>>(device_input_img, device_output_img, image.height, image.width);
	}
    cudaDeviceSynchronize();
	// kernel
    gettimeofday(&end, NULL);   // stop counting time
    el_time = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    // copy the output image from device to host
    cudaMemcpy(host_flatten_output_img, device_output_img, out_image.width * out_image.height * sizeof(int), cudaMemcpyDeviceToHost);
    // convert output image into 2D array
    for (int i = 0; i < out_image.height; i++) {
        for (int j = 0; j < out_image.width; j++) {
            out_image.imageData[i][j] = host_flatten_output_img[i * out_image.width + j];
        }
    }
    min_max_normalization(&out_image, out_image.imageData);
	min_max_normalization(&out_image, out_image.gx);
	min_max_normalization(&out_image, out_image.gy);
	write_pgm_file(&out_image, "./gradient.pgm", out_image.imageData);
	write_pgm_file(&out_image, "./gradient_X.pgm", out_image.gx);
	write_pgm_file(&out_image, "./gradient_Y.pgm", out_image.gy);
	printf("[INFO] results has been saved in current directory.\n");
	printf("[INFO] time consumption: %e s\n", el_time);
	free(image.imageData);
	free(out_image.imageData);
	free(out_image.gx);
	free(out_image.gy);
    return 0;
}

void read_comments(FILE *input_image) {
	char ch;
	char line[100];

	while ((ch = fgetc(input_image)) != EOF && (isspace(ch)))  {
		;
    }
	if (ch == '#') {
        fgets(line, sizeof(line), input_image);
    } 
	else {
		fseek(input_image, -2L, SEEK_CUR);
	}
}

void read_image(char* dir, pgm* image) {
	FILE* input_image; 
	int i, j, num;

	input_image = fopen(dir, "rb");
	if (input_image == NULL) {
		printf("[ERROR] given figure is not found:%s\n", dir);
		return;
	} 
	
	fgets(image->version, sizeof(image->version), input_image);
	read_comments(input_image);

	fscanf(input_image, "%d %d %d", &image->width, &image->height, &image->maxGrayLevel);
	
	image->imageData = (int**) calloc(image->height, sizeof(int*));
	for(i = 0; i < image->height; i++) {
		image->imageData[i] = (int*) calloc(image->width, sizeof(int));
	}
	
	if (strcmp(image->version, "P2") == 0) {
		for (i = 0; i < image->height; i++) {
			for (j = 0; j < image->width; j++) {
				fscanf(input_image, "%d", &num);
				image->imageData[i][j] = num;
			}
		}	
	}
	else if (strcmp(image->version, "P5") == 0) {
		char *buffer;
		int buffer_size = image->height * image->width;
		buffer = (char*) malloc( ( buffer_size + 1) * sizeof(char));
		
		if(buffer == NULL) {
			printf("Can not allocate memory for buffer! \n");
			return;
		}
		fread(buffer, sizeof(char), image->width * image-> height, input_image);
		for (i = 0; i < image->height * image ->width; i++) {
			image->imageData[i / (image->width)][i % image->width] = buffer[i];
		}
		free(buffer);
	}
	fclose(input_image);
	printf("[INFO] pgm version: %s \tWidth: %d \tHeight: %d \tMaximum Gray Level: %d \n", image->version, image->width, image->height, image->maxGrayLevel);
}

void padding(pgm* image) {
	int i;
	for (i = 0; i < image->width; i++) {
		image->imageData[0][i] = 0;
		image->imageData[image->height - 1][i] = 0;
	}
	
	for (i = 0; i < image->height; i++) {
		image->imageData[i][0] = 0;
		image->imageData[i][image->width - 1] = 0;
	} 
}

void init_pgm_image(pgm* out, pgm image) {
	int i, j;
	strcpy(out->version, image.version);
	out->width = image.width;
	out->height = image.height;
	out->maxGrayLevel = image.maxGrayLevel;
	
	out->imageData = (int**) calloc(out->height, sizeof(int*));
	for(i = 0; i < out->height; i++) {
		out->imageData[i] = (int*)calloc(out->width, sizeof(int));
	}
	
	out->gx = (int**) calloc(out->height, sizeof(int*));
	for(i = 0; i < out->height; i++) {
		out->gx[i] = (int*)calloc(out->width, sizeof(int));
	}
	
	out->gy = (int**) calloc(out->height, sizeof(int*));
	for(i = 0; i < out->height; i++) {
		out->gy[i] = (int*)calloc(out->width, sizeof(int));
	}
	
	for(i = 0; i < out->height; i++) {
		for(j = 0; j < out->width; j++) {
			out->imageData[i][j] = image.imageData[i][j];
			out->gx[i][j] = image.imageData[i][j];
			out->gy[i][j] = image.imageData[i][j];
		};
	}
}

int convolution(pgm* image, int kernel[3][3], int row, int col) {
	int i, j, sum = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			sum += image->imageData[i + row][j + col] * kernel[i][j];
		}
	}
	return sum;
}

void min_max_normalization(pgm* image, int** matrix) {
	int min = 1000000, max = 0, i, j;
	
	for(i = 0; i < image->height; i++) {
		for(j = 0; j < image->width; j++) {
			if (matrix[i][j] < min) {
				min = matrix[i][j];
			}
			else if (matrix[i][j] > max) {
				max = matrix[i][j];
			}
		}
	}
	
	for(i = 0; i < image->height; i++) {
		for(j = 0; j < image->width; j++) {
			double ratio = (double) (matrix[i][j] - min) / (max - min);
			matrix[i][j] = ratio * 255;
		}
	} 
}

void write_pgm_file(pgm* image, char* dir, int** matrix) {
	FILE* out_image;
	int i, j, count = 0;
	
	out_image = fopen(dir, "wb");
	fprintf(out_image, "%s\n", image->version);
	fprintf(out_image, "%d %d\n", image->width, image->height);
	fprintf(out_image, "%d\n", image->maxGrayLevel);
	
	if (strcmp(image->version, "P2") == 0) {
		for(i = 0; i < image->height; i++) {
			for(j = 0; j < image->width; j++) {
				fprintf(out_image,"%d", matrix[i][j]);
				if (count % 17 == 0) 
					fprintf(out_image,"\n");
				else 
					fprintf(out_image," ");
				count ++;
			}
		} 
	}
	else if (strcmp(image->version, "P5") == 0) {
		for(i = 0; i < image->height; i++) {
			for(j = 0; j < image->width; j++) {
				char num = matrix[i][j];
				fprintf(out_image,"%c", num);
			}
		} 
	} 
	fclose(out_image);
}