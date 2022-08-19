#include <stdint.h>
#include "2dconvolution.c"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CHANNEL_NUM 1
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define THRESHOLD(x, th) (((x) > (th)) ? (255) : (0))

unsigned char *edges_sobel(unsigned char *im, int w, int h, float threshold, int padding_method);
unsigned char convolution(unsigned char* image, unsigned char kernel[3][3], int row, int col);

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("./lena_gray.png", &width, &height, &channels, 0);
    printf("w:%d h:%d channels:%d\n", width, height, channels);

    unsigned char *result = edges_sobel(img, width, height, 70, 0);
	// unsigned char *result = malloc(width * height * sizeof(unsigned char));
	// for (int i = 0; i < width * height; i++) {
	// 	if (i < 200 * 200) {
	// 		result[i] = 255;
	// 	} else {
	// 		result[i] = 0;
	// 	}
	// }

    stbi_write_png("after_sobel.png", width, height, CHANNEL_NUM, result, width*CHANNEL_NUM);

    return 0;
}

// Sobel edge detector
unsigned char *edges_sobel(unsigned char *im, int w, int h, float threshold, int padding_method) {

	/* Define operators */
	// double sobel_1[9] = {-1,-2,-1, 0, 0, 0, 1, 2, 1};		// SOBEL
	// double sobel_2[9] = {-1, 0, 1,-2, 0, 2,-1, 0, 1};		// OPERATORS
	int i, j, gx, gy;
	int mx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	int my[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};
	// for ( int z=0; z<9; z++ ) {								// NORMALIZATION
	// 	sobel_1[z] /= 8;
	// 	sobel_2[z] /= 8;
	// }
	
	/* Convolution with operators */
	// double *im_s1 = conv2d(im, w, h, sobel_1, 3, padding_method);
	// double *im_s2 = conv2d(im, w, h, sobel_2, 3, padding_method);
	for (i = 1; i < h - 2; i++) {
		for (j = 1; j < w - 2; j++) {
			gx = convolution(image, mx, i, j);
			gy = convolution(image, my, i, j);
			out_image->imageData[i][j] = sqrt(gx*gx + gy*gy);
			out_image->gx[i][j] = gx;
			out_image->gy[i][j] = gy;
		}
	}

	// /* Allocate memory for output image */
	// float *im_sobel = malloc(w*h*sizeof(float));
	// if (im_sobel == NULL){
	// 	fprintf(stderr, "Out of memory...\n");
	// 	exit(EXIT_FAILURE);
	// }
	
	// /* Two images are obtained (one for each operator). Then the gradient magnitude 
	// image is constructed using $M=\sqrt{g_x^2+g_y^2}$. Also the absolute maximum 
	// value of the constructed images is computed */
	// int i,j, fila, col;
	// double max_s = 0;
	// int imax = w*h;
	// for ( i=0; i<imax; i++ ) {
	// 	fila = (int)(i/w);
	// 	col = i - w*fila + 1;
	// 	fila += 1;
	// 	j = col + (w+2)*fila;
	// 	im_sobel[i] = sqrt(im_s1[j]*im_s1[j] + im_s2[j]*im_s2[j]);
	// 	max_s = MAX(max_s,im_sobel[i]);
	// }
	
	// /* Thresholding */
	// for ( i=0; i<imax; i++ ) {
	// 	im_sobel[i] = THRESHOLD(im_sobel[i],threshold*max_s);
	// }
	
	// /* Free memory */
	// free(im_s1);
	// free(im_s2);
	
	// /* Output image */
	// return im_sobel;
}

unsigned char convolution(unsigned char* image, unsigned char kernel[3][3], int row, int col) {
	int i, j, sum = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			sum += image->imageData[i + row][j + col] * kernel[i][j];
		}
	}
	return sum;
}