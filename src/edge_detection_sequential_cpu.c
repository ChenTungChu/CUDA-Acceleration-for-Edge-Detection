/**
 * @file edge_detection_sequential_cpu.c
 * @author jz544 & cc2396
 * @brief 
 * @version 1.0
 * @date 2022-04-30
 * 
 * compile: gcc -o edge_detection_sequential_cpu edge_detection_sequential_cpu.c -lm
 * run: ./edge_detection_sequential_cpu 100  # 100 is the number of images it processes
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

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
void sobel_edge_filter(pgm* image, pgm* out_image);   // sobel edge filter 
void min_max_normalization(pgm* image, int** matrix); // min max normalization
int convolution(pgm* image, int kernel[3][3], int row, int col);
void write_pgm_file(pgm* image, char* dir, int** matrix);

int main(int argc, char *argv[]) {
	pgm image, out_image;   // define the input image and output image
	int num_of_imgs;

	if (argc == 1) {
		printf("[INFO] no arguments, use default number of images\n");
		printf("[INFO] number of images: %d\n", NUM_IMGS);
		num_of_imgs = NUM_IMGS;
	}
	else {
		printf("[INFO] get num of images from command line\n");
		printf("[INFO] number of images: %s\n", argv[1]);
		num_of_imgs = atoi(argv[1]);
	}
	printf("[INFO] test image from %s\n", IMG_PATH);
	
	// printf("Enter the file name: ");
	// scanf("%s", dir);
	// printf("%s\n", dir);
	
	read_image(IMG_PATH, &image);
	padding(&image);
	init_pgm_image(&out_image, image);
	struct timeval start, end;   // start and stop timer
    float el_time;               // elapsed time
    gettimeofday(&start, NULL);  // start counting time
	// kernel
	for (int i = 0; i < num_of_imgs; i++) {
		sobel_edge_filter(&image, &out_image);
	}
	// kernel
	gettimeofday(&end, NULL);   // stop counting time
    el_time = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
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
		image->imageData[i] = calloc(image->width, sizeof(int));

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
		out->imageData[i] = calloc(out->width, sizeof(int));
	}
	
	out->gx = (int**) calloc(out->height, sizeof(int*));
	for(i = 0; i < out->height; i++) {
		out->gx[i] = calloc(out->width, sizeof(int));
	}
	
	out->gy = (int**) calloc(out->height, sizeof(int*));
	for(i = 0; i < out->height; i++) {
		out->gy[i] = calloc(out->width, sizeof(int));
	}
	
	for(i = 0; i < out->height; i++) {
		for(j = 0; j < out->width; j++) {
			out->imageData[i][j] = image.imageData[i][j];
			out->gx[i][j] = image.imageData[i][j];
			out->gy[i][j] = image.imageData[i][j];
		};
	}
}

void sobel_edge_filter(pgm* image, pgm* out_image) {
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
	
	for (i = 1; i < image->height - 2; i++) {
		for (j = 1; j < image->width - 2; j++) {
			gx = convolution(image, mx, i, j);
			gy = convolution(image, my, i, j);
			out_image->imageData[i][j] = sqrt(gx*gx + gy*gy);
			out_image->gx[i][j] = gx;
			out_image->gy[i][j] = gy;
		}
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
