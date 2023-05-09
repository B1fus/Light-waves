#pragma once
#include "framework.h"

#define GPU_THREADS 128

class Light
{
public:
	__device__ __host__ Light(int x, int y);
	void move();
	void get_mat_height(int* mat);
	void set_pixel_mat_height(int x, int y, int value);
	void set_pixel_weight(int x, int y, float value);
	void set_weights_from_array(int* img);
	~Light();
	bool editingMat = 0;
	int segment = 1;
	Point2<int> size;
	Point2<int> pixSize;
	Point2<float>* mat0, * mat1; //x - height, y - velocity
	float* weights;

	float* gpuWeights;
	Point2<float>* gpuMat0, * gpuMat1;
	int* gpuImg;
	float* gpuImgAccum;
	__device__ __host__ inline int get_offset(int x, int y);
private:

};

void move_gpu(Light& l, int* img);
//__device__ void move_pixel_gpu(Point2<float>* setMat, Point2<float>* fromMat);
//void move_pixel_gpu();
