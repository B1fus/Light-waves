#include "framework.h"

//__device__ void Light::move_pixel_gpu(Point2<int> size, Point2<float>* setMat, Point2<float>* fromMat, float* weights) {
//__global__ void move_pixel_gpu(Point2<float>* setMat, Point2<float>* fromMat) {
//	return;
//	//float newHeg = 0;
//	//int offset = 0;
//	//float border = size.x / 10;
//	//float setVal = 1;
//	//for (int x = 0; x < size.x; x++) {
//	//	for (int y = 0; y < size.y; y++) {
//	//		for (int k = 0; k < 3; k++) {
//	//			setVal = 1;
//	//			offset = get_offset(x - 1, y);
//	//			newHeg = offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
//	//			offset = get_offset(x, y + 1);
//	//			newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
//	//			offset = get_offset(x + 1, y);
//	//			newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
//	//			offset = get_offset(x, y - 1);
//	//			newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
//	//			newHeg /= 4;
//	//			if (x<border || y<border || x>size.x - border || y>size.y - border) {
//	//				if (x < border)
//	//					setVal = x / border;
//	//				if (y < border)
//	//					setVal = y / border;
//	//				if (x > size.x - border)
//	//					setVal = (size.x - x) / border;
//	//				if (y > size.y - border)
//	//					setVal = (size.y - y) / border;
//	//				//weights[get_offset(x, y) + k] = setVal;
//	//				setVal = -(pow((setVal - 1), 2)) + 1;
//	//			}
//	//			setMat[get_offset(x, y) + k].y = (fromMat[get_offset(x, y) + k].y + (newHeg - fromMat[get_offset(x, y) + k].x * weights[get_offset(x, y) + k]));
//	//			setMat[get_offset(x, y) + k].x = 0.999 * setVal * (fromMat[get_offset(x, y) + k].x + setMat[get_offset(x, y) + k].y);
//	//		}
//	//	}
//	//}
//
//}

__device__ int get_offset(int x, int y, Point2<int> size) {
	if (x < 0 || x >= size.x || y < 0 || y >= size.y)
		return -1;
	return y * size.x * 3 + x * 3;
}

__global__ void move_pixel_gpu(Point2<int> size, Point2<float>* fromMat, Point2<float>* setMat, float* weights, int* img, int segment, float* imgAccum) {

	float newHeg = 0;
	int offset = 0;
	float border = size.x / 10;
	float setVal = 1;/*
	for (int x = 0; x < size.x; x++) {
		for (int y = 0; y < size.y; y++) {*/
	int x, y;
	//x = 0; y = 0;
	x = (blockIdx.x * GPU_THREADS + threadIdx.x) % size.x;
	y = (blockIdx.x * GPU_THREADS + threadIdx.x) / size.x;
	if (x >= size.x || y >= size.y) return;
	/*for (int i = 0; i < 3; i++) {
		setMat[get_offset(x,y,size) + i].x = 255;
	}*/
	//setMat[get_offset(x, y, size) + 0].x = threadIdx.x/2;
	//setMat[get_offset(x, y, size) + 1].x = blockIdx.x*20;
	//setMat[get_offset(x, y, size) + 2].x = get_offset(x, y, size) == -1 ? 255: 0;

	static int iddd = 0;
	iddd += 1;

	for (int k = 0; k < 3; k++) {
		setVal = 1;
		offset = get_offset(x - 1, y, size);
		newHeg = offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
		offset = get_offset(x, y + 1, size);
		newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
		offset = get_offset(x + 1, y, size);
		newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
		offset = get_offset(x, y - 1, size);
		newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
		newHeg /= 4;
		if (x<border || y<border || x>size.x - border || y>size.y - border) {
			if (x < border)
				setVal = x / border;
			if (y < border)
				setVal = y / border;
			if (x > size.x - border)
				setVal = (size.x - x) / border;
			if (y > size.y - border)
				setVal = (size.y - y) / border;
			//weights[get_offset(x, y) + k] = setVal;
			setVal = -(pow((setVal - 1), 2)) + 1;
		}
		setMat[get_offset(x, y, size) + k].y = (fromMat[get_offset(x, y, size) + k].y + (newHeg - fromMat[get_offset(x, y, size) + k].x * weights[get_offset(x, y, size) + k]));
		setMat[get_offset(x, y, size) + k].x = 0.999 * setVal * (fromMat[get_offset(x, y, size) + k].x + setMat[get_offset(x, y, size) + k].y);
		imgAccum[get_offset(x, y, size) + k] *= 0.995;
		imgAccum[get_offset(x, y, size) + k] += setMat[get_offset(x, y, size) + k].x>0? setMat[get_offset(x, y, size) + k].x : 0;
	}


	//create output image

	int heg = 0, hegOut = 0;
	//heg = (int)fromMat[get_offset(x, y) + 0].x + 127 * weights[get_offset(x,y) + 0] ;
	//heg = (int)setMat[get_offset(x, y, size) + 0].x;
	heg = (int)imgAccum[get_offset(x, y, size) + 0] - 64 * (weights[get_offset(x,y,size)+0]-1);
	if (heg > 255) heg = 255;
	if (heg < 0) heg = 0;
	hegOut = hegOut | (heg << 16);

	//heg = (int)fromMat[get_offset(x, y) + 1].x + 127 * weights[get_offset(x,y) + 1] ;
	//heg = (int)setMat[get_offset(x, y, size) + 1].x;
	heg = (int)imgAccum[get_offset(x, y, size) + 1] - 64 * (weights[get_offset(x,y,size)+1]-1);
	if (heg > 255) heg = 255;
	if (heg < 0) heg = 0;
	hegOut = hegOut | (heg << 8);

	//heg = (int)fromMat[get_offset(x, y) + 2].x + 127 * weights[get_offset(x,y) + 2] ;
	//heg = (int)setMat[get_offset(x, y, size) + 2].x;
	heg = (int)imgAccum[get_offset(x, y, size) + 2] - 64 * (weights[get_offset(x,y,size)+2]-1);
	if (heg > 255) heg = 255;
	if (heg < 0) heg = 0;
	hegOut = hegOut | (heg << 0);

	for (int sx = 0; sx < segment; sx++) {
		for (int sy = 0; sy < segment; sy++) {
			int offset = 0;
			img[y * size.x * segment * segment + sy * size.x * segment + x * segment + sx] = hegOut;
		}
	}

}

__host__ void move_gpu(Light& l, int* img) {
	Point2<float>* setMat, * fromMat;
	if (l.editingMat == 0)
		setMat = l.mat0, fromMat = l.mat1;
	else
		setMat = l.mat1, fromMat = l.mat0;

	cudaMemcpy(l.gpuMat0, fromMat, l.size.x * l.size.y * 3 * sizeof(Point2<float>), cudaMemcpyHostToDevice);
	cudaMemcpy(l.gpuWeights, l.weights, l.size.x * l.size.y * 3 * sizeof(float), cudaMemcpyHostToDevice);

	//move_pixel_gpu << < 1, 1 >> > (l.gpuMat0, l.gpuMat1);
	move_pixel_gpu << < (l.size.x * l.size.y / GPU_THREADS) + 1, GPU_THREADS  >> > (l.size, l.gpuMat0, l.gpuMat1, l.gpuWeights, l.gpuImg, l.segment, l.gpuImgAccum);

	cudaMemcpy(setMat, l.gpuMat1, l.size.x * l.size.y * 3 * sizeof(Point2<float>), cudaMemcpyDeviceToHost);
	cudaMemcpy(img, l.gpuImg, l.pixSize.x * l.pixSize.y * sizeof(int), cudaMemcpyDeviceToHost);

	l.editingMat = !l.editingMat;
	return;
}
