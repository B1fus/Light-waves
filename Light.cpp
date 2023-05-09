#include "Light.cuh"

__device__ __host__ Light::Light(int x, int y) {
	pixSize = { x,y };
	x /= segment;
	y /= segment;
	size = { x, y };
	//mat0 = new Point2<float>[x * y * 3];
	//mat1 = new Point2<float>[x * y * 3];
	//weights = new float[x * y * 3];
	cudaHostAlloc(&mat0, x * y * 3 * sizeof(Point2<float>), 0);
	cudaHostAlloc(&mat1, x * y * 3 * sizeof(Point2<float>), 0);
	cudaHostAlloc(&weights, x * y * 3 * sizeof(float), 0);
	std::fill(mat0, mat0 + x * y * 3, Point2<float>{0,0});
	std::fill(mat1, mat1 + x * y * 3, Point2<float>{0,0});
	std::fill(weights, weights + x * y * 3, 1);

	float setVal = 1;
	float border = 100;
	for (int xi = 0; xi < x; xi++) {
		for (int yi = 0; yi < y; yi++) {
			/*if (x<border || y<border || x>size.x - border || y>size.y - border) {
				if (x < border)
					setVal = x / border;
				if (y < border)
					setVal = y / border;
				if (x > size.x - border)
					setVal = (size.x - x) / border;
				if (y > size.y - border)
					setVal = (size.y - y) / border;
				weights[get_offset(x, y) / 3] = setVal;
			}*/

			//if (xi > 650 / segment && xi<750/segment && yi>100/segment && yi<450/segment) {
			//	if (Vec2<float>((xi - 600 / segment)/1.0, (yi - 280 / segment)/0.7).length() > 120 / segment) {
			//		//if (xi > 700/segment) {
			//		weights[get_offset(xi, yi) + 0] = 1 / 1.42;
			//		weights[get_offset(xi, yi) + 1] = 1 / 1.38;
			//		weights[get_offset(xi, yi) + 2] = 1 / 1.34;
			//		//}
			//	}
			//}

			if (Vec2<float>((xi - 600 / segment) / 1.5, (yi - 280 / segment) / 0.8).length() < 120 / segment) {
				if (xi > 650 / segment) {
					weights[get_offset(xi, yi) + 0] = 1 / 1.42;
					weights[get_offset(xi, yi) + 1] = 1 / 1.38;
					weights[get_offset(xi, yi) + 2] = 1 / 1.34;
				}
			}
		}
	}

	cudaMalloc(&gpuMat0, x * y * 3 * sizeof(Point2<float>));
	cudaMalloc(&gpuMat1, x * y * 3 * sizeof(Point2<float>));
	cudaMalloc(&gpuWeights, x * y * 3 * sizeof(float));
	cudaMalloc(&gpuImg, pixSize.x * pixSize.y * sizeof(int));
	cudaMalloc(&gpuImgAccum, x * y * 3 * sizeof(float));
}

__device__ __host__ __global__ Light::~Light() {
	delete[] mat0, mat1;
	cudaFree(&gpuMat0);
	cudaFree(&gpuMat1);
}

__device__ __host__ inline int Light::get_offset(int x, int y)
{
	if (x < 0 || x >= size.x || y < 0 || y >= size.y)
		return -1;
	return y * size.x * 3 + x * 3;
}

void Light::move() {
	Point2<float>* setMat, * fromMat;
	if (editingMat == 0)
		setMat = mat0, fromMat = mat1;
	else
		setMat = mat1, fromMat = mat0;

	float newHeg = 0;
	int offset = 0;
	float border = size.x / 10;
	float setVal = 1;
	for (int x = 0; x < size.x; x++) {
		for (int y = 0; y < size.y; y++) {
			for (int k = 0; k < 3; k++) {
				setVal = 1;
				offset = get_offset(x - 1, y);
				newHeg = offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
				offset = get_offset(x, y + 1);
				newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
				offset = get_offset(x + 1, y);
				newHeg += offset >= 0 ? fromMat[offset + k].x * weights[offset + k] : 0;
				offset = get_offset(x, y - 1);
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
				setMat[get_offset(x, y) + k].y = (fromMat[get_offset(x, y) + k].y + (newHeg - fromMat[get_offset(x, y) + k].x * weights[get_offset(x, y) + k]));
				setMat[get_offset(x, y) + k].x = 0.999 * setVal * (fromMat[get_offset(x, y) + k].x + setMat[get_offset(x, y) + k].y);
			}
		}
	}

	editingMat = !editingMat;
}

void Light::get_mat_height(int* mat)
{
	int heg, hegOut;
	Point2<float>* fromMat;
	if (editingMat == 1)
		fromMat = mat0;
	else
		fromMat = mat1;

	for (int x = 0; x < size.x; x++) {
		for (int y = 0; y < size.y; y++) {
			hegOut = 0;

			//heg = (int)fromMat[get_offset(x, y) + 0].x + 127 * weights[get_offset(x,y) + 0] ;
			heg = (int)fromMat[get_offset(x, y) + 0].x;
			if (heg > 255) heg = 255;
			if (heg < 0) heg = 0;
			hegOut = hegOut | (heg << 16);

			//heg = (int)fromMat[get_offset(x, y) + 1].x + 127 * weights[get_offset(x,y) + 1] ;
			heg = (int)fromMat[get_offset(x, y) + 1].x;
			if (heg > 255) heg = 255;
			if (heg < 0) heg = 0;
			hegOut = hegOut | (heg << 8);

			//heg = (int)fromMat[get_offset(x, y) + 2].x + 127 * weights[get_offset(x,y) + 2] ;
			heg = (int)fromMat[get_offset(x, y) + 2].x;
			if (heg > 255) heg = 255;
			if (heg < 0) heg = 0;
			hegOut = hegOut | (heg << 0);

			for (int sx = 0; sx < segment; sx++) {
				for (int sy = 0; sy < segment; sy++) {
					int offset = 0;
					mat[y * size.x * segment * segment + sy * size.x * segment + x * segment + sx] = hegOut;
				}
			}

		}
	}
}

void Light::set_pixel_mat_height(int x, int y, int value)
{
	int heg, hegOut;
	Point2<float>* fromMat;
	if (editingMat == 1)
		fromMat = mat0;
	else
		fromMat = mat1;
	static float sina = 1;
	sina += DELTA_TIME;
	float r, g, b;
	x /= segment;
	y /= segment;
	r = (value & (255 << 16)) >> 16;
	g = (value & (255 << 8)) >> 8;
	b = (value & (255 << 0)) >> 0;
	
	float cosa = 0;
	int radius = 100;
	float period = 5;
	for (int c = 0; c < 1; c++) {
		for (int i = -radius; i < radius; i++) {
			for (int j = -radius; j < radius; j++) {
				if ((i) * (i)+(j) * (j) < radius * radius && ((sina - int(sina)) < DELTA_TIME * 2)) {
					//cosa = cos(j * 3.1415 / 2) * (cos(i*3.1415/radius/2));
					//cosa = cos(j * 3.1415 / 2) * (cos(sqrtf(i*i+j*j)*3.1415/4));
					cosa = 1 * cos(j * 3.1415 / period) * (cos((sqrtf(i * i + j * j)) * 3.1415 / 2 / radius));
					fromMat[get_offset(x + j - c*radius*2, y + i) + 0] += { r * cosa,0 };
					fromMat[get_offset(x + j - c*radius*2, y + i) + 1] += { g * cosa,0 };
					fromMat[get_offset(x + j - c*radius*2, y + i) + 2] += { b * cosa,0 };

				}
			}
		}
	}

	/*fromMat[get_offset(x, y) + 0] = { r * sinf(sin),0 };
	fromMat[get_offset(x, y) + 1] = { g * sinf(sin),0 };
	fromMat[get_offset(x, y) + 2] = { b * sinf(sin),0 };*/
}

void Light::set_pixel_weight(int x, int y, float value)
{
	x /= segment;
	y /= segment;
	
	/*for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			fromMat[get_offset(x + j, y + i) + 0] = { r,0 };
			fromMat[get_offset(x + j, y + i) + 1] = { g,0 };
			fromMat[get_offset(x + j, y + i) + 2] = { b,0 };
		}
	}*/

	weights[get_offset(x, y) + 0] = value;
	weights[get_offset(x, y) + 1] = value;
	weights[get_offset(x, y) + 2] = value;
}

void Light::set_weights_from_array(int* img)
{
	int heg, hegOut;
	int r, g, b;

	for (int x = 0; x < size.x; x++) {
		for (int y = 0; y < size.y; y++) {
			r = 0, g = 0, b = 0;

			//heg = (int)fromMat[get_offset(x, y) + 0].x + 127 * weights[get_offset(x,y) + 0] ;

			for (int sx = 0; sx < segment; sx++) {
				for (int sy = 0; sy < segment; sy++) {
					heg = img[get_offset(x + sx, y + sy) / 3] & (0xff << 16);
					r += heg;

					heg = img[get_offset(x + sx, y + sy) / 3] & (0xff << 8);
					g += heg;

					heg = img[get_offset(x + sx, y + sy) / 3] & (0xff << 0);
					b += heg;
				}
			}
			weights[get_offset(x, y) + 0] = r / segment / segment;

			weights[get_offset(x, y) + 1] = g / segment / segment;

			weights[get_offset(x, y) + 2] = b / segment / segment;

		}
	}
}

