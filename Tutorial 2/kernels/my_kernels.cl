/*Kernel for Tutorial 02*/

/*Many Bins Histogram -> https://stackoverflow.com/questions/27947178/opencl-histogram-with-many-bins */

#define RANGE_SIZE 8192

kernel void hist_test(global uint a, constant int a_size) {
	int wid = get_local_id(0);
	int w_size = get_local_size(0);

	int gid = get_group_id(0);
	int n_groups = get_num_groups(0);

	int range_begin = gid * RANGE_SIZE / n_groups;
	int range_end = (gid + 1) * RANGE_SIZE / n_groups;

	local uint tmp_hist[RANGE_SIZE]; uint value;

	for (int i = wid; i < a_size; += w_size) {
		value = a[i];

		if (value >= range_begin && value < range_end) {
			atomic_inc(tmp_hist[value - range_begin]);
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void hist_atomic(global const int* A, local int* H, int nr_bins) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int bin_index = A[id];

	//	Clearing the scratch bins..
	if (lid < nr_bins) { H[lid] = 0; }

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);
}

kernel void hist_local_simple(global const int* A, global int* H, local int* LH, int nr_bins) {
	int id = get_global_id(0); int lid = get_local_id(0);

	int bin_index = A[id];

	//	Clearing the scratch bins..
	if (lid < nr_bins) { H[lid] = 0; }

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&LH[bin_index]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if (id < nr_bins) { atomic_add(&H[lid], LH[lid]); }
}

//a very simple histogram implementation
kernel void hist_simple(global const uchar* A, global int* H) {
	int id = get_global_id(0); int lid = get_local_id(0);

	//assumes that H has been initialised to 0
	int		bin_index = A[id];//take value as a bin index

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying

	atomic_inc(&H[bin_index]);	//serial operation, not very efficient!

	//barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying

	//	Combine all local hist into a global one.
	//	if (id < nr_bins) atomic_add(&H[id], LH[id]); 
}

kernel void hist_cumulative(global int* H, global int* CH) {
	int id = get_global_id(0);

	int n = get_global_size(0);

	//barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = id + 1; i < n; i++) {
		atomic_add(&CH[i], H[id]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}


/* ?? */

//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	if (colour_channel == 0) {
		B[id] = A[id];
	}
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}