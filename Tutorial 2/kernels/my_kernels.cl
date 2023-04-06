/*Kernel for Tutorial 02*/

/*Many Bins Histogram -> https://stackoverflow.com/questions/27947178/opencl-histogram-with-many-bins */

#define RANGE_SIZE 8192

//kernel void hist_test(global uint a, constant int a_size) {
//	int wid = get_local_id(0);
//	int w_size = get_local_size(0);
//
//	int gid = get_group_id(0);
//	int n_groups = get_num_groups(0);
//
//	int range_begin = gid * RANGE_SIZE / n_groups;
//	int range_end = (gid + 1) * RANGE_SIZE / n_groups;
//
//	local uint tmp_hist[RANGE_SIZE]; uint value;
//
//	for (int i = wid; i < a_size; += w_size) {
//		value = a[i];
//
//		if (value >= range_begin && value < range_end) {
//			atomic_inc(tmp_hist[value - range_begin]);
//		}
//	}
//
//	barrier(CLK_GLOBAL_MEM_FENCE);
//}

kernel void hist_atomic(global const int* A, local int* H, int nr_bins) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int bin_index = A[id];

	//	Clearing the scratch bins..
	if (lid < nr_bins) { H[lid] = 0; }

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);
}

kernel void hist_local_simple(global const uchar* A, global int* H, int nr_bins) {
	int id = get_global_id(0); int lid = get_local_id(0);

	int bin_index = A[id];

	//	Clearing the scratch bins..
	if (lid < nr_bins) { H[lid] = 0; }

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);
	//barrier(CLK_LOCAL_MEM_FENCE);

	//if (id < nr_bins) { atomic_add(&H[lid], LH[lid]); }
}

//a very simple histogram implementation
kernel void hist_simple(global const uchar* A, global int* H) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int lid = get_local_id(0);

	int	bin_index = A[id];//take value as a bin index

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying

	atomic_inc(&H[bin_index]);	//serial operation, not very efficient!

	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void hist_serial(global const int* A, global int* H) {
	int id = get_global_id(0); int N = get_global_size(0);

	int bin_index = A[id];

	for (int i = 0; i < N; i++) {
		if (id == 1) { H[bin_index] += 1; }
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void hist_cumulative(global int* H, global int* CH) {
	int id = get_global_id(0);

	int n = get_global_size(0);

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = id + 1; i < n; i++) {
		atomic_add(&CH[i], H[id]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

/* LUT look-up table */
kernel void LUT_table(global int* cumulative_hist, global int* LUT) {
	int id = get_global_id(0);

	barrier(CLK_GLOBAL_MEM_FENCE);

	LUT[id] = cumulative_hist[id] * (double)255 / cumulative_hist[255];
}

/* Copying all pixels from A to B */
kernel void LUT_redirective(global uchar* A, global int* LUT, global uchar* B) {
	int id = get_global_id(0);

	barrier(CLK_GLOBAL_MEM_FENCE);

	B[id] = LUT[A[id]];
}

/* ?? */
