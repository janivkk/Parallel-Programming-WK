#include <iostream>
#include <vector>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//	Create an event and attach it to a queue command responsible for the kernel lunch
		cl::Event prof_event;

		cl::Event A_event;

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input
		std::vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
		std::vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

		//host - input ... but float or double for precision
		//std::vector<float> A = { 0, 1.1, 2.3, 3.5, 4.1, 5.6, 6.7, 7.7, 8.1, 9.8 };
		//std::vector<float> B = { 0, 1.2, 2.5, 0, 1.5, 2.1, 0.4, 1.1, 2.3, 0.7 };
	
		//	Accommodate larger input arrays/vectors
		//std::vector<int> A(10);
		//std::vector<int> B(10);

		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size() * sizeof(int);//size in bytes

		//size_t vector_elements = A.size(); // number of elements
		//size_t vector_size = A.size() * sizeof(float); //	size in bytes

		//host - output
		std::vector<int> C(vector_elements);

		//std::vector<float> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 4 - device operations

		//4.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		////	Parallel addition simply overrides the values storedin vector C
		//kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(0, buffer_C);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		//	Kernel function arguments, just like ^
		cl::Kernel kernel_mult = cl::Kernel(program, "mult");
		kernel_mult.setArg(0, buffer_A);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C);

		// Complex function of C = A * B + B
		/*cl::Kernel kernel_multadd = cl::Kernel(program, "multadd");
		kernel_multadd.setArg(0, buffer_A);
		kernel_multadd.setArg(1, buffer_B);
		kernel_multadd.setArg(2, buffer_C);*/

		//	Kernel function that adds float in parallel
		/*Kernel execution time [ns]: 72704
		Queued 2, Submitted 11, Executed 72, Total 87 [us]*/
		//cl::Kernel kernel_addf = cl::Kernel(program, "addf");
		//kernel_addf.setArg(0, buffer_A);
		//kernel_addf.setArg(1, buffer_B);
		//kernel_addf.setArg(2, buffer_C);

		//	Kernel launches into the queue in the right order
		queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//	Added NULL, &prof_event as per task requirement
		//	Total number of kernel launches is equal to the vector length, which is specified as a parameter for the ... cl::NDRange(vector_elements)

		//	Performance: Kernel execution time [ns]: 4096
		//	Queued 2, Submitted 7, Executed 4, Total 13[us] for default vectors
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//	Performance: Kernel execution time [ns]: 27456
		//	Queued 2, Submitted 11, Executed 27, Total 40[us] for default vectors

		//queue.enqueueNDRangeKernel(kernel_multadd, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//	Performs the longest.
		//queue.enqueueNDRangeKernel(kernel_addf, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		//	Profiling a copy operation on Vector A
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);

		//	Prints out the vector elements
		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;

		//	Display the kernel execution time at the end of the program
		std::cout << "Kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		//	Detailed breakdown of our event in ms, full info about each profiling event including enqueueing and prep time
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}