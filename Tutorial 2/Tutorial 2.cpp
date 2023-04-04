#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

/* Use when running this code on the personal machine. */
//#include <include/CL/cl.h>

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	
	/* Default Images */
	//string image_filename = "test.ppm";
	
	/* Assignment Images */
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		typedef int custom_int; std::vector<custom_int> H_bin(256);
		std::vector<custom_int> CH_bin(256);
		size_t h_size = H_bin.size() * sizeof(custom_int);

		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
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

		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		/* Histogram Buffers */
		cl::Buffer dev_hist_simple_output(context, CL_MEM_READ_WRITE, h_size);
		cl::Buffer dev_hist_cumulative_output(context, CL_MEM_READ_WRITE, h_size);
		cl::Buffer dev_lut_output(context, CL_MEM_READ_WRITE, h_size);

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);
		queue.enqueueFillBuffer(dev_hist_simple_output, 0, 0, h_size);

		//4.2 Setup and execute the kernel (i.e. device code)
		//cl::Kernel kernel = cl::Kernel(program, "identity");
		//kernel.setArg(0, dev_image_input);
		//kernel.setArg(1, dev_image_output);
//		kernel.setArg(2, dev_convolution_mask);

		/* This line uses Intensity Histogram to describe the distribution of the frequency of each pixel from 0 to 255. */
		cl::Kernel kernel_hist_simple = cl::Kernel(program, "hist_simple");
		kernel_hist_simple.setArg(0, dev_image_input);
		kernel_hist_simple.setArg(1, dev_hist_simple_output);

		/* Cumulative Histogram */
		cl::Kernel kernel_cumulative = cl::Kernel(program, "hist_cumulative");
		kernel_cumulative.setArg(0, dev_image_input);
		kernel_cumulative.setArg(1, dev_hist_cumulative_output);

		//cl::Kernel kernel_? = cl::Kernel(program, "/");

		cl::Event prof_event_simple;

		cl::Event prof_event_cumulative;

		queue.enqueueNDRangeKernel(kernel_hist_simple, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event_simple);
		queue.enqueueReadBuffer(dev_hist_simple_output, CL_TRUE, 0, h_size, &H_bin[0]);

		queue.enqueueFillBuffer(dev_hist_cumulative_output, 0, 0, h_size);

		queue.enqueueNDRangeKernel(kernel_cumulative, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event_cumulative);
		queue.enqueueReadBuffer(dev_hist_cumulative_output, CL_TRUE, 0, h_size, &H_bin[0]);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
