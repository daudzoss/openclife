// build with: gcc -o 2048 -L/usr/lib/path_to_libOpenCL.so playgame.c readkern.c -lOpenCL

// renamed from convolution.c in ch. 4 of "Heterogeneous Computing with OpenCL"

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

//#include <termios.h>            //termios, TCSANOW, ECHO, ICANON
//#include <unistd.h>     //STDIN_FILENO


char* readSource(char*);
void chk(cl_int, const char*, cl_device_id*, cl_program*);

void printGrid(int* g, int M, int N) { // g already odd or even
  int i, j, k, l;

  l = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      int c = g[l*2];
      for (k = 31; k > 0; k--)
	if (c & (1<< k))
	  printf("X");
       else
	 printf("-");
    }
    printf("\n");
  }
  printf("\n");
}

inline void done(int* grid){ free(grid); exit(0); }

int main(int argc, char** argv) {
  int i/*, j, k, l*/;

   int up = 0, down = 0, left = 0, right = 0;

   // size of grid in x and y
   int xLog = 2, yLog = 2, xDim, yDim, errDim;
   if (argc > 2) {
     xLog = atoi(argv[2]);
     if (argc > 3) {
       yLog = atoi(argv[3]);
     }
   }
   xDim = 1<<xLog;
   yDim = 1<<yLog;
   errDim = (xDim > yDim) ? xDim : yDim;

   // Initialize the board position
   const size_t dataSize = xDim*yDim*sizeof(int);
   int* grid = (int*) calloc(xDim*yDim, sizeof(int));
   grid[1] = grid[xDim] = 2;
   printGrid(grid, xDim, yDim);
   termsetup(1);
             
   // Set up the OpenCL environment
   const cl_int SLIDE_UP = -xDim;
   const cl_int SLIDE_LF = -1;
   const cl_int SLIDE_RT = 1;
   const cl_int SLIDE_DN = xDim;
   const cl_int log_rep = 0;
   cl_int status;

   // Discover platform
   cl_platform_id platform;
   status = clGetPlatformIDs(1, &platform, NULL);
   chk(status, "clGetPlatformIDs", NULL, NULL);

   // Discover device
   cl_device_id device;
   cl_device_type pu = ((argc>1)?(('G'&*argv[1])=='G'):0) ? CL_DEVICE_TYPE_GPU
                                                          : CL_DEVICE_TYPE_CPU;
   printf("max workgroup size %d, requesting %cPU\n", (xDim>yDim)?xDim:yDim,
	  (pu == CL_DEVICE_TYPE_GPU) ? 'G' : 'C');
   status = clGetDeviceIDs(platform, pu, 1, &device, NULL);
   chk(status, "clGetDeviceIDs", NULL, NULL);

   cl_uint numdims;
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
			    sizeof(cl_uint), &numdims, NULL);
   chk(status, "clGetDeviceInfo", NULL, NULL);
   size_t dims[numdims];
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
			    numdims*sizeof(size_t), &dims, NULL);
   chk(status, "clGetDeviceInfo", NULL, NULL);
   printf("the max workgroup size of which is reported as %d\n", dims[0]);

   // Create context
   cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,
       (cl_context_properties)(platform), 0};
   cl_context context;
   context = clCreateContext(props, 1, &device, NULL, NULL, &status);
   chk(status, "clCreateContext", NULL, NULL);

   // Create command queue
   // FIXME: "warning: ''clCreateCommandQueue'' is deprecated"
   cl_command_queue queue;
   queue = clCreateCommandQueue(context, device, 0, &status);
   chk(status, "clCreateCommandQueue", NULL, NULL);

   // Create a program object with source and build it
   const char* source = readSource("game2048.cl");
   cl_program program;
   program = clCreateProgramWithSource(context, 1, &source, NULL, &status);
   chk(status, "clCreateProgramWithSource", NULL, NULL);
   status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
   chk(status, "clBuildProgram", &device, &program);

   // Create the kernel objects and arguments
   // FIXME: is a cl_int at these addresses changeable? redo clSetKernelArg()?
   cl_kernel tilt = clCreateKernel(program, "tilt", &status);
   chk(status, "clCreateKernel", NULL, NULL);

   // Create space for the grid on the device
   // FIXME: can we use CL_MEM_USE_HOST_PTR to avoid clEnqueueWriteBuffer()?
   // FIXME: inefficient for getting a return value?
   cl_mem d_grid, d_invalid;
   d_grid = clCreateBuffer(context, 0, dataSize, NULL, &status);
   chk(status, "clCreateBuffer", NULL, NULL);
   d_invalid = clCreateBuffer(context, 0, errDim*sizeof(cl_int),
                              NULL, &status);
   chk(status, "clCreateBuffer", NULL, NULL);

   do {
     cl_int nElements, slide_dir;
     cl_int invalid[errDim];

     switch (getchar()) {
     case 'h':case 'a':case '4': left = 1; slide_dir = SLIDE_LF; nElements = xDim; break;
     case 'j':case 's':case '2': down = 1; slide_dir = SLIDE_DN; nElements = yDim; break;
     case 'k':case 'w':case '8': up = 1; slide_dir = SLIDE_UP; nElements = yDim; break;
     case 'l':case 'd':case '6': right = 1; slide_dir = SLIDE_RT; nElements = xDim; break;
     case 'q':case '\033': done(grid);
     default : continue;  // applies to the do...while
     }
     putchar('\n');

     // set arguments each time?
     status  = clSetKernelArg(tilt, 0, sizeof(cl_mem), &d_grid);
     status |= clSetKernelArg(tilt, 1, sizeof(cl_int), &nElements);
     status |= clSetKernelArg(tilt, 2, sizeof(cl_int), &slide_dir);
     status |= clSetKernelArg(tilt, 3, sizeof(cl_int), &log_rep);
     status |= clSetKernelArg(tilt, 4, sizeof(cl_mem), &d_invalid);
     chk(status, "clSetKernelArg", NULL, NULL);

     // Copy inputs to the device
     status = clEnqueueWriteBuffer(queue, d_grid, CL_TRUE /*blocking_write*/,
				   0 /*offset*/, dataSize, grid,
				   0 /*events_in_ ...*/, NULL /*event_wait_list*/,
				   NULL /*event*/);
     chk(status, "clEnqueueWriteBuffer", NULL, NULL);

     // Set the work item dimensions
     size_t globalSize[2] = {xDim, yDim};
     status = clEnqueueNDRangeKernel(queue, tilt, 1, NULL,
				     globalSize + ishoriz(slide_dir), NULL,
				     0, NULL, NULL);
     chk(status, "clEnqueueNDRangeKernel", NULL, NULL);

     status = clEnqueueReadBuffer(queue, d_grid, CL_TRUE /*blocking_read*/,
				  0 /*offset*/, dataSize, grid,
				  0 /*events_in_ ...*/, NULL /*event_wait_list*/,
				  NULL /*event*/);
     chk(status, "clEnqueueReadBuffer", NULL, NULL);
     status = clEnqueueReadBuffer(queue, d_invalid, CL_TRUE /*blocking_read*/,
				  0 /*offset*/, nElements*sizeof(cl_int), invalid,
				  0 /*events_in_ ...*/, NULL /*event_wait_list*/,
				  NULL /*event*/);
     chk(status, "clEnqueueReadBuffer", NULL, NULL);

     for (i = (nElements == xDim) ? yDim-1 : xDim-1; i >= 0; i--)
{
  printf("%d ", invalid[nElements]);
       if (!invalid[i])
	 break; // found a valid move, so i will be >= 0
}
printf("\n");
     if (i >= 0) {
       dropGrid(grid, xDim*yDim, 2<<(1&random()), random()&((1<<(xLog+yLog))-1));
       printGrid(grid, xDim, yDim);
       up = down = left = right = 0;
     }
   } while (!(up && down && left && right));
}
