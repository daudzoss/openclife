// build cmd: gcc -o life -L/usr/lib/path_to_libcl.so playgame.c readkern.c -lcl

// renamed from convolution.c in ch. 4 of "Heterogeneous Computing with OpenCL"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

char* readSource(char*);
void chk(cl_int, const char*, cl_device_id*, cl_program*);

void printGrid(int* g, int M, int N) { // g already odd or even, with gaps of 2!
  int l, vstrips;

  l = 0;
  vstrips = (N+29)/30;

  for (int i = 0; i < M; i++) {
    int n = N;

    for (int j = 0; j < vstrips; j++) {
      int c = g[2 * (l + j*M)];

      for (int k = 30; k && n; k--, n--)
	if (c & (1<<k))
	  printf("X");
	else
	  printf("-");
    }
    l++;
    printf("\n");
  }
  printf("\n");
}

int strtob(char* *ptr, int bits, char const* c0) {
  int retval = 0;

  for (int bit = 1 << (bits - 1); bit; bit >>= 1) {
    char c = toupper(*((*ptr)++));
    if (c) {
      char const* c1;
      
      for (c1 = c0; *c1; c1++)
	if (c == toupper(*c1))
	  break;
      if (!*c1) // c isn't one of the characters in c0, treat as a 1
	retval |= bit;
    } else
      break; // unexpected end of string before reaching "bits" bits
  }

  return retval; 
}

int main(int argc, char** argv) { // filename [<iter> [<skip> [cpu|gpu]]]
   // size of grid in x and y
  cl_int east_wood, M, N;
  char line[1024];
  int item = 0;
  const int max = 2*(2*1024*1024); // 2*2MiW (double buffer)
  int* grid = (int*) malloc(max*sizeof(int));

  // Parse the command line
  if (argc<2) {printf("%s file|- [iterations [skip [c|gpu]]]\n",*argv);exit(1);}
  FILE* in = strcmp(argv[1],"-") ? fopen(argv[1], "r") : NULL;
  cl_device_type pu = CL_DEVICE_TYPE_GPU;
  int iterations, printskip;
  iterations = (argc>2) ? atoi(argv[2]) : 0; // forever, if unspecified
  printskip = (argc>3) ? atoi(argv[3]) : 1; // print each time, if unspecified
  printskip = ((printskip>iterations)||(printskip<1)) ? iterations : printskip;
  if ((argc > 4) && ((argv[4][0])=='C'))
    pu = CL_DEVICE_TYPE_CPU;

  // Initialize M, N and grid[] from stdin until EOF
  for (M = N = 0; (in ? fscanf(in, "%s", line) : scanf("%s", line)) > 0; M++) {
    int len, bin;
    char* word;

    if ((len = strlen(line)) > N)
      N = len;

    for (word = line; len > 30; len -= 30) {
      bin = strtob(&word, 30, "- _") << 1;
      //printf("%s (%d) returned %d\n", line, len, bin);
      grid[item] = bin;
      if ((item += 2) > (max - 2))
	exit(-1);
    }
    grid[item] = strtob(&word, len, "- _") << (31 - len);
    //printf("%s (%d) returned %d\n", line, len, grid[item]);
    item += 2;
  }
  //FIXME: need to transpose if N > 30!
  if (in)
    fclose(in);
  printGrid(grid, M, N);
             
   // Set up the OpenCL environment
  cl_int status;
  int Mwg, Nwg, Mwi, Nwi /* max 32-2=30 in one item */ = (N>30) ? 30 : (N?N:1);
  
   // Discover platform
  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform, NULL);
  chk(status, "clGetPlatformIDs", NULL, NULL);

   // Discover device
  cl_device_id device;
  printf("ideal workgroup size %d*%d, requesting %cPU\n", M, Nwi=(N-1+Nwi)/Nwi,
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
  printf("the max workgroup size of which is reported as %d*%d\n", (int)dims[0],
                                                                 (int) dims[1]);
  Mwi = (dims[0] < M) ? dims[0] : M; // FIXME: will ^D on its own seg fault?
  Mwg = (M-1 + Mwi)/Mwi;
  Nwi = (dims[1] < Nwi) ? dims[1] : Nwi;
  Nwg = ((N-1)/30 + Nwi)/Nwi;
  const size_t dataSize = sizeof(int)*(Mwg*Nwg)*(Mwi*Nwi); //=sizeof(int)*item;
  printf("requesting %d*%d workgroups %d*%d=%d\n", Mwg, Nwg, Mwi, Nwi, item/2);

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
  const char* source = readSource("gamelife.cl");
  cl_program program;
  program = clCreateProgramWithSource(context, 1, &source, NULL, &status);
  chk(status, "clCreateProgramWithSource", NULL, NULL);
  status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  chk(status, "clBuildProgram", &device, &program);

  // Create the kernel objects and arguments
  // FIXME: is a cl_int at these addresses changeable? redo clSetKernelArg()?
  cl_kernel evolve = clCreateKernel(program, "evolve", &status);
  chk(status, "clCreateKernel", NULL, NULL);

  // Create space for the grid on the device
  // FIXME: can we use CL_MEM_USE_HOST_PTR to avoid clEnqueueWriteBuffer()?
  // FIXME: inefficient for getting a return value?
  cl_mem d_grid, d_invalid;
  d_grid = clCreateBuffer(context, 0, dataSize, NULL, &status);
  chk(status, "clCreateBuffer", NULL, NULL);

  for (cl_int i = 0; (iterations == 0) || (i < iterations); i++) {
    if (i == 0) {
      cl_int sl = 2, sh = 3, rl = 3, rh = 3;

      // set don't need to set arguments each time
      status  = clSetKernelArg(evolve, 0, sizeof(cl_mem), &d_grid);
      status |= clSetKernelArg(evolve, 1, sizeof(cl_int), &sl);
      status |= clSetKernelArg(evolve, 2, sizeof(cl_int), &sh);
      status |= clSetKernelArg(evolve, 3, sizeof(cl_int), &rl);
      status |= clSetKernelArg(evolve, 4, sizeof(cl_int), &rh);
      status |= clSetKernelArg(evolve, 5, sizeof(cl_int), &i);
      chk(status, "clSetKernelArg", NULL, NULL);

      // don't need to send inputs to the device each time: try SHARED MEMORY?
      status = clEnqueueWriteBuffer(queue, d_grid, CL_TRUE /*blocking_write*/,
				    0 /*offset*/, dataSize, grid,
				    0 /*events_in_ ...*/,
				    NULL /*event_wait_list*/,
				    NULL /*event*/);
      chk(status, "clEnqueueWriteBuffer", NULL, NULL);
    }
    
    // Set the work item dimensions
    size_t globalSize[2] = {M, Nwg};
    size_t localSize[2] = {Mwi, 1};
    status = clEnqueueNDRangeKernel(queue, evolve, 2 /*dim*/, NULL,
				    globalSize, localSize, 0, NULL, NULL);
    chk(status, "clEnqueueNDRangeKernel", NULL, NULL);
    
    if ((printskip == 0) || ((i+1) % printskip == 0)) {
      // Copy outputs back from the device: OMIT?  SHARED MEMORY?
      status = clEnqueueReadBuffer(queue, d_grid, CL_TRUE /*blocking_read*/,
				   0 /*offset*/, dataSize, grid,
				   0 /*events_in_ ...*/,
				   NULL /*event_wait_list*/,
				   NULL /*event*/);
      chk(status, "clEnqueueReadBuffer", NULL, NULL);
      printf("%d/%d:\n", (i+1), iterations);
      printGrid(&grid[i&1], M, N); //<- thus is handled the odd/even word offset
    }
  }
  exit(0);
}
