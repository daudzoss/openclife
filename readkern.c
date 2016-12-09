// renamed from convolution.c in ch. 4 of "Heterogeneous Computing with OpenCL"

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// This function reads in a text file and stores it as a char pointer
char* readSource(char* kernelPath) {

   cl_int status;
   FILE *fp;
   char *source;
   long int size;

   printf("Program file is: %s\n", kernelPath);

   fp = fopen(kernelPath, "rb");
   if(!fp) {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   status = fseek(fp, 0, SEEK_END);
   if(status != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   source = (char *)malloc(size + 1);

   int i;
   for (i = 0; i < size+1; i++) {
      source[i]='\0';
   }

   if(source == NULL) {
      printf("Error allocating space for the kernel source\n");
      exit(-1);
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}

void chk(cl_int status, const char* cmd, cl_device_id* dev, cl_program* program) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);

      // from buildProgramDebug.c found in a blog at http://dhruba.name

      if (program && dev) {
         // build failed
         char* programLog;
         size_t logSize;

	 // check build error and build status first
	 clGetProgramBuildInfo(*program, *dev, CL_PROGRAM_BUILD_STATUS,
			       sizeof(cl_build_status), &status, NULL);
 
	 // check build log
	 clGetProgramBuildInfo(*program, *dev,
			       CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	 programLog = (char*) calloc (logSize+1, sizeof(char));
	 clGetProgramBuildInfo(*program, *dev,
			       CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
	 printf("Build failed; status=%d, programLog:nn%s",
		status, programLog);
	 free(programLog);
      }

      exit(-1);
   }
}
