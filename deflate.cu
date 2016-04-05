#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/time.h>

#include "deflate_kernel.cu"

// defined in deflate_kernel.cu
// define CHUNK_SIZE 32768
// define THREAD_NUM 1024


// Input: Filename
int main(int argc, char *argv[]) 
{
   int i;
   int f_handle;
   char *f_in;
   char *f_out;
   struct stat finfo;  
   char * inputfname;
   char * outputfname;

   if (argc < 3)
   {
      printf("USAGE: %s <input filename> <output filename>\n", argv[0]);
      exit(1);
   }

   inputfname = argv[1];
   outputfname = argv[2];

   f_handle = open(inputfname, O_RDONLY);
   fstat(f_handle, &finfo);

   f_in = (char*) malloc(finfo.st_size);
   f_out = (char*) malloc(finfo.st_size);
   unsigned int data_bytes = (unsigned int)finfo.st_size;
   printf("This file has %d bytes data\n", data_bytes);

   read (f_handle, f_in, data_bytes);
   
   

   //Set the number of blocks and threads
   dim3 grid(1, 1, 1);
   dim3 block(THREAD_NUM, 1, 1);

   char* d_in;
   cudaMalloc((void**) &d_in, data_bytes);
   cudaMemcpy(d_in, f_in, data_bytes, cudaMemcpyHostToDevice);

   char* d_out;
   cudaMalloc((void**) &d_out, data_bytes);
   cudaMemset(d_out, 0, data_bytes);

   
   struct timeval start_tv, end_tv;
   time_t sec;
   time_t ms;
   time_t diff;
   gettimeofday(&start_tv, NULL);

   deflatekernel<<<grid, block>>>(data_bytes, d_in, d_out);

   cudaThreadSynchronize();

   gettimeofday(&end_tv, NULL);
   sec = end_tv.tv_sec - start_tv.tv_sec;
   ms = end_tv.tv_usec - start_tv.tv_usec;

   diff = sec * 1000000 + ms;

   printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));


   cudaMemcpy(f_out, d_out, data_bytes, cudaMemcpyDeviceToHost);

   // Inflate data_out using zlib
      // Meh
   // Compare inflated data with input
      // whatever

   FILE *writeFile;
   writeFile = fopen(outputfname,"w+");
   for(i = 0; i < data_bytes; i++)
      fprintf(writeFile,"%c", f_out[i]);
   fclose(writeFile);

   return 0;
} 
