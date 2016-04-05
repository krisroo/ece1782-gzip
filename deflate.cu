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

#define CHUNK_SIZE 32000

// Input: Filename
int main(int argc, char *argv[]) 
{
   int i;
   int f_in;
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

   f_in = open(inputfname, O_RDONLY);
   fstat(f_in, &finfo);

   f_out = (char*) malloc(finfo.st_size);
   unsigned int data_bytes = (unsigned int)finfo.st_size;
   printf("This file has %d bytes data\n", data_bytes);

   read (f_in, f_out, data_bytes);
   
   

   //Set the number of blocks and threads
   dim3 grid(1, 1, 1);
   dim3 block(1024, 1, 1);

   char* d_in;
   cudaMalloc((void**) &d_in, data_bytes);
   cudaMemcpy(d_in, f_in, data_bytes, cudaMemcpyHostToDevice);

   char* d_out;
   cudaMalloc((void**) &d_out, data_bytes);
   cudaMemset(d_out, 0, data_bytes);

   deflatekernel<<<grid, block>>>(data_bytes, d_in, d_out);

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
