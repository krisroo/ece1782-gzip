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

#define CHUNK_SIZE 1024

// Input: Filename
int main(int argc, char *argv[]) 
{
   int i;
   int fd;
   char *fdata;
   struct stat finfo;  
   char * inputfname;
   char * outputfname;

   unsigned int size;
   unsigned int *huffman_table;
   char *data_in;
   char *data_out;

   if (argc < 3)
   {
      printf("USAGE: %s <input filename> <output filename>\n", argv[0]);
      exit(1);
   }

   inputfname = argv[1];
   outputfname = argv[2];

   fd = open(inputfname, O_RDONLY);
   fstat(fd, &finfo);

   fdata = (char*) malloc(finfo.st_size);
   int data_bytes = (int)finfo.st_size;
   printf("This file has %d bytes data\n", data_bytes);

   read (fd, fdata, data_bytes);
   

   //Set the number of blocks and threads
//   dim3 grid(4, 3, 1);
//   dim3 block(32, 32, 1);

//   deflatekernel<<<grid, block>>>(unsigned int size, unsigned int *huffman_table, char *data_in, char *data_out);

   // Inflate data_out using zlib

   // Compare inflated data with input

   FILE *writeFile;
   writeFile = fopen(outputfname,"w+");
   for(i = 0; i < data_bytes; i++)
      fprintf(writeFile,"%c", fdata[i]);
   fclose(writeFile);

   return 0;
} 
