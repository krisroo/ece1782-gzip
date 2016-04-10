
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

#define KB 1024
#define MB 1024*KB
#define GB 1024*MB

#define THREAD_NUM 1024
#define FLAG 0xCC

// Assuming not going to reach 10% compression ratio
#define OUTSIZE_MULTIPLIER 10

char *inflate(char* compressed, size_t compressed_size, size_t *uncompressed_size)
{
    if ((compressed == NULL) | (uncompressed_size == NULL))
    {
        printf("Error: Invalid arguments for inflate.\n");
        *uncompressed_size = 0;
        return NULL;
    }
    
    if (compressed_size == 0)
    {
        printf("Error: Compressed file size was 0\n");
        *uncompressed_size = 0;
        return NULL;
    }
    
    char *uncompressed = (char*) malloc (OUTSIZE_MULTIPLIER*compressed_size);
    if (uncompressed == NULL)
    {
        printf("Error: Couldn't allocate memory for uncompressed file in inflate\n");
        printf("       Maybe the compressed file was too big\n");
        *uncompressed_size = 0;
        return NULL;
    }
    
    char *temp_compressed = compressed;
    char *temp_uncompressed = uncompressed;
    size_t outsize = 0;
    
    char *buffer;
    buffer = (char *) malloc(96*KB);
    
    size_t inbuffer = 0;
    
    for (size_t i = 0; i < compressed_size;)
    {
        //getchar();
        
        // Move 32KB over to output whenever there's more than 64KB
        while (inbuffer > 64*KB)
        {
            if (outsize > OUTSIZE_MULTIPLIER*compressed_size) {
                printf("Too lazy to dynamically increase output buffer\n");
                printf("Open source code and modify OUTSIZE_MULTIPLIER\n");
                free(uncompressed);
                *uncompressed_size = 0;
                return NULL;
            }
            memcpy(temp_uncompressed, buffer, 32*KB);
            memcpy(buffer, (buffer + 32*KB), 32*KB);
            inbuffer -= 32*KB;
            temp_uncompressed += 32*KB;
        }
        
        if ((*temp_compressed & 0xff) == FLAG)
        {
            unsigned int back = *(temp_compressed + 1) & 0xFF;
            back = (back << 8) + (*(temp_compressed + 2) & 0xFF);
            unsigned int length = *(temp_compressed + 3) & 0xFF;
            length = (length << 8) + (*(temp_compressed + 4) & 0xFF);
            printf("match found distance %04x and length %04x\n", back, length);
            temp_compressed += 5;
            i += 5;
            outsize += length;
            
            unsigned int back_location = inbuffer - back;

            while (length > 0)
            {
                unsigned int copy_amount = (length > back) ? back : length;
                memcpy((buffer+inbuffer), (buffer+back_location), copy_amount);
                length -= copy_amount;
                inbuffer += copy_amount;
                
            }
        } else
        {
            printf("no match found char is %02x\n", *temp_compressed);
            *(buffer + inbuffer) = *temp_compressed;
            temp_compressed += 1;
            inbuffer += 1;
            i += 1;
            outsize += 1;
        }
    }
    if (inbuffer > 0)
        memcpy(temp_uncompressed, buffer, inbuffer);
    
    return compressed;
    
}

static int inflate_flag;

// Input: Filename
int main(int argc, char *argv[]) 
{
   int f_handle;
   char *f_in;
   char *f_out;
   struct stat finfo;  
   char * inputfname;
   char * outputfname;
   char * outputfname_inflated;

   if (argc < 3)
   {
      printf("USAGE: %s <input filename> <output filename>\n", argv[0]);
      exit(1);
   }
   
   inflate_flag = 0;
   
   if (argc > 3)
      inflate_flag = 1;

   inputfname = argv[1];
   outputfname = argv[2];
   outputfname_inflated = argv[3];

   f_handle = open(inputfname, O_RDONLY);
   fstat(f_handle, &finfo);

   f_in = (char*) malloc(finfo.st_size);
   f_out = (char*) malloc(finfo.st_size);
   unsigned int data_bytes = (unsigned int)finfo.st_size;
   printf("This file has %d bytes data\n", data_bytes);

   read (f_handle, f_in, data_bytes);
   
   

   //Set the number of blocks and threads
   dim3 grid(12, 1, 1);
   dim3 block(THREAD_NUM, 1, 1);

   char* d_in;
   cudaMalloc((void**) &d_in, data_bytes);
   cudaMemcpy(d_in, f_in, data_bytes, cudaMemcpyHostToDevice);

   char* d_out;
   cudaMalloc((void**) &d_out, data_bytes);
   cudaMemset(d_out, 0, data_bytes);

   int aligned_chunk_count = ((data_bytes + CHUNK - 1)/CHUNK) - 1;
   printf("Number of blocks = %i\n",aligned_chunk_count);
   unsigned int *output_size;
   cudaMalloc((void**) &output_size, sizeof(unsigned int)*aligned_chunk_count);
   cudaMemset(output_size, 0, sizeof(unsigned int)*aligned_chunk_count);

   
   struct timeval start_tv, end_tv;
   time_t sec;
   time_t ms;
   time_t diff;
   gettimeofday(&start_tv, NULL);

   deflatekernel<<<grid, block>>>(data_bytes, d_in, d_out, output_size);

   cudaThreadSynchronize();

   gettimeofday(&end_tv, NULL);
   sec = end_tv.tv_sec - start_tv.tv_sec;
   ms = end_tv.tv_usec - start_tv.tv_usec;

   diff = sec * 1000000 + ms;

   printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));

   unsigned int *host_output_size = (unsigned int *)malloc(sizeof(unsigned int)*aligned_chunk_count);
   
   cudaMemcpy(host_output_size, output_size, sizeof(unsigned int)*aligned_chunk_count, cudaMemcpyDeviceToHost);
   
   unsigned int total_size = 0;
   for (int i=0; i < aligned_chunk_count; i++) {
      //printf("this chunk size is %u\n", *(host_output_size+i));
      total_size += *(host_output_size + i);
   }
      
   printf("total size was %u\n", total_size);
   
   cudaMemcpy(f_out, d_out, data_bytes, cudaMemcpyDeviceToHost);
   char *deflated_data = (char *) malloc (total_size);
   unsigned int deflated_data_pointer = 0;
  
   // Compare inflated data with input
      // whatever

   FILE *writeFile;
   writeFile = fopen(outputfname,"w+");
   for (unsigned int j = 0; j < aligned_chunk_count; j++) {
       for(unsigned int i = 0; i < *(host_output_size + j); i++)
       {
          fprintf(writeFile,"%c", f_out[i + (j*CHUNK)]);
          *(deflated_data + deflated_data_pointer) = f_out[i + (j*CHUNK)];
          deflated_data_pointer += 1;
       }
   }
   fclose(writeFile);
   
   if (inflate_flag)
   {
       // Inflate data_out using kris lib
       size_t inflated_data_size;
       char * inflated_data = inflate(deflated_data, total_size, &inflated_data_size);
       printf("Inflated data size was %u :: compared to original file %u\n", inflated_data_size, data_bytes);
       
       FILE *writeFile_inflated;
       writeFile_inflated = fopen(outputfname_inflated,"w+");
       for (unsigned int j = 0; j < inflated_data_size; j++) {
              fprintf(writeFile_inflated,"%c", inflated_data[j]);
       }
       fclose(writeFile_inflated);
       free(inflated_data);
   }
   free(deflated_data);
   return 0;
} 
