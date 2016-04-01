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

// Input: Filename
int main(int argc, char *argv[]) 
{
	// Do huffman

	//Set the number of blocks and threads
	dim3 grid(4, 3, 1);
	dim3 block(32, 32, 1);
	
	deflatekernel<<<grid, block>>>(unsigned int size, unsigned int *huffman_table, char *data_in, char *data_out);
	
	// Inflate data_out using zlib
	
	// Compare inflated data with input

	return 0;
} 
