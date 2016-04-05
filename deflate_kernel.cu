#ifndef _DEFLATE_KERNEL_H_
#define _DEFLATE_KERNEL_H_

#define CHUNK_SIZE 32768
#define THREAD_NUM 1024


__global__ void deflatekernel(unsigned int size, char *data_in, char *data_out)
{
   __shared__ char sdata[CHUNK_SIZE];
   int idx = threadIdx.x;
   int iteration = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
   
   // Iterate through all of chunks
   for (int iter = 0; iter < iteration; iter++)
   {
      // Load your own share of data, threads..
      for (int i = 0; i < CHUNK_SIZE/THREAD_NUM; i++)
      {
         int cpByte_in = idx + i * THREAD_NUM + iter * CHUNK_SIZE;
         int cpByte_s  = idx + i * THREAD_NUM;
         
         if (cpByte_in < size)
         {
            sdata[cpByte_s] = data_in[cpByte_in];
         }
      }
      __syncthreads();

      // Search algo and fill output

      // temp code to straight copy to output
      for (int i = 0; i < CHUNK_SIZE/THREAD_NUM; i++)
      {
         int cpByte_out = idx + i * THREAD_NUM + iter * CHUNK_SIZE;
         int cpByte_s  = idx + i * THREAD_NUM;
         
         if (cpByte_out < size)
         {
            data_out[cpByte_out] = sdata[cpByte_s];
         }
      }
      __syncthreads();
   }

   return;
}

#endif // _DEFLATE_KERNEL_H_

