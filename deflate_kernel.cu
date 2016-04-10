#ifndef _DEFLATE_KERNEL_H_
#define _DEFLATE_KERNEL_H_

#define CHUNK 32768
#define SMEM_W 128
#define SMEM_H 128

#define REPL_BIT 24  // 3bytes
#define FLAG_BIT 1
#define DIST_BIT 15
#define LEN_BIT  8

#define MAX_MATCH 255
#define MIN_MATCH 3

__global__ void deflatekernel(unsigned int size, char *data_in, char *data_out, unsigned int* output_size)
{
    __shared__ char window[CHUNK];
    __shared__ unsigned int max[1024];
    __shared__ unsigned int pointer[1024];
    
    
    int blockid = blockIdx.x;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    
    int aligned_chunk_count = ((size + CHUNK - 1)/CHUNK);
    
    int last_chunk_size = (size%CHUNK > 0) ? size%CHUNK : CHUNK;
    
    //unsigned int last_chunk_size = aligned_chunk_count*CHUNK - size;
    
    for (unsigned int chunk_repeat = 0; chunk_repeat < aligned_chunk_count; chunk_repeat += gridDim.x)
    {
        unsigned int device_output_size = 0;
        if (chunk_repeat + blockid >= aligned_chunk_count)
            break;
            
            
        char *current_output = data_out + (chunk_repeat + blockid)*CHUNK;
        int block_size = (chunk_repeat + blockid + 1 == aligned_chunk_count) ? last_chunk_size : CHUNK;
        
        max[tid] = 0;
        for (int i = 0; i < block_size; i = i + blockDim.x)
        {
            window[i + tid] = data_in[i + tid + (chunk_repeat+blockid)*CHUNK];
        }
        __syncthreads();
        
        int divider = 0;

        while(divider < block_size)
        {
            max[tid] = 0;
            pointer[tid] = 0;
            
            for (int match_offset = 0; match_offset < divider; match_offset = match_offset + blockDim.x)
            {
                if (match_offset + tid >= divider)
                    break;
                    
                unsigned int length = 0;
                
                while (  (length < MAX_MATCH)
                       &&((divider+length)<block_size) 
                       &&(window[match_offset+tid+length] == window[divider+length]))
                    length++;

                if (length < MIN_MATCH)
                    continue;

                if (length >= max[tid])
                {
                    max[tid] = length;
                    pointer[tid] = divider - (match_offset + tid);
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int i = 512; i > 0; i = i >> 1)
            {
                if (tid < i)
                {
                    if (max[tid+i] > max[tid])
                    {
                        max[tid] = max[tid+i];
                        pointer[tid] = pointer[tid+i];
                    }
                }
                __syncthreads();
            }
            
            if ((tid == 0) && (pointer[0] == 0))
            {
                //*(current_output) = 1;
                *(current_output) = window[divider];
                current_output = current_output + 1;
                device_output_size += 1;
            } else if (tid == 0)
            {
                *(current_output+0) = (pointer[0] >> 8) | 0x80 & 0xff; // OR flag bit
                *(current_output+1) = pointer[0] & 0xff;
                *(current_output+2) = max[0] & 0xff;
                current_output = current_output + 3;
                device_output_size += 3;
            }
            if (pointer[0] == 0)
            {
                divider = divider + 1;
            } else
            {
                divider = divider + max[0];
            }
            
            __syncthreads();
        }
        if (tid == 0)
            *(output_size + chunk_repeat + blockid) = device_output_size;
    }
        
    return;
}

#endif // _DEFLATE_KERNEL_H_
