default:
    nvcc -arch=sm_52 -o deflate deflate.cu
clean:
    rm -f deflate
