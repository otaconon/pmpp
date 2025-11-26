## 1. 
If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index?

`i = blockIdx.x*blockDim.x + threadIdx.x`
## 2. 
Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

`i = (blockIdx.x*blockDim.x + threadIdx.x)*2

## 3. 
We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2*blockDim.x` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable `i` should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

`i = blockIdx.x*blockDim.x*2 + threadIdx.x`

## 4. 
For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

`8192`

## 5. 
If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

`v * sizeof(int)`

## 6. 
If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?

`(void**)&A_d`

## 7. 
If we want to copy 3000 bytes of data from host array `A_h` (`A_h` is a pointer to element 0 of the source array) to device array `A_d` (`A_d` is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?

`cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`

## 8.
How would one declare a variable err that can appropriately receive the
returned value of a CUDA API call?

`cudaError_t err;`

## 9.
Consider the following CUDA kernel and the corresponding host function
that calls it:
```cpp
__global__ void foo_kernel(float* a, float* b, unsigned int N){
  unsigned int i=blockIdx.x*blockDim.x + threadIdx.x;
  if(i < N) {
    b[i]=2.7f*a[i] - 4.3f;
  }
}

void foo(float* a_d, float* b_d) {
  unsigned int N=200000;
  foo_kernel<<<(N + 128-1)/128, 128>>>(a_d, b_d, N);
}
```

### a) What is the number of threads per block?

128 as indicated by the second parameter inside `<<<>>>`.
### b) What is the number of threads in the grid?

The number of blocks is $\frac{N + 128 - 1}{128} = 1563$, and the number of threads per block is $128$, thus the number of threads in total is $1563 \cdot 128 = 200064$.

### c) What is the number of blocks in the grid?

As stated in b), its $1563$.

### d) What is the number of threads that execute the code on line 02?

Every thread will execute this line, thus the answer is $200064$.

### e) What is the number of threads that execute the code on line 04?

Only threads with index `i < N` will execute line 4, thus the answer is $200000$.

## 10.
A new summer intern was frustrated with CUDA. He has been complaining
that CUDA is very tedious. He had to declare many functions that he plans
to execute on both the host and the device twice, once as a host function and
once as a device function. What is your response?

Bro you are clueless, just define em' `__global__`.

