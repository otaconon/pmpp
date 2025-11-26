## Program Structure

1. Execution of a cuda program starts with host code.
2. When a kernel function is called number of threads are launched on a device to execute the kernel. Those threads are called a **grid**.
3. When all threads in a grid have completed the execution, the grid terminates, and host execution continues.

## CUDA design

- CUDA C programming is single-program multiple-data (SPMD).
- Threads are grouped into blocks which make up the grid.

| Qualifier Keyword | Callable From  | Executed On | Executed By                |
| ----------------- | -------------- | ----------- | -------------------------- |
| `__host__`        | Host           | Host        | Caller host thread         |
| `__global__`      | Host or Device | Device      | New grid of device threads |
| `__device__`      | Device         | Device      | Caller device thread       |

## Key properties of the GPU

- Launching a thread takes very few clock cycles.
- Modern GPU's often come with their own dynamic random-access memory, called **global memory**.

## Compilation

 CUDA program is compiler with an NVCC compiler:
 1. NVCC compiler separates code into host and device part. Device code is compiled into virtual binary files called PTX files.
 2. Host code is compiled with a standard C/C++ compiler.
 3. PTX files are further compiled by the runtime NVCC component, into real object files executed on a CUDA-capable GPU device.