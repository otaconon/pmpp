## The Grid

- `gridDim.x` values are in range from $1$ to $2^{31}-1$, and those of `gridDim.y` and `gridDim.z` are in range from $1$ to $2^{16}-1$.
- Block indices are in range of the grid dimensions

## Blocks

- Blocks are three dimensional
- Block can have at most 1024 threads, which can be distributed in any way between the blocks dimensions
