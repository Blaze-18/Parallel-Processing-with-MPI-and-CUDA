# Matrix Multiplication Using CUDA

### Overall Flow of Execution

**Step 1** – Read command-line inputs
**Step 2** – Compute total memory sizes
**Step 3** – Allocate memory on CPU
**Step 4** – Initialize matrices with random values
**Step 5** – Allocate memory on GPU
**Step 6** – Copy matrices from CPU → GPU
**Step 7** – Launch CUDA kernel in batches
**Step 8** – Perform matrix multiplication in GPU threads
**Step 9** – Copy result from GPU → CPU
**Step 10** – Print first matrices and result
**Step 11** – Free GPU and CPU memory

---

## Step-by-Step Description

### Step 1 – Read command-line inputs

The program reads:

* number of **threads**
* number of **matrices (K)**
* matrix dimensions **M, N, P**

These control **parallel execution** and **matrix size**.

---

### Step 2 – Compute total memory sizes

It calculates how many elements are needed for:

* all **A matrices**
* all **B matrices**
* all **result matrices**

Because **K matrices** are processed in parallel.

---

### Step 3 – Allocate memory on CPU

Memory is reserved in RAM for:

* `h_A` → matrix A data
* `h_B` → matrix B data
* `h_R` → result matrix

These are **host (CPU) arrays**.

---

### Step 4 – Initialize matrices with random values

Each element of A and B is filled with **random numbers (0–9)**.

This creates **test input data** for multiplication.

---

### Step 5 – Allocate memory on GPU

Memory is created in **device (GPU) memory**:

* `d_A`, `d_B`, `d_R`

GPU cannot directly use CPU memory, so **separate allocation** is required.

---

### Step 6 – Copy matrices from CPU → GPU

Using `cudaMemcpy`:

* A and B are transferred to GPU.
* Result matrix is initialized to **zero**.

Now the GPU has **all required data**.

---

### Step 7 – Launch CUDA kernel in batches

Because GPU threads are limited:

* Matrices are processed in **batches = number of threads**.
* Loop keeps launching kernel until **all K matrices** are processed.

This enables **parallel execution of many matrices**.

---

### Step 8 – Perform matrix multiplication in GPU threads

Inside the CUDA kernel:

* Each **thread handles one matrix pair**.
* Standard **triple nested loops** compute multiplication.
* Outer loop repeats 100 times to **increase workload for timing**.

This is where **true parallelism happens**.

---

### Step 9 – Copy result from GPU → CPU

After computation:

* Result matrices are copied back to **CPU memory (`h_R`)**.

So CPU can **display the output**.

---

### Step 10 – Print first matrices and result

Program prints:

* first matrix **A[0]**
* first matrix **B[0]**
* first result **R[0]**

This verifies **correct multiplication**.

---

## Step 11 – Free GPU and CPU memory

Finally:

* `cudaFree()` releases GPU memory
* `free()` releases CPU memory

Prevents **memory leaks**.


## Compile and RUN Code

**Compile**: nvcc -arch=sm_75 matrix.cu -o matrix
**Run**: mpirun ./matrix 128 200 3 3 3