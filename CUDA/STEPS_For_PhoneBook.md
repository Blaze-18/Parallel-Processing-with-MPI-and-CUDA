
# CUDA Phonebook Search — Step-by-Step Meaning

## STEP 0 — Headers, constants, structures

**Purpose:** Prepare everything required before execution.

* Include **C++ + CUDA libraries** for file handling, memory, and GPU execution.
* Define `MAX_STR_LEN = 50` → fixed memory size per contact name (important for GPU).
* Create `ResultContact` struct → used **after GPU search** to store and **sort matches alphabetically on CPU**.

👉 GPU does searching, **CPU handles sorting & printing**.

---

## STEP 1 — Read command-line input

**Purpose:** Get user search parameters.

* User provides:

  * `search_string`
  * `threads_per_block`
* Replace `_` with space → allows searching names like `"Md_Rahim"`.

👉 Defines **what to search** and **how many GPU threads per block**.

---

## STEP 2 — Load phonebook on CPU

**Purpose:** Read file into normal RAM first.

* Open phonebook text file.
* Extract:

  * **name**
  * **phone number**
* Store in:

  * `host_names_vec`
  * `host_numbers_vec`
* Count total contacts.

👉 GPU **cannot read files directly**, so CPU must load first.

---

## STEP 3 — Prepare flat memory for GPU

**Purpose:** Convert flexible C++ strings → fixed GPU-friendly array.

* Allocate:

  * `h_names` → continuous char array (`num_contacts × MAX_STR_LEN`)
  * `h_results` → match flags.
* Copy each name into fixed-size slot.

👉 GPUs work best with **simple linear memory**, not `vector<string>`.

---

## STEP 4 — Allocate GPU memory

**Purpose:** Create storage on the **device (VRAM)**.

* `d_names` → all names
* `d_results` → match results
* `d_search_name` → search keyword

👉 Without this, GPU **cannot access data**.

---

## STEP 5 — Copy CPU → GPU

**Purpose:** Move data from **host RAM → device VRAM**.

* Transfer:

  * names
  * search string

👉 This transfer is **mandatory before kernel execution**.

---

## STEP 6 — Configure CUDA grid

**Purpose:** Decide **parallel execution size**.

* Compute number of **blocks** needed:

```
blocks = ceil(num_contacts / threads_per_block)
```

👉 Ensures **one GPU thread per contact**.

---

## STEP 7 — Launch kernel (parallel search)

**Purpose:** Perform **actual phonebook search on GPU**.

Inside kernel:

1. Each thread computes **global index**

```
idx = blockIdx.x * blockDim.x + threadIdx.x
```

2. If index valid:

   * Read **one contact name**
   * Run **substring match (`check`)**
   * Store **0 or 1** in results.

👉 **Thousands of contacts searched simultaneously**.

---

## STEP 8 — Copy results back to CPU

**Purpose:** Retrieve GPU computation outcome.

* Transfer `d_results → h_results`.

👉 CPU now knows **which contacts matched**.

---

## STEP 9 — Collect matched contacts

**Purpose:** Build readable result list.

* For each index:

  * If result = 1 → push `{name, number}` into vector.

👉 Converts **binary flags → real contacts**.

---

## STEP 10 — Sort alphabetically

**Purpose:** Improve output readability.

* Use C++ `sort()` with overloaded `<` operator.

👉 Sorting is done on **CPU (simpler than GPU sorting)**.

---

## STEP 11 — Print results

**Purpose:** Show final search output.

* Display:

```
Name  Number
```

in **ascending order**.

---

## STEP 12 — Free memory

**Purpose:** Prevent memory leaks.

* Free:

  * CPU memory (`free`)
  * GPU memory (`cudaFree`)

👉 Always required in CUDA programs.

---

# One-Line Flow Summary

```
Read file (CPU)
   ↓
Copy data to GPU
   ↓
Parallel search by thousands of threads
   ↓
Copy results back
   ↓
Sort + print on CPU
```
