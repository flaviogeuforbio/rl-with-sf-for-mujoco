# 3D Features Implementation

This branch contains the initial implementation of the 3D feature logic for Successor Features transfer learning. 

## Implemented Tasks
The codebase currently supports the following sequential task evaluations:
* **HalfCheetah Forward $\rightarrow$ HalfCheetah Backward**
* **HalfCheetah Forward $\rightarrow$ Walker2d Forward**

## Important Note on Checkpointing
This branch does **not** include the checkpointing infrastructure. For the finalized Walker2d task optimization, including full state restoration and safe execution checkpointing, please refer to the `optimization` branch.
