![Sparrow](sparrow.png)

Sparrow LLM is a mini llm inference engine

# Dev log
- [x] Run llama 3.2 1B on a GPU using this engine.
    - [x] Load HF config.json
    - [x] Load HF weights
    - [x] Implement Llama layers
    - [x] Instantiate model using config.json
    - [x] Load weights into model class
- [ ] inference optimizations
    - [ ] Flashinfer kernels
    - [ ] KV Caching
    - [ ] Tensor Parallelism
    - [ ] Radix Tree
- [ ] usability
    - [ ] Connect with an HTTP endpoint, hit /generate.
- [ ] testing
    - [ ] add an all-close test with hf 1B, 8B.