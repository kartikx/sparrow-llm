Sparrow LLM is a mini llm inference engine

# Dev log
- [ ] Run llama 3.2 1B on a GPU using this engine.
    - [x] Load HF config.json
    - [x] Load HF weights
    - [ ] Implement Llama layers
    - [ ] Instantiate model using config.json
    - [ ] Load weights into model class
    - [ ] Flashinfer kernels
    - [ ] KV Caching (later)
    - [ ] Tensor Parallelism
- [ ] Connect with an HTTP endpoint, hit /generate.