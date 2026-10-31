# inference-engine

A from-scratch C++23 port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)

Wanted to dip my toes into LLMs by starting out with an inference engine.
Intending to potentially find performance improvements and support other open
weight models.

## Quick start

```sh
make download   # fetches stories15M.bin and tokenizer.bin
make
./run -m stories15M.bin -t 0 -n 256 -i "Once upon a time"
```

## Credits

Based on [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, MIT licensed.
