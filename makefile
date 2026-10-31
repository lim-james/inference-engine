CXX      ?= g++
SRC      ?= main.cpp
BIN      ?= run
STD      := -std=c++23
WARN     := -Wall -Wextra
LDLIBS   := -lm

.PHONY: run fast omp debug clean

# default: plain optimised build
run: $(SRC)
	$(CXX) $(STD) $(WARN) -O3 -o $(BIN) $(SRC) $(LDLIBS)

# aggressive maths and CPU-specific instructions
fast: $(SRC)
	$(CXX) $(STD) $(WARN) -Ofast -march=native -o $(BIN) $(SRC) $(LDLIBS)

# multithreaded via OpenMP
omp: $(SRC)
	$(CXX) $(STD) $(WARN) -O3 -march=native -fopenmp -o $(BIN) $(SRC) $(LDLIBS)

# correctness build: sanitizers, bounds-checked std library, single-threaded
debug: $(SRC)
	$(CXX) $(STD) $(WARN) -Wconversion -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		-D_GLIBCXX_ASSERTIONS -o $(BIN) $(SRC) $(LDLIBS)

clean:
	rm -f $(BIN)

download:
	wget -nc https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
	wget -nc https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
