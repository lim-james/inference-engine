#include <algorithm>
#include <chrono>
#include <string_view>
#include <print>
#include <utility>
#include <optional>
#include <string>
#include <vector>

#include <cstring>
#include <cstdio>
#include <cmath>

#include <getopt.h> 
#include <fcntl.h> 
#include <sys/mman.h> 

#include <type_traits>
#include <memory>
#include <concepts>
#include <span>

template<typename U, typename T>
concept HasPointerGetter = requires (U u) { {u.get()} -> std::same_as<T*>; };

template<typename U, typename T>
concept HasDataGetter = requires (U u) { {u.data()} -> std::same_as<T*>; };

template<typename T>
class view {
public:
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    constexpr view(T* ptr = nullptr) noexcept : ptr_{ptr} {}

    template<typename Ptr> requires HasPointerGetter<Ptr, T>
    constexpr view(const Ptr& ptr) noexcept : ptr_{ptr.get()} {}

    template<typename Data> requires HasDataGetter<Data, T>
    constexpr view(Data& data) noexcept : ptr_{data.data()} {}

    constexpr pointer data() { return ptr_; }
    constexpr const_pointer data() const { return ptr_; }

    constexpr reference operator*() { return *ptr_; }
    constexpr const_reference operator*() const { return *ptr_; }

    constexpr reference operator[](std::size_t idx) { return ptr_[idx]; }
    constexpr const_reference operator[](std::size_t idx) const { return ptr_[idx]; }

    constexpr pointer operator->() { return ptr_; }
    constexpr const_pointer operator->() const { return ptr_; }

    constexpr view operator+(int off) { return view{ptr_ + off}; }
    constexpr view operator+(int off) const { return view{ptr_ + off}; }

    constexpr view& operator++() { ++ptr_; return *this; }
    constexpr view operator++(int) { view copy = *this; ++*this; return copy; }

    constexpr view& operator--() { --ptr_; return *this; }
    constexpr view operator--(int) { view copy = *this; --*this; return copy; }

    constexpr operator value_type() { return *ptr_; }
    constexpr operator value_type() const { return *ptr_; }
    constexpr operator reference() { return *ptr_; }
    constexpr operator const_reference() const { return *ptr_; }
    constexpr operator pointer() { return ptr_; }
    constexpr operator const_pointer() const { return ptr_; }

    constexpr auto operator<=>(const view&) const = default;
    constexpr bool operator==(const view&) const = default;

private:
    T* ptr_;
};

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    // token embedding table
    view<float> token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    view<float> rms_att_weight; // (layer, dim) rmsnorm weights
    view<float> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    view<float> wq; // (layer, dim, n_heads * head_size)
    view<float> wk; // (layer, dim, n_kv_heads * head_size)
    view<float> wv; // (layer, dim, n_kv_heads * head_size)
    view<float> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    view<float> w1; // (layer, hidden_dim, dim)
    view<float> w2; // (layer, dim, hidden_dim)
    view<float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    view<float> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    view<float> wcls;
};

struct RunState {
    // current wave of activations
    std::unique_ptr<float[]> x; // activation at current time stamp (dim,)
    std::unique_ptr<float[]> xb; // same, but inside a residual branch (dim,)
    std::unique_ptr<float[]> xb2; // an additional buffer just for convenience (dim,)
    std::unique_ptr<float[]> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> q; // query (dim,)
    std::unique_ptr<float[]> att; // buffer for scores/attention values (n_heads, seq_len)
    std::unique_ptr<float[]> logits; // output logits
    // kv cache
    std::unique_ptr<float[]> key_cache;   // (layer, seq_len, dim)
    std::unique_ptr<float[]> value_cache; // (layer, seq_len, dim)

    view<float> k; // key (dim,)
    view<float> v; // value (dim,)
};

struct Transformer {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    std::span<float> data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
};

void malloc_run_state(RunState& state, const Config& p) {
    const int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    RunState s{
        .x = std::make_unique<float[]>(p.dim),
        .xb = std::make_unique<float[]>(p.dim),
        .xb2 = std::make_unique<float[]>(p.dim),
        .hb = std::make_unique<float[]>(p.hidden_dim),
        .hb2 = std::make_unique<float[]>(p.hidden_dim),
        .q = std::make_unique<float[]>(p.dim),
        .att = std::make_unique<float[]>(p.n_heads * p.seq_len),
        .logits = std::make_unique<float[]>(p.vocab_size),
        .key_cache = std::make_unique<float[]>(p.n_layers * p.seq_len * kv_dim),
        .value_cache = std::make_unique<float[]>(p.n_layers * p.seq_len * kv_dim),
    };
    // ensure all mallocs went fine
    if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 || !s.q
     || !s.key_cache || !s.value_cache || !s.att || !s.logits) {
        std::println(stderr, "malloc failed!");
        std::exit(EXIT_FAILURE);
    }
    state = std::move(s);
}

void memory_map_weights(TransformerWeights& w, Config& p, std::span<float> ptr, int shared_weights) {
    int head_size = p.dim / p.n_heads;

    unsigned long long n_layers = p.n_layers;
    w.token_embedding_table = ptr;
    ptr = ptr.subspan(p.vocab_size * p.dim);
    w.rms_att_weight = ptr;
    ptr = ptr.subspan(n_layers * p.dim);
    w.wq = ptr;
    ptr = ptr.subspan(n_layers * p.dim * (p.n_heads * head_size));
    w.wk = ptr;
    ptr = ptr.subspan(n_layers * p.dim * (p.n_kv_heads * head_size));
    w.wv = ptr;
    ptr = ptr.subspan(n_layers * p.dim * (p.n_kv_heads * head_size));
    w.wo = ptr;
    ptr = ptr.subspan(n_layers * (p.n_heads * head_size) * p.dim);
    w.rms_ffn_weight = ptr;
    ptr = ptr.subspan(n_layers * p.dim);
    w.w1 = ptr;
    ptr = ptr.subspan(n_layers * p.dim * p.hidden_dim);
    w.w2 = ptr;
    ptr = ptr.subspan(n_layers * p.hidden_dim * p.dim);
    w.w3 = ptr;
    ptr = ptr.subspan(n_layers * p.dim * p.hidden_dim);
    w.rms_final_weight = ptr;
    ptr = ptr.subspan(p.dim);
    ptr = ptr.subspan(p.seq_len * head_size / 2); // skip what used to be freq_cis_real (for RoPE)
    ptr = ptr.subspan(p.seq_len * head_size / 2); // skip what used to be freq_cis_imag (for RoPE)
    w.wcls = shared_weights ? w.token_embedding_table : ptr;
}

void read_checkpoint(std::string_view checkpoint, Config& config, TransformerWeights& weights,
                     int& fd, std::span<float>& data, ssize_t& file_size) {
    FILE *file = std::fopen(checkpoint.data(), "rb");
    if (!file) { std::println(stderr, "Couldn't open file {}", checkpoint); std::exit(EXIT_FAILURE); }
    if (std::fread(&config, sizeof(config), 1, file) != 1) std::exit(EXIT_FAILURE);

    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = std::abs(config.vocab_size);

    std::fseek(file, 0, SEEK_END);
    file_size = std::ftell(file);
    std::fclose(file);

    fd = open(checkpoint.data(), O_RDONLY);
    if (fd == -1) { std::println(stderr, "open failed"); std::exit(EXIT_FAILURE); }
    float* mmap_data = static_cast<float*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (mmap_data == MAP_FAILED) { std::println(stderr, "mmap failed"); std::exit(EXIT_FAILURE); }
    data = std::span{mmap_data, static_cast<std::size_t>(file_size)};

    auto weights_ptr = data.subspan(sizeof(config) / sizeof(float));
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

[[nodiscard]] Transformer make_transformer(std::string_view checkpoint_path) {
    Transformer t{};
    read_checkpoint(checkpoint_path, t.config, t.weights, t.fd, t.data, t.file_size);
    malloc_run_state(t.state, t.config);
    return t;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(view<float> o, view<float> x, view<float> weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; ++j)
        ss += x[j] * x[j];

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / std::sqrtf(ss);
    
    for (int j = 0; j < size; ++j)
        o[j] = weight[j] * (ss * x[j]);
}

void softmax(view<float> x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void matmul(view<float> xout, view<float> x, view<float> w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; ++i) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

view<float> forward(Transformer& transformer, int token, int pos) {
    Config& p = transformer.config;
    TransformerWeights& w = transformer.weights;
    RunState& s = transformer.state;

    view<float> x = s.x;
    int dim = p.dim;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p.hidden_dim;
    int head_size = dim / p.n_heads;

    auto content_row = w.token_embedding_table + token * dim; 
    std::copy_n(content_row.data(), dim, x.data());

    for (int l = 0; l < p.n_layers; ++l) {
        rmsnorm(s.xb, x, w.rms_att_weight + l * dim, dim);

        int loff = l * p.seq_len * kv_dim;
        s.k = s.key_cache.get() + loff + pos * kv_dim;
        s.v = s.value_cache.get() + loff + pos * kv_dim;

        matmul(s.q, s.xb, w.wq + l * dim * dim, dim, dim);
        matmul(s.k, s.xb, w.wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s.v, s.xb, w.wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / std::powf(10'000.0f, head_dim / static_cast<float>(head_size));
            float val = pos * freq;
            float fcr = std::cosf(val);
            float fci = std::sinf(val);
            int rotn = 1 < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; ++v) {
                view<float> vec = v == 0 ? s.q : s.k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p.n_heads; ++h) {
            view<float> q = s.q.get() + h * head_size;
            view<float> att = s.att.get() + h * p.seq_len;
            for (int t = 0; t <= pos; ++t) {
                view<float> k = s.key_cache.get() + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) {
                    score += q[i] * k[i];
                }
                score /= std::sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            view<float> xb = s.xb.get() + h * head_size;
            std::memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                view<float> v = s.value_cache.get() + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; ++i) {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim);

        for (int i = 0; i < dim; ++i) {
            x[i] += s.xb2[i];
        }

        rmsnorm(s.xb, x, w.rms_ffn_weight + l * dim, dim);

        matmul(s.hb, s.xb, w.w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3 + l * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; ++i) {
            float val = s.hb[i];
            val *= (1.0f / (1.0f + std::expf(-val)));
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        matmul(s.xb, s.hb, w.w2 + l * dim * hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; ++i) {
            x[i] += s.xb[i];
        }
    }

    rmsnorm(x, x, w.rms_final_weight, dim);

    matmul(s.logits, x, w.wcls, p.dim, p.vocab_size);
    return s.logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <. tokens

struct TokenIndex {
    std::string_view str;
    int id;
    auto operator<=>(const TokenIndex& other) const { return str <=> other.str; }
    bool operator==(const TokenIndex& other) const { return str == other.str; }
};

struct Tokenizer {
    std::unique_ptr<std::string[]> vocab;
    std::unique_ptr<float[]> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    std::array<unsigned char, 512> byte_pieces; 
};

[[nodiscard]] Tokenizer make_tokenizer(std::string_view tokenizer_path, int vocab_size) {
    Tokenizer t{};
    t.vocab_size = vocab_size;
    t.vocab = std::make_unique<std::string[]>(vocab_size);
    t.vocab_scores = std::make_unique<float[]>(vocab_size);
    for (int i = 0; i < 256; ++i) {
        t.byte_pieces[i * 2] = static_cast<unsigned char>(i);
        t.byte_pieces[i * 2 + 1] = '\0';
    }
    
    FILE* file = std::fopen(tokenizer_path.data(), "rb");
    if (!file) { std::println(stderr, "Couldn't open file {}", tokenizer_path); std::exit(EXIT_FAILURE); }
    if (std::fread(&t.max_token_length, sizeof(int), 1, file) != 1) { std::println(stderr, "failed read"); std::exit(EXIT_FAILURE); }

    int len;
    for (int i = 0; i < vocab_size; ++i) {
        if (std::fread(&t.vocab_scores[i], sizeof(float), 1, file) != 1) { std::println(stderr, "failed read"); std::exit(EXIT_FAILURE); }
        if (std::fread(&len, sizeof(int), 1, file) != 1) { std::println(stderr, "failed read"); std::exit(EXIT_FAILURE); }
        t.vocab[i].resize(len);
        if (std::fread(t.vocab[i].data(), len, 1, file) != 1) { std::println(stderr, "failed read"); std::exit(EXIT_FAILURE); }
    }

    fclose(file);

    return t;
}

std::string_view decode(Tokenizer& t, int prev_token, int token) {
    std::string_view piece = t.vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece = piece.substr(1);
    unsigned char byte_val;
    if (sscanf(piece.data(), "<0x%02hhX>", &byte_val) == 1) {
        piece = std::string_view{reinterpret_cast<char*>(t.byte_pieces.data()) + byte_val * 2};
    }
    return piece;
}

void safe_printf(std::string_view piece) {
    if (piece.empty()) return;
    if (piece.size() == 1) {
        unsigned char byte_val = piece[0];
        if (!(std::isprint(byte_val) || std::isspace(byte_val))) {
            return; 
        }
    }
    std::print("{}", piece);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct ProbIndex {
    float prob;
    int index;
}; 

struct Sampler {
    int vocab_size;
    std::unique_ptr<ProbIndex[]> probindex; 
    float temperature;
    float topp;
    unsigned long long rng_state;
};

[[nodiscard]] Sampler make_sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    return Sampler{
        .vocab_size = vocab_size,
        .probindex = std::make_unique<ProbIndex[]>(vocab_size),
        .temperature = temperature,
        .topp = topp,
        .rng_state = rng_seed,
    };
}

int sample_argmax(view<float> probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; ++i) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(view<float> probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int sample_topp(view<float> probabilities, int n, float topp, view<ProbIndex> probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; ++i) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    auto compare = [](const ProbIndex& a, const ProbIndex& b) {
        return a.prob > b.prob;
    };

    std::sort(probindex.data(), probindex.data() + n0, compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; ++i) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; ++i) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

unsigned int random_u32(unsigned long long& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long& state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler& sampler, view<float> logits) {
    int next;
    if (sampler.temperature == 0.0f) {
        next = sample_argmax(logits, sampler.vocab_size);
    } else {
        for (int q = 0; q < sampler.vocab_size; ++q) logits[q] /= sampler.temperature;
        softmax(logits, sampler.vocab_size);
        float coin = random_f32(sampler.rng_state);
        if (sampler.topp <= 0 || sampler.topp >= 1) {
            next = sample_mult(logits, sampler.vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
        }
    }
    return next;
}



// ----------------------------------------------------------------------------
// generation loop

int str_lookup(std::string_view str, std::vector<TokenIndex>& sorted_vocab, int vocab_size) {
    auto it = std::ranges::lower_bound(sorted_vocab, TokenIndex{.str=str});
    return (it != sorted_vocab.end() && it->str == str) ? it->id : -1;
}

void encode(Tokenizer& t, std::string_view text, int8_t bos, int8_t eos, view<int> tokens, int& n_tokens) {
    if (t.sorted_vocab.empty()) {
        t.sorted_vocab.reserve(t.vocab_size);
        for (int i = 0; i < t.vocab_size; ++i) {
            t.sorted_vocab.emplace_back(t.vocab[i], i);
        }
        std::ranges::sort(t.sorted_vocab);
    }

    std::string str_buffer;
    str_buffer.resize(t.max_token_length * 2 + 1 + 2);

    std::size_t str_len{};
    n_tokens = 0;

    if (bos) tokens[n_tokens++] = 1;

    if (!text.empty()) {
        tokens[n_tokens++] = str_lookup(" ", t.sorted_vocab, t.vocab_size);
    }

    for (const char *c = text.data(); *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;

        int id = str_lookup(std::string_view{str_buffer.data(), str_len}, t.sorted_vocab, t.vocab_size);

        if (id != -1) {
            tokens[n_tokens++] = id;
        } else {
            for (int i = 0; i < str_len; ++i) {
                tokens[n_tokens++] = static_cast<unsigned char>(str_buffer[i]) + 3;
            }
        }
        str_len = 0;
    }

    for (;;) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < n_tokens - 1; ++i) {
            str_buffer = std::format("{}{}", t.vocab[tokens[i]], t.vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t.sorted_vocab, t.vocab_size);
            if (id != -1 && t.vocab_scores[id] > best_score) {
                best_score = t.vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break; 

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; ++i)
            tokens[i] = tokens[i + 1];
        
        --n_tokens;
    }

    if (eos) tokens[n_tokens++] = 2;
}

void generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler, std::string_view prompt, int steps) {
    int num_prompt_tokens = 0;
    auto prompt_tokens = std::make_unique<int[]>(prompt.length() + 3); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, num_prompt_tokens); 
    if (num_prompt_tokens < 1) {
        std::println(stderr, "something is wrong, expected at least 1 prompt token");
        std::exit(EXIT_FAILURE);
    }


    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        view<float> logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        ++pos;

        if (next == 1) break;

        auto piece = decode(tokenizer, token, next);
        safe_printf(piece);
        std::fflush(stdout);
        token = next;
    }
    std::println();
}

void error_usage() {
    std::println(stderr, "Usage:   run <checkpoint> [options]\n");
    std::println(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    std::println(stderr, "Options:\n");
    std::println(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    std::println(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    std::println(stderr, "  -s <int>    random seed, default time(NULL)\n");
    std::println(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    std::println(stderr, "  -i <string> input prompt\n");
    std::println(stderr, "  -z <string> optional path to custom tokenizer\n");
    std::println(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    struct option options[] = {
        { .name = "temp",   .has_arg = optional_argument, .flag = nullptr, .val = 't' },
        { .name = "topp",   .has_arg = optional_argument, .flag = nullptr, .val = 'p' },
        { .name = "seed",   .has_arg = optional_argument, .flag = nullptr, .val = 's' },
        { .name = "steps",  .has_arg = optional_argument, .flag = nullptr, .val = 'n' },
        { .name = "input",  .has_arg = optional_argument, .flag = nullptr, .val = 'i' },
        { .name = "token",  .has_arg = optional_argument, .flag = nullptr, .val = 'z' },
        { .name = "model",  .has_arg = optional_argument, .flag = nullptr, .val = 'm' },
        { .name = "system", .has_arg = optional_argument, .flag = nullptr, .val = 'y' },
        { 0 }
    };

    float temperature = 1.f;
    float topp = 0.9f;
    std::optional<unsigned long long> rng_seed = 0;
    int steps = 256;

    std::string_view prompt{};
    std::string_view checkpoint_path{};
    std::string_view tokenizer_path = "tokenizer.bin";
    std::string_view system_prompt{};

    int opt;
    while ((opt = getopt_long(argc, argv, "t:p:s:n:i:z:m:y:", options, nullptr)) != -1) {
        switch (opt) {
            case 't': temperature = atof(optarg); break;
            case 'p': topp = atof(optarg); break;
            case 's': rng_seed = atoll(optarg); break;
            case 'n': steps = atoi(optarg); break;
            case 'i': prompt = optarg; break;
            case 'm': checkpoint_path = optarg; break;
            case 'z': tokenizer_path = optarg; break;
            case 'y': system_prompt = optarg; break;
            default: break;
        }
    }

    if (rng_seed == 0) rng_seed = static_cast<unsigned long long>(time(nullptr));
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    auto transformer = make_transformer(checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    auto tokenizer = make_tokenizer(tokenizer_path, transformer.config.vocab_size);
    auto sampler   = make_sampler(transformer.config.vocab_size, temperature, topp, *rng_seed);

    generate(transformer, tokenizer, sampler, prompt, steps);

    return 0;
}
