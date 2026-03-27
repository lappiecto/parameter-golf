# Parameter Golf: The Complete Story

## What We're Actually Doing

We are building a tiny brain.

Not metaphorically — literally. We are constructing a neural network that reads text and learns to predict what word comes next. The same fundamental mechanism that powers every large language model. The difference is scale: those models have hundreds of billions of parameters and train on trillions of tokens. Ours must fit in 16 megabytes.

To put that in perspective: a single high-resolution photo on your phone is about 5MB. We're building an entire language model in the space of three photos.

The competition — OpenAI's Parameter Golf — is beautiful in its constraint. It asks: given a brutally tight space budget, how smart can you make a model? Not how big. How *smart*. Every byte matters. Every architectural decision is a trade-off between capacity and compression. It's the engineering equivalent of writing poetry — every word must earn its place.

---

## The Machine: What a Transformer Actually Is

### The Core Insight

A transformer is a function that takes a sequence of tokens (chunks of text) and, for each position in that sequence, produces a probability distribution over what token comes next. That's it. Everything else — the attention heads, the MLP layers, the normalisation, the positional encoding — exists in service of that single goal: predicting the next token as accurately as possible.

But here's the thing that makes this profound: a system that can predict text well enough must, by necessity, build an internal model of the world that generated that text. To predict that "the cat sat on the ___" is probably followed by "mat", the model must encode something about cats, sitting, surfaces, and the English language. Prediction is a proxy for understanding.

### Tokens: The Atoms of Language

Before the model can process text, it needs to convert characters into numbers. This is tokenisation. Our model uses a 1024-token vocabulary — meaning it knows exactly 1024 "words" (some are actual words, some are fragments, some are individual characters).

For comparison, GPT-4 uses about 100,000 tokens. Our tiny vocabulary means the model needs more tokens to express the same text (like trying to write with only 1024 Scrabble tiles), but it also means the embedding table is small — and the embedding table is one of the biggest space consumers in the model.

The embedding table is a matrix of shape `[1024 × 512]`. Each row is a 512-dimensional vector that represents one token. When the model sees token #437, it looks up row 437 and gets a vector of 512 numbers. These numbers mean nothing at initialisation — they're random. But by the end of training, they encode rich semantic information: similar words end up as similar vectors, and the geometry of the space captures relationships between concepts.

### The Embedding: Where Meaning Lives

Each token starts as an index — a bare number. The embedding table converts that index into a vector in a 512-dimensional space. Think of this as giving each token a position in a vast room with 512 axes (we can't visualise this, but the mathematics works the same as in 3D space).

In this space, the model learns to place semantically similar tokens near each other. "Cat" and "dog" might end up close together because they appear in similar contexts. "The" and "a" cluster together because they're both articles. This geometric structure emerges purely from the training signal — nobody programs these relationships in.

We use **tied embeddings**: the same matrix that converts token IDs into vectors at the input is reused at the output to convert vectors back into token probabilities. This is a critical space-saving trick. Instead of two separate 1024×512 matrices (1 million parameters total), we use one (524,288 parameters). The model learns to make this single matrix work for both directions — encoding and decoding.

### Attention: The Model's Ability to Look Around

This is the heart of the transformer. Attention is the mechanism by which each token in the sequence can look at every other token that came before it and decide which ones are relevant.

Here's how it works mechanically:

1. For each token position, the model computes three vectors:
   - **Query (Q)**: "What am I looking for?"
   - **Key (K)**: "What do I contain?"
   - **Value (V)**: "What information can I provide?"

2. The model takes each Query and compares it against all preceding Keys using a dot product. This produces a score for each pair — high scores mean "this Key is relevant to this Query."

3. These scores are passed through softmax (which turns them into probabilities that sum to 1), creating an **attention pattern**: a distribution over all previous positions.

4. The Values are weighted by these probabilities and summed up. The result is a context-aware representation that mixes information from all relevant positions.

The magic: this is fully differentiable. The model can learn, through gradient descent, exactly which tokens should attend to which other tokens. Early in training, attention patterns are diffuse (everything looks at everything roughly equally). By the end, they're sharp and meaningful — one head might track syntactic relationships, another might track co-reference, another might handle positional patterns.

#### Multi-Head Attention

Our model uses 8 attention heads, each operating on 64 dimensions (8 × 64 = 512). Each head learns a different "type" of looking. One head might learn to always look at the previous token. Another might learn to look at the subject of the sentence. Another might specialise in detecting certain grammatical patterns.

The outputs of all 8 heads are concatenated back into a 512-dimensional vector and projected through a linear layer.

#### Grouped Query Attention (GQA)

Our model has 8 Query heads but only 4 Key-Value heads. This means pairs of Query heads share their Keys and Values. This saves 25% of the attention parameters with minimal quality loss — a technique pioneered by Llama 2 and now standard in efficient architectures.

The intuition: the diversity of "what am I looking for?" (Queries) matters more than the diversity of "what do I contain?" (Keys/Values). Two slightly different questions can productively share the same answer pool.

#### RoPE: Rotary Position Embeddings

The model needs to know *where* each token is in the sequence. RoPE encodes position by rotating the Query and Key vectors in a specific mathematical pattern. Tokens at position 5 get rotated differently than tokens at position 100. The dot product between a rotated Query and a rotated Key naturally encodes how far apart they are.

The beautiful thing about RoPE: it's relative, not absolute. The model doesn't learn "position 5 means X" — it learns "a distance of 3 positions means Y." This generalises better and works at any sequence length.

### The MLP: Where Knowledge Is Stored

After attention, each token's representation passes through a feed-forward network (MLP). This is a two-layer neural network:

1. **Expand**: project from 512 dimensions to 1536 (3× expansion — one of our key improvements)
2. **Nonlinearity**: apply relu² (ReLU then square the result)
3. **Compress**: project back from 1536 to 512 dimensions

The MLP is where the model stores factual knowledge. Research has shown that individual neurons in the MLP layer activate for specific concepts — there are neurons that fire for "years", neurons for "names of countries", neurons for "words that come after 'the'". The expanded dimension (1536) gives the model more "storage slots" for this kind of knowledge.

**relu²** is an interesting choice. Standard ReLU just zeroes out negative values. relu² zeros them out *and* amplifies the contrast between weakly positive and strongly positive activations. This creates sparser activations (more zeros), which helps with compression and makes the model's internal representations more interpretable.

### The U-Net Skip Connections

Our model borrows an idea from image processing: U-Net skip connections. The layers are split into an "encoder" half and a "decoder" half. The encoder processes the input with increasing abstraction. The decoder produces the output. Skip connections link encoder layers to decoder layers, allowing low-level features to flow directly to the output without being distorted by deep processing.

In our 10-layer model: layers 0-4 are the encoder, layers 5-9 are the decoder. Layer 0 connects to layer 9, layer 1 to layer 8, and so on. Each skip connection has a learned weight that controls how much of the encoder signal to blend in.

This helps with a fundamental problem in deep networks: as you stack more layers, the signal from early layers gets diluted. Skip connections provide shortcuts that preserve early information.

### RMS Normalisation

Before each attention and MLP sub-layer, the input is normalised using Root Mean Square Normalisation. This divides each vector by the square root of the mean of its squared elements.

Why? Neural networks are numerically sensitive. If activations grow too large or too small, gradients (the learning signal) become unstable. RMS normalisation keeps everything in a controlled range without shifting the mean — it preserves the direction of each vector while standardising its magnitude.

### The Residual Stream

Every sub-layer (attention and MLP) adds its output to the input rather than replacing it:

```
output = input + attention(normalise(input))
output = output + mlp(normalise(output))
```

This is the **residual stream** — the running highway of information that flows through the entire model. Each layer reads from this stream, does its processing, and writes a small update back. If a layer has nothing useful to contribute (common early in training), it can simply output near-zero and the stream passes through unchanged.

This is why output projections are initialised to zero: at the start of training, the model is an identity function. Every layer contributes nothing. Then, gradually, each layer learns to make useful contributions. This is much more stable than starting with random contributions.

---

## What We Built: The SOTA Stack

### SmearGate: Teaching the Model About Its Neighbours

Standard transformers process each token's embedding independently before the first attention layer. SmearGate changes this by blending each token's embedding with the previous token's embedding:

```
gate = sigmoid(learned_per_dimension_weights)
output = (1 - gate) * current_token + gate * previous_token
```

The gate starts at 0.5 (sigmoid of zero) — an equal blend. During training, each of the 512 dimensions learns whether looking backward helps. Some dimensions might learn gate values near 1.0 (mostly use the previous token's information), others near 0.0 (keep your own information).

Why this works: language is fundamentally sequential. The meaning of "New" depends enormously on whether it's followed by "York" or "car". SmearGate gives the model this sequential context *before* the expensive attention mechanism. It's like giving someone a tiny preview of context before asking them to process a sentence.

### BigramHash: Explicit Pair Features

BigramHash goes further than SmearGate. Instead of just blending embeddings, it explicitly hashes each (previous_token, current_token) pair into a learned lookup table.

The hash function: `XOR(36313 × current, 27191 × previous) % 4096`

Those numbers (36313, 27191) are coprime constants chosen to spread hash values uniformly. The XOR operation mixes the bits of both token IDs. The result is an index into a table of 4096 learned 128-dimensional embeddings.

This is conceptually simple but powerful: the model gets a dedicated representation for common token pairs. "th" + "e" → bucket 2847 → a learned vector that means "the common word 'the'". Different pairs can collide in the same bucket (hash collision), but with 4096 buckets and a 1024-token vocabulary, the collision rate is manageable.

The embedding starts at all zeros and scales up via a learned factor (starting at 0.05). This prevents the bigram signal from overwhelming the token embeddings early in training.

### Orthogonal Initialisation: The Perfect Starting Point

At the moment of birth, before any training, our model's weight matrices are set using QR decomposition of random matrices. This produces **orthogonal** matrices — matrices whose singular values are all exactly 1.

Why does this matter? When you multiply a vector by an orthogonal matrix, the vector's length is perfectly preserved. No shrinking, no growing. In a neural network, this means the gradient signal flows through each layer without being amplified or attenuated. The model can actually learn from the first step, rather than spending hundreds of steps just stabilising its internal dynamics.

Compare this to random Gaussian initialisation (the default): the signal going through 10 layers either explodes exponentially or vanishes to zero, depending on the random seed. Orthogonal init guarantees perfectly balanced signal flow.

### Muon Optimiser: The Gradient Surgeon

Standard optimisers (SGD, Adam) take the raw gradient — the direction of steepest descent — and step in that direction. Muon does something more sophisticated.

For each weight matrix, Muon:
1. Accumulates momentum (a running average of recent gradients)
2. Orthogonalises the result using the Newton-Schulz iteration
3. Applies the orthogonalised update

The Newton-Schulz step is the key innovation. It finds the nearest orthogonal matrix to the gradient — meaning it preserves the direction of the update while making all its singular values equal to 1. This prevents any single direction in weight space from receiving a disproportionately large update.

In plain terms: it's like a surgeon who can see the tumour (gradient) but uses precise, measured cuts (orthogonalised updates) rather than just hacking in the direction of the tumour.

### Weight Decay: Controlled Forgetting

Every step, before the gradient update, we shrink all Muon-optimised weights by a factor of `(1 - 0.04 × learning_rate)`. This is **decoupled weight decay**.

Two purposes:
1. **Regularisation**: prevents the model from memorising the training data. By constantly pulling weights toward zero, the model must continually justify every non-zero weight. Only truly useful connections survive.
2. **Compression**: keeps weight magnitudes small and tightly distributed around zero. When we later quantise the model to int8 (mapping the continuous weight values to 256 discrete levels), a tight distribution means less quantisation error. The weight decay is literally shaping the weights to be more compressible.

### Stochastic Weight Averaging: The Wisdom of the Crowd

During the last 40% of warmdown (when the learning rate is decaying toward zero), we snapshot the model every 50 steps. At the end of training, we average all these snapshots.

This works because the model, during late training, is orbiting around a good solution. Each snapshot is a slightly different version of a good model. Some are a bit too far in one direction, others in another. The average sits in the centre of this orbit — closer to the true optimal solution than any individual snapshot.

It's the same principle as surveying 20 people about the weight of a cow. Each person's guess has noise, but the average of all guesses is remarkably accurate (the "wisdom of crowds" effect).

### Sliding Window Evaluation: Maximum Context, Maximum Score

This is the most elegant trick in the entire stack — it improves your score without changing the model at all.

Standard evaluation chops the validation text into non-overlapping 2048-token chunks. Problem: the first token in each chunk has zero context. The model sees "The" and has to predict the next word knowing nothing about what came before. It's like opening a book to a random page and being asked to predict the next sentence.

Sliding window eval fixes this. We advance the window by only 64 tokens at a time, but score only those last 64 tokens. Each scored token has 1984 tokens of context behind it — nearly the full window. Every token in the validation set gets scored exactly once, but with near-maximum context.

The BPB improvement is about 0.03 — entirely free, no retraining needed. It's not cheating; it's measuring the model's actual capability more accurately. The model was always this good — we were just measuring it badly before.

---

## The Journey So Far

### Experiment 1: Baseline (BPB 2.4087)
9 layers, 2× MLP, 1024 seq, 200 iterations, Muon 0.95, no advanced techniques. The model learned the absolute basics of English in 200 steps on 1 shard of data.

### Experiment 2: Wider + Deeper + Tuned (BPB 1.9364)
10 layers, 3× MLP, 2048 seq, 500 iterations, Muon 0.99, matrix LR 0.02. Each change stacked:
- **3× MLP**: more knowledge storage per layer
- **10 layers**: deeper reasoning
- **2048 seq**: more context per prediction
- **Muon 0.99**: smoother optimisation
- **Lower matrix LR**: more precise weight updates

### Experiment 3: Full SOTA Stack (BPB 1.6893)
Added SmearGate, BigramHash, orthogonal init, weight decay, warmdown 3000, grad clip 0.3. Plus 1000 iterations on multiple shards.

### Current Run: Everything + SWA + Sliding Window + 20 Shards
This run has every technique we know about, training on 2 billion tokens of fresh data. When it finishes, the sliding window eval and SWA will give it an additional ~0.04 BPB boost on top of whatever the training loss achieves.

---

## Where We're Going

The gap between our best (1.6893) and the global SOTA (1.1428) is 0.546 BPB. Here's what closes it:

1. **More iterations**: Our 1000 steps vs the competition's 20,000. This is the single biggest factor. The M4 Max can do ~25k tokens/sec; H100s do 500k+. We compensate with patience.

2. **Quantisation-aware training (QAT)**: Instead of training in float32 and quantising afterward (losing precision), QAT simulates quantisation *during* training so the model learns weights that quantise cleanly. The leaders use int5 for MLP weights and int6 for attention, saving bytes that fund additional parameters.

3. **Bigger BigramHash**: Moving from 4096 to 10240 buckets reduces hash collisions.

4. **Custom tokenisers**: The 1024-token vocabulary is a starting point. A better vocabulary (one that captures common byte sequences more efficiently) could reduce the token count per document, meaning each model forward pass covers more text.

5. **Cloud deployment**: For a real competition score, we'd deploy to 8×H100 GPUs via RunPod, run the full 20,000 iterations in 10 minutes, and submit. Everything we've built locally translates directly — same code, same techniques, just faster hardware.

---

## The Deeper Lesson

What we're really doing here is exploring the fundamental trade-off at the heart of intelligence: **how much can you know given a fixed amount of space to remember it in?**

A 16MB model can't memorise the internet. It must compress — must find patterns, must generalise, must build abstractions. Every parameter must earn its keep. The competition rewards the teams that find the most efficient representations of language — the deepest compressions of human knowledge into the smallest space.

This is, in miniature, the same challenge that every intelligence faces. Your brain compresses a lifetime of experience into ~2.5 petabytes of synaptic connections. A language model compresses internet-scale text into billions of floating-point numbers. Parameter Golf compresses it into 16 million bytes.

The techniques we're using — attention, skip connections, weight averaging, sliding window evaluation — aren't just engineering tricks. They're insights about the structure of language and the nature of prediction itself. Each one represents a discovery about how information flows, how patterns compress, and how intelligence can be made more efficient.

Cape Town's contribution to this competition isn't just a score on a leaderboard. It's a demonstration that with the right tools, the right understanding, and the right community, frontier AI research can happen anywhere.
