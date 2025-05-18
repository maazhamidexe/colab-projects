# Building a Mini-GPT From Scratch 🧠✨

Hey there! Welcome to this notebook where we dive deep into the mechanics of Large Language Models (LLMs) by building our very own simplified GPT-style (decoder-only Transformer) model from the ground up using PyTorch.

**The Goal?** To demystify what goes on under the hood of models like GPT-2, understand the core components, and even get our mini-model to generate some text!

## What's Inside? 🚀

This notebook walks you through implementing, training, and evaluating a character-level language model based on the Transformer architecture. Here's a breakdown of the journey:

1.  **Setting the Stage (Task Overview & Imports):**
    *   We kick things off by outlining the task: analyze a given LLM architecture (a decoder-only Transformer) and implement its simplified version.
    *   Essential libraries like PyTorch, Matplotlib, NLTK, and Transformers are imported.

2.  **📜 Token & Position Embedding (Task 1):**
    *   **What:** Learn how to convert text (characters) into numerical representations (token embeddings) that the model can understand. Then, we add positional information because Transformers don't inherently know the order of tokens.
    *   **Implementation:** `TokenAndPositionEmbedding` class.
    *   **Key Question:** *Add or concatenate positional embeddings?* (Spoiler: We add 'em!)
    *   **Viz:** A sneak peek at token embeddings using PCA/t-SNE.

3.  **👀 Multi-Head Attention (Task 2):**
    *   **What:** The heart of the Transformer! This mechanism allows the model to weigh the importance of different tokens in the input sequence when processing a particular token. "Multi-head" means it does this from different perspectives simultaneously.
    *   **Implementation:** `Head` (single attention head) and `MultiHeadAttention` classes, including causal masking for autoregressive generation.
    *   **Key Question:** *How does the number of heads affect learning?* (Think: diverse perspectives vs. capacity per head).
    *   **Viz:** Heatmaps of attention weights, showing what the model "focuses" on.

4.  **🧠 Feed-Forward Network (Task 3):**
    *   **What:** After attention, each token's representation is processed independently by a simple neural network (two linear layers with an activation function). This adds more computational depth.
    *   **Implementation:** `FeedForward` class (using GELU activation).
    *   **Viz:** Histograms showing how data distributions change before and after the activation function.

5.  **🧱 Transformer Block (Task 4):**
    *   **What:** Combining Multi-Head Attention and the Feed-Forward Network, along with crucial helpers: Layer Normalization (stabilizes training) and Residual Connections (helps with deep networks). This is the repeatable unit of our LLM.
    *   **Implementation:** `TransformerBlock` class (using a Pre-LayerNorm architecture).
    *   **Key Question:** *How do residuals help in deep networks?* (Hint: gradients & identity).

6.  **✍️ Text Generation Function (Task 5):**
    *   **What:** Now for the fun part! A function to make our model generate new text given a starting prompt. We explore different decoding strategies.
    *   **Implementation:** `generate_text` function supporting:
        *   Greedy decoding
        *   Sampling (with temperature and top-k filtering)
        *   A simplified take on Beam Search (defaults to greedy in our setup).
    *   **Viz:** Comparing outputs from different generation methods.

7.  **🧩 Full Model Integration & Training (Task 6):**
    *   **What:** Assembling all the pieces into a complete `GPTModel` class. Then, we train it on a small dataset (Tiny Shakespeare!) to teach it to predict the next character.
    *   **Implementation:** `GPTModel` class and a training loop.
    *   **Viz:** Plot of training and validation loss to see how our model learns (or tries to!).
    *   **Comparison:** Text generation quality before and after training.

8.  **📊 Model Evaluation (Task 7):**
    *   **What:** How good is our model? We implement metrics to find out.
    *   **Implementation:**
        *   **Perplexity:** Measures how "surprised" the model is by new text (lower is better).
        *   **BLEU Score:** Compares model-generated text to reference text (higher is better, though a bit nuanced for char-level generation).
    *   **Output:** A table summarizing these scores for different generation methods.

9.  **🚀 Pre-training, Fine-tuning & Beyond (Task 8):**
    *   **What:** Stepping into the world of pre-trained models.
    *   **Hugging Face GPT-2:**
        *   Load a pre-trained GPT-2 model.
        *   **Quantization:** Briefly explore dynamic quantization to see how model size can be reduced (with a note on CPU vs. GPU compatibility for this technique).
        *   **Zero/Few-Shot "Fine-tuning":** Using prompting to make the pre-trained model perform tasks like QA and NMT *without* further training on specific datasets for those tasks.
    *   **Classification Head:** Discuss and show how to add a classification layer to a GPT-2 model, enabling it for tasks like sentiment analysis.

## How to Use This Notebook 🛠️

1.  **Environment:** Make sure you have Python and the necessary libraries installed (see imports at the beginning). A GPU is recommended for faster training, but the code will run on CPU too.
2.  **Run the Cells:** Execute the cells sequentially. Some cells (especially model training) might take a while.
3.  **Experiment!**
    *   Change hyperparameters (embedding dimension, number of heads/layers, block size, learning rate).
    *   Try a different small dataset for training.
    *   Play with the `generate_text` parameters.

## Key Architectural Choices & Notes:

*   **Decoder-Only Transformer:** Our model architecture mirrors the left-hand diagram provided in the task, focusing on a GPT-style decoder.
*   **Character-Level:** We're training on characters, so the vocabulary is small.
*   **Pre-LayerNorm:** Our Transformer blocks use Layer Normalization *before* the main sub-layer (attention/FFN), a common practice in models like GPT-2.
*   **Simplified:** This is a "mini" version designed for learning. Production LLMs are vastly larger and trained on massive datasets.

---

Happy coding and exploring the fascinating world of LLMs! If you have questions or find cool ways to improve this, feel free to experiment.
