# GPT-1

batch_size: Number of sequences processed simultaneously.
block_size: Maximum length of the input sequence (context).
max_iters: Total number of training iterations.
eval_interval: Evaluate the model every 500 steps.
learning_rate: Speed of learning during training.
device: Whether to use GPU (cuda) or CPU.
n_embd: Size of token embeddings (dimensionality of the model).
n_head, n_layer: Number of attention heads and transformer layers.
dropout: Regularization to prevent overfitting.


Reads input.txt, containing the training text.
chars: Unique characters in the text.
stoi / itos: Map characters to integers (and vice versa).
encode/decode: Convert between text and numerical sequences.

Converts the entire dataset into a tensor of integers.
Splits data into training (90%) and validation (10%).

split:

Determines whether the batch is for training ('train') or validation ('val').
Picks data from train_data or val_data accordingly.
Random Sampling:

A random starting index (ix) is chosen from the dataset for the batch.
Example: If block_size=256 and batch_size=2, and we randomly sample ix = [100, 5000], we extract sequences starting from positions 100 and 5000 in the dataset.
x and y Construction:

x is a batch of input sequences of length block_size.
y is the same as x but shifted by one position. It serves as the target for predicting the next token.

Example:
Suppose block_size=5 and data="hello world" encoded as integers:
data = [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]
ix = [0, 5]

x = [[7, 4, 11, 11, 14],  # "hello"
     
     [26, 22, 14, 17, 11]] # " worl"
y = [[4, 11, 11, 14, 26],  # "ello "
     
     [22, 14, 17, 11, 3]]  # "world"


Evaluation Mode:

The model is put in evaluation mode with model.eval(), which disables dropout.
Compute Loss:

For each batch, model(X, Y) returns predictions (logits) and loss. The loss measures how well the model predicts the next token.
Average Loss:

The average loss across multiple batches (eval_iters) is computed for stability.
Training Mode:

Once done, the model is switched back to training mode using model.train().
Example:
Assume:

Training loss = [1.2, 1.1, 1.15, 1.05]
Validation loss = [1.5, 1.45, 1.6, 1.4] The function outputs:

{'train': 1.125, 'val': 1.4875}

Head Class
This represents a single head of self-attention, a crucial building block for Transformers.

Key, Query, Value:

Each token generates:
Key (k): Represents what information this token has.
Query (q): Represents what information this token seeks.
Value (v): Represents the actual information of the token.
Example: For a token "I" in a sentence, q may seek a subject, k may indicate that "I" is a subject, and v holds details about "I."
Attention Scores:

Scores (wei) are computed as the dot product of queries and keys.
Scores are scaled (* k.shape[-1]**-0.5) to stabilize gradients.
Masking:

The tril buffer ensures that attention is limited to past tokens (causal masking).
Weighted Aggregation:

Attention scores (wei) are applied to values (v), emphasizing relevant tokens.

Combines multiple Head instances to capture diverse patterns.
Example:
Each head might focus on a different relationship (e.g., grammatical structure, semantic meaning).

The GPTLanguageModel combines embedding layers, Transformer blocks (Block), and a final linear layer to generate predictions.
The training loop iterates over batches, computes loss, and updates the model using backpropagation.
