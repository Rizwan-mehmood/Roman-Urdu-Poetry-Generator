import os

os.environ["STREAMLIT_SERVER_WATCHED_FILES"] = ""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Model Architecture (must match your training setup)
# ---------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
        )


# ---------------------------
# Postprocessing (for structure tokens)
# ---------------------------
def postprocess_generated_text(text):
    text = text.replace(" <stanza> ", "\n\n")
    text = text.replace(" <line> ", "\n")
    text = text.replace("<poem> ", "")
    text = text.replace(" </poem>", "")
    return text


# ---------------------------
# Generation Function
# ---------------------------
def generate_text(
    model,
    token_to_idx,
    idx_to_token,
    prompt,
    generation_length,
    temperature=1.0,
    device="cpu",
):
    # Use default seed if the prompt is empty or only whitespace.
    if not prompt.strip():
        prompt = "ishq ka junoon hai"

    model.eval()
    tokens = prompt.split()
    # Convert tokens to indices (default to 0 if token is missing)
    input_ids = [token_to_idx.get(token, 0) for token in tokens]

    # Fallback in case tokenization results in an empty list
    if len(input_ids) == 0:
        input_ids = [0]
        tokens = [idx_to_token[0]]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    hidden = model.init_hidden(1, device)
    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(generation_length):
            logits, hidden = model(input_tensor, hidden)
            # Use only the last timestep's logits, adjust with temperature
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            next_token = idx_to_token[next_token_id]
            generated.append(next_token)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

    output_text = " ".join(generated)
    return postprocess_generated_text(output_text)


# ---------------------------
# Load the Saved Checkpoint (no training needed)
# ---------------------------
@st.cache_resource
def load_model_checkpoint():
    checkpoint_path = "poetry_model/lstm_poetry_model.pt"  # Update path if needed
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Retrieve token mappings from the checkpoint.
    token_to_idx = checkpoint["token_to_idx"]
    idx_to_token = checkpoint["idx_to_token"]
    vocab_size = len(token_to_idx)

    # Use the same parameters that were used during training.
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.2

    model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, token_to_idx, idx_to_token


# Determine device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, token_to_idx, idx_to_token = load_model_checkpoint()
model.to(device)


# ---------------------------
# Streamlit User Interface
# ---------------------------
st.title("Roman Urdu Poetry Generator (PyTorch)")
st.markdown("Generate beautiful poetry using your saved model!")

# User input: seed text for generation
seed_text = st.text_input("Enter seed text:", placeholder="ishq ka junoon hai")

# Slider to control the length of the generated output (in tokens)
generation_length = st.slider(
    "Generation Length (number of tokens):",
    min_value=20,
    max_value=200,
    value=100,
    step=10,
)

# Slider to adjust temperature (controls creativity/randomness)
temperature = st.slider(
    "Temperature (creativity):", min_value=0.1, max_value=2.0, value=1.0, step=0.1
)

if st.button("Generate Poetry"):
    st.info("Generating poetry, please wait...")
    poetry = generate_text(
        model,
        token_to_idx,
        idx_to_token,
        seed_text,
        generation_length,
        temperature,
        device=device,
    )
    st.text_area("Generated Poetry", value=poetry, height=400)
