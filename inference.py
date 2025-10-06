import torch
from transformers import GPT2Tokenizer
from MiniGPT import MiniGPT  # Your trained model class
import torch.nn.functional as F

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load model
model = MiniGPT(vocab_size=tokenizer.vocab_size, d_model=256, nhead=8, num_layers=4)
model.load_state_dict(torch.load("minigpt_physics.pt", map_location=device))
model.to(device)
model.eval()

# ✅ Simple text generation
def generate_response(prompt, max_new_tokens=40, temperature=0.7, top_k=40):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_values, dim=-1)
            next_token_id = top_k_indices[0, torch.multinomial(probs, num_samples=1)]

            generated = torch.cat((generated, next_token_id.view(1, 1)), dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

# ✅ Terminal loop
print("Ask your physics question (or type 'exit'):")
while True:
    user_input = input(">> ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_response(user_input)
    print("\n🤖 MiniGPT Answer:\n", response)
