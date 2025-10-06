import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from physics_dataset import PhysicsQADataset
from MiniGPT import MiniGPT  # This must contain your MiniGPT class

# ✅ Config
EPOCHS = 15
BATCH_SIZE = 2
MAX_LEN = 80 
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load tokenizer and dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ✅ Use EOS token as pad token
tokenizer.pad_token = tokenizer.eos_token

dataset = PhysicsQADataset("physics_qa.jsonl", tokenizer, max_len=MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ Model setup
model = MiniGPT(vocab_size=tokenizer.vocab_size, d_model=256, nhead=8, num_layers=4).to(DEVICE)

# ✅ Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ✅ Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i, (input_ids, target_ids) in enumerate(loader):
        input_ids = input_ids.to(DEVICE)         # (B, T)
        target_ids = target_ids.to(DEVICE)       # (B, T)

        optimizer.zero_grad()
        outputs = model(input_ids)               # (B, T, V)

        # Reshape for loss: (B*T, V) vs (B*T)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 10 == 0 or i == len(loader) - 1:
            print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} average loss: {total_loss/len(loader):.4f}")

# ✅ Save trained model
torch.save(model.state_dict(), "minigpt_physics.pt")
print("\n✅ Training complete! Model saved as 'minigpt_physics.pt'")

