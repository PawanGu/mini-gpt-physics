import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=128):
        super(MiniGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt):
        x = self.embedding(tgt) + self.pos_embedding[:, :tgt.size(1), :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        return self.fc_out(out)

def generate_response(model, tokenizer, input_text, max_new_tokens=20):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token_id), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    vocab_size = tokenizer.vocab_size
    model = MiniGPT(vocab_size=vocab_size)

    print("Enter a physics question (or type 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        output = generate_response(model, tokenizer, user_input)
        print("\nGenerated (raw prediction):\n", output)
