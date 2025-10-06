import json
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class PhysicsQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=64):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Support both ("prompt", "response") and ("question", "answer")
                prompt = entry.get("prompt") or entry.get("question")
                response = entry.get("response") or entry.get("answer")

                self.data.append((prompt, response))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, response = self.data[idx]

        # Tokenize prompt (input) and response (target)
        enc_input = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        enc_target = self.tokenizer(
            response,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc_input.input_ids.squeeze(0)
        target_ids = enc_target.input_ids.squeeze(0)

        return input_ids, target_ids
