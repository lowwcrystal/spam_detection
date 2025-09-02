from transformers import BartTokenizer, BartModel
import torch
from tqdm import tqdm

def convert_to_embeddings(messages, batch_size=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_model = BartModel.from_pretrained("facebook/bart-base").to(device)
    bart_model.eval()
    
    embeddings_list = []
    
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i+batch_size]
        
        out = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        out = {k: v.to(device) for k, v in out.items()}
        
        with torch.no_grad():
            output = bart_model(**out)
            embeddings = output.last_hidden_state.mean(dim=1)
            embeddings_list.append(embeddings.cpu())  

    return torch.cat(embeddings_list, dim=0)
