import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd

s3_path = 's3://sagemaker-us-west-2-218892669261/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df[['TITLE', 'CATEGORY']]

my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}

def update_cat(x):
    return my_dict[x]

df['CATEGORY'] = df['CATEGORY'].apply(lambda x : update_cat(x))

print(df)

# This is a tip: sometimes when you spend a day training a model, but sagemaker failed at a certain point.
# It waste lots of money but with a failed job. What we could do here is to firstly train a tiny model job.
# The reason is just to verify that the setup / E2E works well, then we can start training large model.

# frac=0.05 - Means “select a random fraction of the rows” — here 5% of the entire DataFrame.
# random_state=1 - Ensures reproducibility: you’ll get the same 5% subset every time you run the code.
df = df.sample(frac=0.05, random_state=1)
# Resets the row index after sampling.
# drop=True discards the old index instead of keeping it as a new column.
df = df.reset_index(drop=True)

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x : encode_cat(x))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split()) # title.trim()?

        inputs = self.tokenizer.encode_plus(
            title,
            None, # what is this?
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            return_token_type_ids = True, # Separating different sentence's tokens
            return_attention_mask = True, # Identify which tokens are real vs padding
            truncation = True # cutoff
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        '''
        example inputs:
        {
         'input_ids':      tensor([[  101,  1045, 2293,  3899,   102,  1045, 2293,  4937,   102,     0]]),
         'token_type_ids': tensor([[   0,    0,    0,    0,    0,    1,    1,    1,    1,     0]]),
         'attention_mask': tensor([[   1,    1,    1,    1,    1,    1,    1,    1,    1,     0]])
        }
        '''

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'target': torch.tensor(self.data.ENCODE_CAT[index],dtype=torch.long)
        }
        
    def __len__(self):
        return self.len


train_size = 0.8
train_dataset = df.sample(frac=train_size,random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_dataset.reset_index(drop=True)

# print("Full dataset: {}".format(df.shape))
# These logs will be showed in CloudWatch later
print(f"Full dataset {df.shape}")
print(f"Train dataset: {train_dataset.shape}")
print(f"Test dataset: {test_dataset.shape}")

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

test_parameters = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

training_loader = DataLoader(training_set,**train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)

'''
for batch in training_loader:
    print(batch['ids'].shape)
    print(batch['mask'].shape)
    print(batch['target'].shape)
    break

torch.Size([4, 512])   # 4 samples, each 512 tokens long (MAX_LEN)
torch.Size([4, 512])   # 4 attention masks
torch.Size([4])        # 4 labels

# Single iteration breakdown:
batch = next(iter(training_loader))

# batch contains:
{
    'ids': tensor([[101, 2023, 2003, ...],     # Article 1 tokens
                   [101, 2028, 2062, ...],     # Article 2 tokens  
                   [101, 3899, 2024, ...],     # Article 3 tokens
                   [101, 2061, 2003, ...]]),   # Article 4 tokens (shape: [4, 512])
    
    'mask': tensor([[1, 1, 1, ..., 0, 0],      # Article 1 attention mask
                    [1, 1, 1, ..., 1, 0],      # Article 2 attention mask
                    [1, 1, 1, ..., 0, 0],      # Article 3 attention mask  
                    [1, 1, 1, ..., 1, 1]]),    # Article 4 attention mask (shape: [4, 512])
    
    'target': tensor([0, 2, 1, 3])             # Labels: [Entertainment, Science, Business, Health]
}
'''

class DistilBERTClass(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Downloads a pretrained DistilBERT encoder (6 transformer layers, hidden size 768).
        # Base model (66M parameters).
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')

        '''
        A fully connected layer (Linear transformation).
        Input = 768-dim hidden vector (DistilBERT’s output size).
        Output = 768-dim vector (same size, but transformed).
        Think of this as a small extra feature transformation before classification.
        '''
        self.pre_classifier = torch.nn.Linear(768, 768)

        '''
        Randomly sets 30% of inputs to zero during training.
        Prevents overfitting (forces the model to not rely on specific neurons too much).
        During inference, dropout is disabled.
        '''
        self.dropout = torch.nn.Dropout(0.3)

        # Input 768, output 4 -> 4 class classification
        self.classifier = torch.nn.Linear(768, 4)

    # Text → Tokenizer → DistilBERT → [CLS] → Linear → ReLU → Dropout → Linear → Logits → Softmax
    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        # last hidden state (last vector right after the last transformer layer and right before the classification layer)
        hidden_state = output_1[0]

        # Pick [CLS]
        pooler = hidden_state[:, 0]

        # It’s not a transformer block at all. It’s just a linear (dense) layer that you added after DistilBERT.
        # First, remix features linearly.
        pooler = self.pre_classifier(pooler)

        # If a number is positive, keep it. If negetive, make it 0.
        # Then, push through ReLU to introduce nonlinearity.
        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        # Logits output
        '''
        # outputs shape: [4, 4]
        # Each row is one example's logits for 4 classes
        outputs = [
            [-0.2,  1.5, -0.8,  0.3],  # Example 1's logits
            [ 0.7, -1.2,  2.1, -0.5],  # Example 2's logits  
            [-1.1,  0.4, -0.3,  1.8],  # Example 3's logits
            [ 1.2, -0.7,  0.9, -1.4]   # Example 4's logits
        ]
        '''
        output = self.classifier(pooler)

        return output


def calculate_accu(big_idx, targets):
    # Example:
    # print(big_idx == targets) -> tensor([True, False, False, True])
    # Print(big_idx == targets).sum() -> tensor(2)
    # Print(big_idx == targets).sum().item() -> 2
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def train(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for idx, data in enumerate(training_loader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask) # By default same as model.forward(ids, mask)

        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0) # batch size, 4 here in this example.

        if idx % 5000 == 0: # 10k / 4 = 2,500
            ave_loss_step = tr_loss / nb_tr_steps
            ave_accu_step = n_correct / nb_tr_examples * 100
            print(f"Training loss per 5000 steps: {ave_loss_step}")
            print(f"Training Accuracy per 5000 steps: {ave_accu_step}")

        # Fine-tuning logics here!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps # 0.13
    epoch_accu = n_correct / nb_tr_examples * 100 # 93%
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training accuracy Epoch: {epoch_accu}")

    return

def valid(epoch, model, testing_loader, device, loss_function):
    model.eval()

    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    with torch.no_grad():

        for idx, data in enumerate(testing_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask).squeeze()

            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if idx % 1000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = n_correct / nb_tr_examples * 100
                print(f"Validation loss per 1000 steps: {loss_step}")
                print(f"Validation accuracy per 1000 steps: {accu_step}")

            epoch_loss = tr_loss / nb_tr_steps
            epoch_accu = n_correct / nb_tr_examples * 100
            print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
            print(f"Validation accuracy epoch: {epoch_accu} at epoch {epoch}")

    return epoch_accu

def main():
    print("start")

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)

    args = parser.parse_args()

    args.epochs
    args.train_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = DistilBERTClass()
    model.to(device)

    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

    loss_function = torch.nn.CrossEntropy

    # Train loop
    EPOCHS = 4

    for epoch in range(EPOCHS):
        train(epoch, model, device, training_loader, optimizer, loss_function)

        valid(epoch, model, testing_loader, device, loss_function)

    output_dir = os.environ['SM_MODEL_DIR']

    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')

    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')

    torch.save(model.state_dict(), output_model_file)

    tokenizer.save_vocabulary(output_vocab_file)

if __name__ == '__main__':
    main()











