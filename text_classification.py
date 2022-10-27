from datasets import list_datasets

all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currently available on the Hub")
print(f"The first 10 are: {all_datasets[:10]}")

from datasets import load_dataset

emotions = load_dataset("emotion")
emotions

# Get the training dataset
# Datasets is based upon apache arrows
train_ds = emotions["train"]
train_ds
len(train_ds)
# Information about the columns
train_ds.column_names
print(train_ds.features)
# Indexing & Sliced indexing
train_ds[0]
print(train_ds[:5])
# Using labels for column filtering
print(train_ds["text"][:5])

# From dataset to pandas
import pandas as pd

emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()

#  We can unencode the labels
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)
df.head()

# Visualize the class distribution
import matplotlib.pyplot as plt

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# DistillBert allows 512 tokens. Check for longer texts
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot(
    "Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black"
)
plt.suptitle("")
plt.xlabel("")
plt.show()

# Convert back to a dataset
emotions.reset_format()

# Text needs to be tokenized and encoded to work with transformers

### Character tokenization
# We can get tokens with python list
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)
## We need to numericalize the characters
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

# Apply the indexer
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

# For one hot encoding we can get help from pytorch
import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
one_hot_encodings.shape

print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")

###  Word tokenization
# We can get words by splitting a text from the spaces
tokenized_text = text.split()
print(tokenized_text)

# Usually the vocabulary gets narrowed down from the corpus
# Words that are not part of the vocabulary will be mapped as UNK

### Subword Tokenization
# We want to split rare words into smaller units to allow
# the model to deal with complex words and misspellings
# We want to keep frequent words as unique entities
# so that we can keep the length of our inputs to a manageable size

from transformers import AutoTokenizer

#  Subword tokenizer WordPiece is used for distilbert
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Apply the tokenizer to text to get encodings
encoded_text = tokenizer(text)
print(encoded_text)

# Get the tokens back from the encoding
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

# CLS and SEP mark the start and end of the sequence
# Tokens are lowercased
# The ## prefix in ##izing and ##p means that
# the preceding string is not whitespace
print(tokenizer.convert_tokens_to_string(tokens))

# We can access details about the specific tokenizer
tokenizer.vocab_size
tokenizer.model_max_length
# Name of the fields model expects on its forward pass
tokenizer.model_input_names

# This function applies the tokenizer to batches of text
#  padding=True will pad the examples with zeros to the size of the longest one in a batch
# truncation=True will truncate the examples to the model’s maximum context size
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# We can see padded endings are all 0
# Attention mask array allows the model to ignore the paddings
print(tokenize(emotions["train"][:2]))

# Apply the tokenize function do DatasetDict via map
# Setting batched true converts the data to batches before applying
# batch_size=None means tokenize will be applied on the full dataset as a single batch
# This ensures that the input tensors and attention masks have the same shape globally
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# input_ids and attention mask columns are added to our dataset
print(emotions_encoded["train"].column_names)
# print(emotions["train"].column_names)

#  Automodel contains pretrained models
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# return tensors pt will return tensors for pytorch
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")

#  We can apply the loaded model to tokenized inputs
inputs = {k: v.to(device) for k, v in inputs.items()}
#  Forward propagation for getting results doesn't need to track gradients
# More memory efficient this way
with torch.no_grad():
    outputs = model(**inputs)
    print(outputs)

# Our outputs are instance of a BaseModelOutput
# We can access its attributes by name
# Current model returns only last hidden state

outputs.last_hidden_state.size()

# We can see that 768 dimensional hidden size vectors are returned
# For each of the 6 input tokens

# For classification tasks it is common practice to use the
# hidden state associated with the [CLS] token as the input feature
# this token appears at the start of each sequence

outputs.last_hidden_state[:, 0].size()

# Helper function getting last hidden state of all the data
# The map() method requires the processing function to return Python or NumPy
# objects when we’re using batched inputs
def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }  # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


#  Our model expects torch tensors as input so reconvert here
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# We did not set batch size=None so default batch size of 1000 is used
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

print(emotions_hidden["train"].column_names)

# hidden states will be input to our classifier
import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
X_train.shape, X_valid.shape

#  Visualize the data
# pip install umap-learn
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# UMAP algorithm works better on features are scaled to lie on [0,1]
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine").fit(
    X_scaled
)  # Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()


fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names
for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(
        df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,)
    )
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])
plt.tight_layout()
plt.show()

# We can train a classifier on the hidden states
from sklearn.linear_model import LogisticRegression

# We increase `max_iter` to guarantee convergence
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)

#  Compare the model score with the dummy model
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)

# AutoModelForSequenceClassification gets models with classification head
# on top of the pretrained model outputs
from transformers import AutoModelForSequenceClassification

#  6 labels for 6 emotions
num_labels = 6
label2id = {
    emotions["train"].features["label"].int2str(0): 0,
    emotions["train"].features["label"].int2str(1): 1,
    emotions["train"].features["label"].int2str(2): 2,
    emotions["train"].features["label"].int2str(3): 3,
    emotions["train"].features["label"].int2str(4): 4,
    emotions["train"].features["label"].int2str(5): 5,
}
id2label = {
    0: emotions["train"].features["label"].int2str(0),
    1: emotions["train"].features["label"].int2str(1),
    2: emotions["train"].features["label"].int2str(2),
    3: emotions["train"].features["label"].int2str(3),
    4: emotions["train"].features["label"].int2str(4),
    5: emotions["train"].features["label"].int2str(5),
}
# We get a warning that some ome parts of the model are randomly initialized.
# This is normal since the classification head has not yet been trained.
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
).to(device)

from sklearn.metrics import accuracy_score, f1_score

# Function for computing metrics of the model
# pred is a named tuple
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# Trainer and Trainig arguments for custom training
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
# Model name will be used as the training output directory
# Push to hub will result in pushing to huggingsface hub
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True,
    log_level="error",
)


# Custom train the model from the checkpoint
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer,
)
trainer.train()

# trainer.save_model("Model")
# # Continue from uploaded or saved model
# model_ckpt = "uck/distilbert-base-uncased-finetuned-emotion"
# model_ckpt = "distilbert-base-uncased-finetuned-emotion"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_ckpt, num_labels=num_labels,id2label=id2label,label2id=label2id
# ).to(device)


# We can see improved metrics and precictions
preds_output = trainer.predict(emotions_encoded["validation"])
preds_output.metrics

y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)

## Cross entropy will apply softmax inside and expects logits
from torch.nn.functional import cross_entropy


def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model inputs = {k:v.to(device) for k,v in batch.items()
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}


# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16
)

emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = df_test["predicted_label"].apply(label_int2str)

df_test.sort_values("loss", ascending=False).head(10)
df_test.sort_values("loss", ascending=True).head(10)


trainer.push_to_hub(commit_message="Training completed!")
