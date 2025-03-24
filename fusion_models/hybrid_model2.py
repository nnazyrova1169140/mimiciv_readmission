import pandas as pd
import numpy as np
from tqdm import tqdm

#################################
# 1. Temporal Data
#################################
#----------------------------------
# 1.1 Load Four Temporal Datasets
#-----------------------------------
blood_chemistry    = pd.read_csv("blood_chemistry.csv")       # columns: hadm_id, charttime, ~12 numeric features
complete_blood_cnt = pd.read_csv("complete_blood_count.csv")  # columns: hadm_id, charttime, ~9  numeric features
blood_differential = pd.read_csv("blood_differential.csv")    # columns: hadm_id, charttime, ~16 numeric features
vital_signs        = pd.read_csv("vital_signs.csv")           # columns: hadm_id, charttime, ~11 numeric features

#-------------------------------------
# 1.2 Outer Join on (hadm_id, charttime) for Early Fusion
#--------------------------------------
merged_df = pd.merge(blood_chemistry,    complete_blood_cnt, on=["hadm_id","charttime"], how="outer")
merged_df = pd.merge(merged_df, blood_differential,          on=["hadm_id","charttime"], how="outer")
merged_df = pd.merge(merged_df, vital_signs,                 on=["hadm_id","charttime"], how="outer")

#-------------------------------------
# 1.3 Append Structured Features by Broadcasting Across Timesteps
#--------------------------------------
structured_df = pd.read_csv("structured_data.csv")  # e.g., hadm_id + 10 static features

# structured featred are merged into merge_df by hadm_id only and broadcasted across all charttime rows for that hadm_id only. 

merged_df = pd.merge(merged_df, structured_df, on="hadm_id", how="left")

#--------------------------------------
# 1.4 Forward Fill Missing Lab/Vital Values
#--------------------------------------

merged_df = merged_df.sort_values(["hadm_id","charttime"])
all_cols = list(merged_df.columns)
exclude  = ["hadm_id", "charttime"]
value_cols = [c for c in all_cols if c not in exclude]

# Forward-fill per hadm_id only on labs/vitals; 
merged_df[value_cols] = (
    merged_df.groupby("hadm_id")[value_cols]
             .apply(lambda x: x.ffill())
             .reset_index(level=0, drop=True)
)

#--------------------------------------
# 1.5 Group by hadm_id, Sort by charttime, Pad to T=25
#--------------------------------------
T = 25
grouped_sequences = []
hadm_ids = []

for hid, grp in tqdm(merged_df.groupby("hadm_id")):
    grp_sorted = grp.sort_values("charttime")
    # drop hadm_id/charttime now that the data is sorted
    # the rest are numeric columns (labs, vitals, plus broadcasted structured features)
    X = grp_sorted[value_cols].values  # shape: (timesteps_i, F_early)
    grouped_sequences.append(X)
    hadm_ids.append(hid)

# Zero-pad to (T, F_early)
F_early = len(value_cols)
padded_sequences = []
for seq in grouped_sequences:
    length = seq.shape[0]
    if length >= T:
        padded_seq = seq[:T]
    else:
        padded_seq = np.vstack([seq, np.zeros((T-length, F_early))])
    padded_sequences.append(padded_seq)

padded_sequences = np.array(padded_sequences)  # shape => (N, T, F_early)

############################################
# 2. Structured Data
##############################################
import torch
from transformers import AutoTokenizer, AutoModel

#--------------------------------------
# 2.1 Load Discharge Summaries
#--------------------------------------
discharge_summaries = pd.read_csv("discharge_summaries.csv")
# columns: hadm_id, text

discharge_summaries = discharge_summaries.groupby("hadm_id")["text"].apply(lambda x: " ".join(str(t) for t in x)).reset_index()

#---------------------------------------------
# 2.2 Compute or Load Pre-Computed Embeddings
#---------------------------------------------
model_name = "nazyrova/clinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
clinical_bert = AutoModel.from_pretrained(model_name).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clinical_bert.to(device)

embedding_list = []
for idx, row in tqdm(discharge_summaries.iterrows(), total=len(discharge_summaries)):
    hid = row["hadm_id"]
    text= row["text"] if isinstance(row["text"], str) else ""

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        outputs = clinical_bert(**encoded)
        last_hidden_state = outputs.last_hidden_state  # (batch=1, seq_len, 768)
        # CLS token embedding
        cls_emb = last_hidden_state[:, 0, :]  # shape: (1, 768)
        cls_emb = cls_emb.squeeze().cpu().numpy()  # (768,)

    embedding_list.append([hid] + cls_emb.tolist())

cols = ["hadm_id"] + [f"emb_{i}" for i in range(768)]
clinicalbert_df = pd.DataFrame(embedding_list, columns=cols)

clinicalbert_df.to_csv("clinicalbert_embeddings.csv", index=False)

# 2.3 Align with hadm_ids
clinicalbert_df = clinicalbert_df.set_index("hadm_id").reindex(hadm_ids)
textual_data_array = clinicalbert_df.values  # shape => (N, 768)

##############################################
# 3. Labels & Train/Test split
##############################################


labels_df = pd.read_csv("readmission_labels.csv").set_index("hadm_id")
labels_df = labels_df.reindex(hadm_ids)
y = labels_df["label"].values  # shape (N,)

train_ids = pd.read_csv("train_ids.csv")["hadm_id"].unique()
test_ids  = pd.read_csv("test_ids.csv")["hadm_id"].unique()

# hadm_id -> index in padded_sequences
hadm_id_to_index = {hid: i for i, hid in enumerate(hadm_ids)}

train_idx = [hadm_id_to_index[h] for h in train_ids if h in hadm_id_to_index]
test_idx  = [hadm_id_to_index[h] for h in test_ids  if h in hadm_id_to_index]

X_early_train = padded_sequences[train_idx]      # shape => (N_train, T, F_early)
X_text_train  = textual_data_array[train_idx]    # shape => (N_train, 768)
y_train       = y[train_idx]

X_early_test = padded_sequences[test_idx]
X_text_test  = textual_data_array[test_idx]
y_test       = y[test_idx]

print("Train shapes:")
print("  Early-Fused:", X_early_train.shape)
print("  Textual:", X_text_train.shape)
print("  Label:", y_train.shape)


########################################
# 4. Model Architecture in Keras
############################################
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Optional: Simple Attention for LSTM sequence
# ---------------------------
class SimpleAttention(layers.Layer):
    def __init__(self, units=64):
        super(SimpleAttention, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        # inputs: (batch, timesteps, hidden_dim)
        score = self.V(tf.nn.tanh(self.W(inputs)))  # (batch, timesteps, 1)
        att_w = tf.nn.softmax(score, axis=1)
        context = att_w * inputs
        context = tf.reduce_sum(context, axis=1)   # (batch, hidden_dim)
        return context, att_w

# ---------------------------
# 4.1 Early-Fused Branch (Temporal + Structured)
# ---------------------------
T, F_early = X_early_train.shape[1], X_early_train.shape[2]  # e.g., T=25, F_early = labs+vitals+struct
early_input = Input(shape=(T, F_early), name="early_fused_input")

x = layers.Masking(mask_value=0.0)(early_input)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)

# Apply attention or just use last hidden state
x, weights = SimpleAttention(units=64)(x)  # shape => (batch, 128)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
Z_temp_struct = x  # final numeric embedding from (temporal+structured)

# ---------------------------
# 4.2 Textual Embedding Input
# ---------------------------
E = X_text_train.shape[1]  # e.g., 768
text_input = Input(shape=(E,), name="clinicalbert_input")
z = layers.Dense(64, activation='relu')(text_input)
z = layers.Dropout(0.2)(z)
Z_text = z  # textual embedding

# ---------------------------
# 4.3 Cross-Fusion with Multi-Head Attention
#     We'll treat Z_temp_struct and Z_text as 2 tokens in shape (batch, 2, hidden_dim)
# ---------------------------
# First, unify dimensionality. Suppose we want to use dimension 64 for both
common_dim = 64

numeric_proj = layers.Dense(common_dim, activation=None, name="numeric_projection")(Z_temp_struct)  # (batch, 64)
text_proj    = layers.Dense(common_dim, activation=None, name="text_projection")(Z_text)            # (batch, 64)

# Reshape to (batch, 2, common_dim)
# token 0 => numeric, token 1 => text
fusion_tokens = layers.Concatenate(axis=1)([layers.Reshape((1, common_dim))(numeric_proj),
                                           layers.Reshape((1, common_dim))(text_proj)])

# Apply multi-head self-attention over these 2 tokens
attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32)(fusion_tokens, fusion_tokens)
# Add skip connection + LN
fusion_tokens = layers.Add()([fusion_tokens, attn_out])  # (batch, 2, common_dim)
fusion_tokens = layers.LayerNormalization()(fusion_tokens)

# Flatten or pool
fusion_vector = layers.Flatten()(fusion_tokens)  # shape => (batch, 2*common_dim)=128 if 2 tokens

# ---------------------------
# 4.4 Final Classifier
# ---------------------------
final = layers.Dense(64, activation='relu')(fusion_vector)
final = layers.Dropout(0.2)(final)
final = layers.Dense(32, activation='relu')(final)
final = layers.Dropout(0.2)(final)
output = layers.Dense(1, activation='sigmoid')(final)

model = Model(inputs=[early_input, text_input], outputs=output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['AUC']
)
model.summary()




###########################################
5. Training & Evaluation
###########################################

# =======================
# 5.1 Train
# =======================
history = model.fit(
    x=[X_early_train, X_text_train],
    y=y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', 
            patience=5, 
            mode='max', 
            restore_best_weights=True
        )
    ]
)

# =======================
# 5.2 Test Evaluation
# =======================
test_loss, test_auc = model.evaluate(
    x=[X_early_test, X_text_test],
    y=y_test
)
print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
