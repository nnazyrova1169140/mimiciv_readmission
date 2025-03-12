import pandas as pd
import numpy as np
from tqdm import tqdm
######################################
# 1. Temporal Data
######################################
# --------------------------------------------------
# Load temporal datasets
# --------------------------------------------------
blood_chemistry    = pd.read_csv('blood_chemistry.csv')
complete_blood_cnt = pd.read_csv('complete_blood_count.csv')
blood_differential = pd.read_csv('blood_differential.csv')
vital_signs        = pd.read_csv('vital_signs.csv')

# --------------------------------------------------
# 1.1 Early Fusion: Merge on (hadm_id, charttime)
#     Keep all rows (outer join)
# --------------------------------------------------
merged_df = pd.merge(blood_chemistry,    complete_blood_cnt, on=['hadm_id','charttime'], how='outer')
merged_df = pd.merge(merged_df, blood_differential, on=['hadm_id','charttime'], how='outer')
merged_df = pd.merge(merged_df, vital_signs,        on=['hadm_id','charttime'], how='outer')

# --------------------------------------------------
# 1.2 Sort and forward-fill missing values by hadm_id
# --------------------------------------------------
merged_df = merged_df.sort_values(by=['hadm_id','charttime'])
feature_cols = [c for c in merged_df.columns if c not in ['hadm_id','charttime']]

# Forward fill within each hadm_id group
merged_df[feature_cols] = (
    merged_df.groupby('hadm_id')[feature_cols]
             .apply(lambda df: df.ffill())
             .reset_index(level=0, drop=True)
)

# --------------------------------------------------
# 1.3 Create sequences per hadm_id (chronologically)
# --------------------------------------------------
grouped_sequences = []
hadm_ids = []
for hadm_id, grp in tqdm(merged_df.groupby('hadm_id')):
    grp_sorted = grp.sort_values('charttime')
    seq = grp_sorted[feature_cols].values
    grouped_sequences.append(seq)
    hadm_ids.append(hadm_id)

# --------------------------------------------------
# 1.4 Zero-pad sequences to fixed length T
# --------------------------------------------------
T = 25                       # max timesteps
F = len(feature_cols)        # total features from the 4 merged datasets
padded_sequences = []

for seq in grouped_sequences:
    if len(seq) >= T:
        padded_seq = seq[:T, :]  # truncate if longer
    else:
        padded_seq = np.vstack([seq, np.zeros((T - len(seq), F))])
    padded_sequences.append(padded_seq)

padded_sequences = np.array(padded_sequences)  # shape: (N, T, F)

###############################
# 2. Structured Data
###############################

structured_df = pd.read_csv('structured_data.csv')  # columns: hadm_id + 113 features
structured_df = structured_df.set_index('hadm_id')
# We'll align it with the same hadm_ids order:
structured_df = structured_df.reindex(hadm_ids)
structured_data_array = structured_df.values

########################################
# 3. Unstrcutred Data
########################################
import torch
from transformers import AutoTokenizer, AutoModel

discharge_summaries = pd.read_csv("discharge_summaries.csv")  


# Initialize the model
tokenizer = AutoTokenizer.from_pretrained("nazyrova/clinicalBERT")
clinical_bert = AutoModel.from_pretrained("nazyrova/clinicalBERT").eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clinical_bert.to(device)

embeddings = []
max_length = 512

for idx, row in tqdm(discharge_summaries.iterrows(), total=len(discharge_summaries)):
    hadm_id = row['hadm_id']
    text    = row['text'] if isinstance(row['text'], str) else ""

    # Tokenize
    encoded = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass
    with torch.no_grad():
        outputs = clinical_bert(**encoded)
        last_hidden_state = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]

    # Take CLS token embedding at index 0
    cls_emb = last_hidden_state[:, 0, :]
    cls_emb = cls_emb.squeeze().cpu().numpy()

    embeddings.append([hadm_id] + cls_emb.tolist())

# Convert to DataFrame
dim = len(embeddings[0]) - 1
cols = ['hadm_id'] + [f'emb_{i}' for i in range(dim)]
clinicalbert_df = pd.DataFrame(embeddings, columns=cols)

# Save
clinicalbert_df.to_csv("clinicalbert_embeddings.csv", index=False)

####################################
# 4. Labels & Train/Test Split
####################################

labels_df = pd.read_csv('readmission_labels.csv').set_index('hadm_id')
labels_df = labels_df.reindex(hadm_ids)
y = labels_df['label'].values  # shape: (N,)

# Train/test IDs
train_ids = pd.read_csv('training_ids.csv')['hadm_id'].unique()
test_ids  = pd.read_csv('test_ids.csv')['hadm_id'].unique()

# Create index mapping
hadm_id_to_idx = {hid: i for i, hid in enumerate(hadm_ids)}

train_idx = [hadm_id_to_idx[hid] for hid in train_ids if hid in hadm_id_to_idx]
test_idx  = [hadm_id_to_idx[hid] for hid in test_ids  if hid in hadm_id_to_idx]

X_seq_train = padded_sequences[train_idx]          # (N_train, T, F)
X_struct_train = structured_data_array[train_idx]  # (N_train, D)
X_text_train = textual_data_array[train_idx]       # (N_train, E=768)
y_train = y[train_idx]

X_seq_test = padded_sequences[test_idx]
X_struct_test = structured_data_array[test_idx]
X_text_test = textual_data_array[test_idx]
y_test = y[test_idx]

print("Train shapes:")
print("  Temporal:", X_seq_train.shape)
print("  Structured:", X_struct_train.shape)
print("  Textual:", X_text_train.shape)
print("  Labels:", y_train.shape)

print("Test shapes:")
print("  Temporal:", X_seq_test.shape)
print("  Structured:", X_struct_test.shape)
print("  Textual:", X_text_test.shape)
print("  Labels:", y_test.shape)



####################################
# 5. Model Definition
####################################
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Attention Layer
# ---------------------------
class AttentionLayer(layers.Layer):
    """Simple self-attention that learns a weight for each timestep."""
    def __init__(self, units=64):
        super(AttentionLayer, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        # inputs: shape (batch, timesteps, hidden_dim)
        score = self.V(tf.nn.tanh(self.W(inputs)))  # shape (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_dim)
        return context_vector, attention_weights

# ---------------------------
# 5.1 Temporal Branch
# ---------------------------
T, F = 25, 48
temporal_input = Input(shape=(T, F), name='temporal_input')

x = layers.Masking(mask_value=0.0)(temporal_input)
x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)

# Apply attention to get a single vector
att = AttentionLayer(units=64)
x, att_w = att(x)  # x -> shape (batch, 128) after bidirectional
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
Z_temp = layers.Dropout(0.2)(x)  # Temporal embedding

# ---------------------------
# 5.2 Structured Branch (MLP)
# ---------------------------
structured_input = Input(shape=(structured_data_array.shape[1],), name='structured_input')
y = layers.Dense(64, activation='relu')(structured_input)
y = layers.Dropout(0.2)(y)
y = layers.Dense(32, activation='relu')(y)
Z_struct = layers.Dropout(0.2)(y)  # structured embedding

# ---------------------------
# 5.3 Textual Branch (ClinicalBERT) 
#     Already precomputed to shape (N, 768)
# ---------------------------
text_input = Input(shape=(textual_data_array.shape[1],), name='clinicalbert_input')
z = layers.Dense(128, activation='relu')(text_input)
z = layers.Dropout(0.2)(z)
z = layers.Dense(64, activation='relu')(z)
Z_text = layers.Dropout(0.2)(z)  # textual embedding

# ---------------------------
# 5.4 Late Fusion
#     Concatenate the 3 embeddings -> Dense -> final output
# ---------------------------
fusion = layers.Concatenate(axis=-1)([Z_temp, Z_struct, Z_text]) 
fusion = layers.Dense(128, activation='relu')(fusion)
fusion = layers.Dropout(0.2)(fusion)
fusion = layers.Dense(64, activation='relu')(fusion)
fusion = layers.Dropout(0.2)(fusion)
output = layers.Dense(1, activation='sigmoid')(fusion)

# ---------------------------
# Create and compile model
# ---------------------------
model = Model(
    inputs=[temporal_input, structured_input, text_input],
    outputs=output
)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['AUC']
)

model.summary()


#################################
# 6. Training and Evaluation
#################################


# Training
history = model.fit(
    x=[X_seq_train, X_struct_train, X_text_train],
    y=y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', 
            mode='max',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluation
results = model.evaluate(
    x=[X_seq_test, X_struct_test, X_text_test],
    y=y_test
)
print("Test Loss, Test AUC:", results)
