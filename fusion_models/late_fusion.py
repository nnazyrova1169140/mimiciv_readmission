import pandas as pd
import numpy as np
from tqdm import tqdm


###############################
# 1. Load Data
###############################

# -------------------------------------------------------------------
#   X_chem.shape = (N, T, F_chem)
#   X_cbc.shape  = (N, T, F_cbc)
#   X_bdiff.shape= (N, T, F_bdiff)
#   X_vitals.shape=(N, T, F_vitals)
# -------------------------------------------------------------------

# Load each domain's preprocessed time-series (already padded to T)
X_chem   = np.load("X_chem_padded.npy")
X_cbc    = np.load("X_cbc_padded.npy")
X_bdiff  = np.load("X_bdiff_padded.npy")
X_vitals = np.load("X_vitals_padded.npy")

print("Chemistry shape:", X_chem.shape)
print("CBC shape:",       X_cbc.shape)
print("Blood diff shape:",X_bdiff.shape)
print("Vitals shape:",    X_vitals.shape)

# Readmission labels
labels_df = pd.read_csv("readmission_labels.csv")  # columns: hadm_id, label
hadm_ids = labels_df["hadm_id"].values
y = labels_df["label"].values  # shape (N,)

# Train/test splits by hadm_id, or load them
train_ids = pd.read_csv("train_ids.csv")["hadm_id"].unique()
test_ids  = pd.read_csv("test_ids.csv")["hadm_id"].unique()

# Build a map from hadm_id -> index
hadm_id_to_idx = {hadm: i for i, hadm in enumerate(hadm_ids)}

train_idx = [hadm_id_to_idx[h] for h in train_ids if h in hadm_id_to_idx]
test_idx  = [hadm_id_to_idx[h] for h in test_ids  if h in hadm_id_to_idx]

X_chem_train   = X_chem[train_idx]
X_chem_test    = X_chem[test_idx]
X_cbc_train    = X_cbc[train_idx]
X_cbc_test     = X_cbc[test_idx]
X_bdiff_train  = X_bdiff[train_idx]
X_bdiff_test   = X_bdiff[test_idx]
X_vitals_train = X_vitals[train_idx]
X_vitals_test  = X_vitals[test_idx]

y_train = y[train_idx]
y_test  = y[test_idx]



################################################
# 2. Structured Data
#################################################

import xgboost as xgb


structured_df = pd.read_csv("structured_data.csv")  
structured_df = structured_df.set_index("hadm_id").reindex(hadm_ids).fillna(0)
X_struct = structured_df.values

X_struct_train = X_struct[train_idx]
X_struct_test  = X_struct[test_idx]

# Train an XGBoost model
dtrain = xgb.DMatrix(X_struct_train, label=y_train)
dtest  = xgb.DMatrix(X_struct_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 42,
    "max_depth": 3,
    "eta": 0.1,
	"scale_pos_weight": 8,
}
bst = xgb.train(params, dtrain, num_boost_round=100)

# single probability as the "structured embedding"
Z_struct_train = bst.predict(dtrain)
Z_struct_test  = bst.predict(dtest)


Z_struct_train = Z_struct_train.reshape(-1, 1)  # shape => (N_train, 1)
Z_struct_test  = Z_struct_test.reshape(-1, 1)   # shape => (N_test, 1)



##################################################
#3. Unstructured Data
##################################################

import torch
from transformers import AutoTokenizer, AutoModel

# Example of precomputed embeddings
clinicalbert_df = pd.read_csv("clinicalbert_embeddings.csv")  # shape => (N, 768+1)
clinicalbert_df = clinicalbert_df.set_index("hadm_id").reindex(hadm_ids).fillna(0)
X_text = clinicalbert_df.values  # shape (N, 768)

X_text_train = X_text[train_idx]
X_text_test  = X_text[test_idx]


#############################################
# 4. Model Architecture
#############################################
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Masking
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_lstm_branch(timesteps, features, branch_name="chem"):
    """Returns a small LSTM-based sub-model that outputs a single embedding."""
    inp = Input(shape=(timesteps, features), name=f"{branch_name}_input")
    x = Masking(mask_value=0.0)(inp)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
    # Return sequences=False => final hidden state only (embedding)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(16, activation='relu', name=f"{branch_name}_embedding")(x)  
    # 16-d embedding for demonstration
    return Model(inp, out, name=f"{branch_name}_branch")

# Build each domain branch
T_chem, F_chem = X_chem_train.shape[1], X_chem_train.shape[2]
T_cbc,  F_cbc  = X_cbc_train.shape[1],  X_cbc_train.shape[2]
T_bdiff,F_bdiff= X_bdiff_train.shape[1],X_bdiff_train.shape[2]
T_vit,  F_vit  = X_vitals_train.shape[1],X_vitals_train.shape[2]

chem_branch   = build_lstm_branch(T_chem,   F_chem,   "chem")   # outputs 16-d
cbc_branch    = build_lstm_branch(T_cbc,    F_cbc,    "cbc")    
bdiff_branch  = build_lstm_branch(T_bdiff,  F_bdiff,  "bdiff")
vitals_branch = build_lstm_branch(T_vit,    F_vit,    "vitals")

# ------------------------------------------------------------------
# 2.2 Textual branch
# ------------------------------------------------------------------
text_input = Input(shape=(768,), name="text_input")
z = Dense(64, activation='relu')(text_input)
z = Dropout(0.2)(z)
text_output = Dense(16, activation='relu', name="text_embedding")(z)
text_branch = Model(text_input, text_output, name="text_branch")

# ------------------------------------------------------------------
# 2.3 XGBoost output (structured)
# ------------------------------------------------------------------

xgb_input = Input(shape=(1,), name="xgboost_input")  # shape => (batch, 1)
# Optionally pass it through a small transform
xgb_output = Dense(8, activation='relu', name="xgb_embedding")(xgb_input)
xgb_branch = Model(xgb_input, xgb_output, name="xgb_branch")

# ------------------------------------------------------------------
# 2.4 Late Fusion: Concatenate all sub-branch outputs
# ------------------------------------------------------------------

# Instantiate input placeholders
chem_input   = chem_branch.input
cbc_input    = cbc_branch.input
bdiff_input  = bdiff_branch.input
vitals_input = vitals_branch.input
text_in      = text_branch.input
xgb_in       = xgb_branch.input

# Get each embedding
Z_chem   = chem_branch(chem_input)     # shape (None, 16)
Z_cbc    = cbc_branch(cbc_input)       # shape (None, 16)
Z_bdiff  = bdiff_branch(bdiff_input)   # shape (None, 16)
Z_vitals = vitals_branch(vitals_input) # shape (None, 16)
Z_text   = text_branch(text_in)        # shape (None, 16)
Z_xgb    = xgb_branch(xgb_in)          # shape (None, 8)

# Concatenate
from tensorflow.keras.layers import Concatenate

fusion_input = Concatenate(name="fusion_concat")([
    Z_chem, Z_cbc, Z_bdiff, Z_vitals, Z_text, Z_xgb
])
# e.g. final shape => (None, 16+16+16+16+16+8) = (None, 88)

# Final classifier
fusion_hidden = Dense(32, activation='relu')(fusion_input)
fusion_out = Dense(1, activation='sigmoid', name="readmission")(fusion_hidden)

# Build full model
model = Model(
    inputs=[chem_input, cbc_input, bdiff_input, vitals_input, text_in, xgb_in],
    outputs=fusion_out,
    name="LateFusionModel"
)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['AUC']
)

model.summary()
