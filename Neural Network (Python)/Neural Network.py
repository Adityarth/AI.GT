
import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# =========================================================
# SETTINGS
# =========================================================

USE_EXCEL_FILE = False

CSV_DATASET_PATH = "traffic_dataset.csv"

EXCEL_DATASET_PATH = "traffic_dataset.xlsx"

EXPORT_ONNX = False

TRAIN_MODEL = True

EPOCHS = 100

BATCH_SIZE = 5000

LEARNING_RATE = 0.001

# =========================================================
# MODEL FILES
# =========================================================

MASTER_MODEL_PATH = "Traffic_AI_Master.pth"

BEST_MODEL_PATH = "Traffic_AI_Best.pth"

ONNX_PATH = "traffic_model.onnx"

# =========================================================
# CUDA SETTINGS
# =========================================================

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===================================")
print("DEVICE:", device)
print("===================================")

device_type = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    print("CUDA:", torch.version.cuda)

# =========================================================
# LOAD DATASET
# =========================================================

if not USE_EXCEL_FILE:

    print("\n===================================")
    print("LOADING CSV DATASET")
    print("===================================")

    df = pd.read_csv("traffic_dataset.csv")

else:

    print("\n===================================")
    print("LOADING EXCEL DATASET")
    print("===================================")

    df = pd.read_excel(
        EXCEL_DATASET_PATH
    )

# =========================================================
# SHOW DATASET
# =========================================================

print("\n===================================")
print("DATASET LOADED")
print("===================================")

print(df.head())

print("\nDATASET SHAPE:")

print(df.shape)

# =========================================================
# INPUTS / OUTPUTS
# =========================================================

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
#
# d1,d2,d3,d4
# f1,f2,f3,f4
# sw
# emg
# emg_way
#
# TOTAL = 11 INPUTS
#
# ---------------------------------------------------------

X = df.iloc[:, 0:11].values.astype(np.float32)

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

y_timer = df["timer"].values.astype(np.float32)

y_emg = df["emergency_signal"].values.astype(np.int64)

y_way = df["out_way"].values.astype(np.int64)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, \
y_timer_train, y_timer_test, \
y_emg_train, y_emg_test, \
y_way_train, y_way_test = train_test_split(X, y_timer, y_emg, y_way, test_size=0.2, random_state=42)

# =========================================================
# CONVERT TO TENSORS
# =========================================================

X_train = torch.tensor( X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_timer_train = torch.tensor(y_timer_train, dtype=torch.float32).view(-1, 1)

y_timer_test = torch.tensor(y_timer_test, dtype=torch.float32).view(-1, 1)

y_emg_train = torch.tensor(y_emg_train, dtype=torch.float32).view(-1,1)

y_emg_test = torch.tensor(y_emg_test, dtype=torch.float32).view(-1,1)

y_way_train = torch.tensor(y_way_train, dtype=torch.float32).view(-1,1)

y_way_test = torch.tensor(y_way_test, dtype=torch.float32).view(-1,1)

# =========================================================
# DATALOADER
# =========================================================

train_dataset = TensorDataset( X_train, y_timer_train, y_emg_train, y_way_train)

train_loader = DataLoader( train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

# =========================================================
# MODEL
# =========================================================

class TrafficNet(nn.Module):

    def __init__(self):

        super(TrafficNet, self).__init__()

        # -------------------------------------------------
        # SHARED LAYERS
        # -------------------------------------------------

        self.shared = nn.Sequential(

            nn.Linear(11, 128),

            nn.ReLU(),

            nn.Linear(128, 256),

            nn.ReLU(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU()
        )

        # -------------------------------------------------
        # OUTPUT HEADS
        # -------------------------------------------------

        self.timer_head = nn.Linear(64, 1)

        self.emg_head = nn.Linear(64, 1)

        self.way_head = nn.Linear(64, 1)

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x):

        x = self.shared(x)

        timer = self.timer_head(x)

        emg = self.emg_head(x)

        way = self.way_head(x)

        return timer, emg, way

# =========================================================
# CREATE MODEL
# =========================================================

model = TrafficNet().to(device)

# =========================================================
# LOAD EXISTING MASTER MODEL
# =========================================================

if os.path.exists(MASTER_MODEL_PATH):

    print("\n===================================")
    print("LOADING EXISTING MODEL")
    print("===================================")

    model.load_state_dict(

        torch.load(MASTER_MODEL_PATH, map_location=device)
    )

    print("MASTER MODEL LOADED")

else:

    print("\n===================================")
    print("NO EXISTING MODEL FOUND")
    print("CREATING NEW MODEL")
    print("===================================")

# =========================================================
# LOSS FUNCTIONS
# =========================================================

timer_loss_fn = nn.MSELoss()

emg_loss_fn = nn.BCEWithLogitsLoss()

way_loss_fn = nn.MSELoss()

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================================================
# FP16 MIXED PRECISION
# =========================================================

scaler_amp = torch.amp.GradScaler("cuda")

# =========================================================
# TRAINING
# =========================================================

best_loss = float('inf')

if TRAIN_MODEL:

    print("\n===================================")
    print("TRAINING STARTED")
    print("===================================")

    for epoch in range(EPOCHS):

        model.train()

        total_loss_epoch = 0

        for batch in train_loader:

            x_batch, \
            timer_batch, \
            emg_batch, \
            way_batch = batch

            # -------------------------------------------------
            # MOVE TO GPU
            # -------------------------------------------------

            x_batch = x_batch.to(device,non_blocking=True)

            timer_batch = timer_batch.to(device,non_blocking=True)

            emg_batch = emg_batch.to(device,non_blocking=True)

            way_batch = way_batch.to(device,non_blocking=True)

            optimizer.zero_grad()

            # ================================================
            # FP16 FORWARD
            # ================================================

            with torch.amp.autocast(device_type=device_type):

                pred_timer, \
                pred_emg, \
                pred_way = model(x_batch)

                # --------------------------------------------
                # LOSSES
                # --------------------------------------------

                loss_timer = timer_loss_fn(pred_timer,timer_batch)

                loss_emg = emg_loss_fn(pred_emg,emg_batch)

                loss_way = way_loss_fn(pred_way,way_batch)

                # --------------------------------------------
                # TOTAL LOSS
                # --------------------------------------------

                total_loss = (loss_timer +loss_emg +loss_way)

            # ================================================
            # BACKPROP
            # ================================================

            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)

            scaler_amp.update()

            total_loss_epoch += total_loss.item()

        # =================================================
        # PRINT
        # =================================================

        print(
            f"\nEpoch [{epoch+1}/{EPOCHS}] "
            f"\nLoss: {total_loss_epoch:.4f}"
            f"\tTimer={loss_timer.item():.4f}",
            f"\tEmergency={loss_emg.item():.4f}",
            f"\tWay={loss_way.item():.4f}",
            f"\tTotal={total_loss.item():.4f}"
        )

        # =================================================
        # SAVE MASTER MODEL
        # =================================================

        torch.save(

            model.state_dict(),

            MASTER_MODEL_PATH
        )

        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if total_loss_epoch < best_loss:

            best_loss = total_loss_epoch

            torch.save(model.state_dict(),BEST_MODEL_PATH)

            print("BEST MODEL SAVED")

# =========================================================
# OPTIONAL ONNX EXPORT
# =========================================================

if EXPORT_ONNX:

    print("\n===================================")
    print("EXPORTING ONNX")
    print("===================================")

    dummy_input = torch.randn(1,11).to(device)

    torch.onnx.export(

        model,

        dummy_input,

        ONNX_PATH,

        export_params=True,

        opset_version=11,

        do_constant_folding=True,

        input_names=['input'],

        output_names=['timer','emergency','way'],

        dynamic_axes={

            'input': {
                0: 'batch_size'
            },

            'timer': {
                0: 'batch_size'
            },

            'emergency': {
                0: 'batch_size'
            },

            'way': {
                0: 'batch_size'
            }
        }
    )

    print("ONNX EXPORTED")

# =========================================================
# REAL-TIME PREDICTION
# =========================================================

def predict_traffic(

    d1, d2, d3, d4,

    f1, f2, f3, f4,

    sw,

    emg,

    emg_way):

    model.eval()

    # =====================================================
    # EMERGENCY OVERRIDE
    # =====================================================

    if emg == 1:

        print("\n===================================")
        print("EMERGENCY OVERRIDE ACTIVATED")
        print("===================================")

        emergency_timer = 120

        emergency_way = emg_way

        return (emergency_timer, 1, emergency_way
        )

    # =====================================================
    # NORMAL AI INFERENCE
    # =====================================================

    input_data = np.array([[
        d1, d2, d3, d4,

        f1, f2, f3, f4,

        sw,

        emg,

        emg_way ]], dtype=np.float32)

    # -----------------------------------------------------
    # NORMALIZE INPUT
    # -----------------------------------------------------

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():

        with torch.cuda.amp.autocast():

            timer_out, \
              emg_out, \
            way_out = model(input_tensor)

    # =====================================================
    # OUTPUTS
    # =====================================================

    timer = timer_out.item()

    emergency = round(torch.sigmoid(emg_out).item()
)

    way = round(way_out.item())

    return timer, emergency, way
