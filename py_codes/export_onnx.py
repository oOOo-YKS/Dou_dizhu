# export_onnx.py
import torch
import numpy as np
from intelligence import Model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pth", required=True, help="path to your model .pth")
parser.add_argument("--out-dir", default="onnx_models", help="where to write ONNX files")
parser.add_argument("--opset", type=int, default=14)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# 1) Load model (handle both state_dict and saved model cases)
ckpt = torch.load(args.pth, map_location="cpu")
try:
    model = Model()
    model.load_state_dict(ckpt)
    print("Loaded state_dict into Model()")
except Exception:
    # If the .pth was saved as the entire model object:
    model = ckpt
    print("Loaded .pth as full model object")

model.eval()
model.to("cpu")

# 2) Export wrappers -------------------------------------------------------
# Policy wrapper: scores for N candidate actions
class PolicyExporter(torch.nn.Module):
    def __init__(self, model, head="play_first"):
        super().__init__()
        self.state = model.state  # StateModel
        if head == "play_first":
            self.head = model.play_first_head
        elif head == "respond":
            self.head = model.respond_head
        else:
            raise ValueError("head must be 'play_first' or 'respond'")

    def forward(self, history, hand, actions, lord):
        # history: (seq_len, 18)  float32
        # hand: (15,)            float32
        # actions: (num_actions, 15)
        # lord: (1,)             float32
        st = self.state(history, hand)          # (128,)
        n = actions.shape[0]
        st_exp = st.unsqueeze(0).expand(n, -1)  # (n,128)
        lord_exp = lord.expand(n, -1)           # (n,1)
        x = torch.cat([st_exp, actions, lord_exp], dim=1)  # (n, 144)
        out = self.head(x)   # (n,1) expected
        return out.squeeze(-1)  # (n,)

# Bid wrapper: logits for bidding
class BidExporter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.state = model.state
        self.bid_head = model.bid_head

    def forward(self, history, hand):
        st = self.state(history, hand)  # (128,)
        logits = self.bid_head(st)      # (3,) or (1,3)
        return logits

# Prepare dummy inputs (shapes chosen for example; dynamic axes will allow other sizes)
dummy_history = torch.zeros(5, 18, dtype=torch.float32)   # seq_len 5
dummy_hand = torch.zeros(15, dtype=torch.float32)
dummy_actions = torch.zeros(3, 15, dtype=torch.float32)  # 3 candidate moves
dummy_lord = torch.zeros(1, dtype=torch.float32)

policy_exporter = PolicyExporter(model, head="play_first")
policy_exporter.eval()

bid_exporter = BidExporter(model)
bid_exporter.eval()

policy_path = os.path.join(args.out_dir, "policy.onnx")
bid_path = os.path.join(args.out_dir, "bid.onnx")

print("Exporting policy ONNX ->", policy_path)
with torch.no_grad():
    torch.onnx.export(
        policy_exporter,
        (dummy_history, dummy_hand, dummy_actions, dummy_lord),
        policy_path,
        opset_version=args.opset,
        input_names=["history","hand","actions","lord"],
        output_names=["scores"],
        dynamic_axes={
            "history": {0: "seq_len"},
            "actions": {0: "num_actions"},
            "scores": {0: "num_actions"}
        },
        do_constant_folding=True,
    )

print("Exporting bid ONNX ->", bid_path)
with torch.no_grad():
    torch.onnx.export(
        bid_exporter,
        (dummy_history, dummy_hand),
        bid_path,
        opset_version=args.opset,
        input_names=["history","hand"],
        output_names=["logits"],
        dynamic_axes={
            "history": {0: "seq_len"},
        },
        do_constant_folding=True,
    )

print("Done. ONNX models written to", args.out_dir)

# "D:\Models\model_20250827_101447.pth"