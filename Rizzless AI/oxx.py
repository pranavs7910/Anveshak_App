"""
export_onnx.py  -  Convert trained SimpleCNN to ONNX format
Usage:
    python export_onnx.py --model model.pth --data "C:\\path\\to\\data_main"
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="model.pth")
parser.add_argument("--output",  default="model.onnx")
parser.add_argument("--classes", default=3, type=int,
                    help="Number of classes (blue, green, orange = 3)")
parser.add_argument("--img",     default=128, type=int)
args = parser.parse_args()

DEVICE = torch.device("cpu")   # ONNX export must be done on CPU

# ──────────────────────────────────────────────
# MODEL  (must match train.py exactly)
# ──────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":

    # ──────────────────────────────────────────────
    # STEP 1 - Load trained weights
    # ──────────────────────────────────────────────
    model = SimpleCNN(args.classes).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()   # CRITICAL - must be in eval mode before export
                   # (disables Dropout, fixes BatchNorm if any)
    print(f"[1/4] Loaded weights from: {args.model}")

    # ──────────────────────────────────────────────
    # STEP 2 - Create dummy input (same shape as real input)
    # ONNX traces the model using this input to record all operations
    # batch_size=1, 3 channels, IMG_SIZE x IMG_SIZE
    # ──────────────────────────────────────────────
    dummy_input = torch.randn(1, 3, args.img, args.img, device=DEVICE)
    print(f"[2/4] Dummy input shape: {list(dummy_input.shape)}")

    # ──────────────────────────────────────────────
    # STEP 3 - Export to ONNX
    # dynamic_axes lets the model accept any batch size at inference
    # ──────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,          # store weights inside the .onnx file
        opset_version=11,            # opset 11 is widely supported
        do_constant_folding=True,    # optimise constants at export time
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},   # variable batch size
            "output": {0: "batch_size"},
        }
    )
    print(f"[3/4] Exported ONNX model -> {args.output}")

    # ──────────────────────────────────────────────
    # STEP 4 - Verify ONNX model with onnxruntime
    # Run the same dummy input through PyTorch and ONNX
    # and confirm outputs match
    # ──────────────────────────────────────────────
    try:
        import onnx
        import onnxruntime as ort

        # check model is valid
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("[4/4] ONNX model structure: VALID")

        # compare outputs
        ort_session = ort.InferenceSession(args.output)
        ort_inputs  = {"input": dummy_input.numpy()}
        ort_output  = ort_session.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_output = model(dummy_input).numpy()

        max_diff = np.max(np.abs(torch_output - ort_output))
        print(f"      Max output difference (PyTorch vs ONNX): {max_diff:.6f}")

        if max_diff < 1e-4:
            print("      Outputs match. Model is ready for deployment!")
        else:
            print("      WARNING: outputs differ more than expected, check export.")

        
    except ImportError:
        print("\n[4/4] onnx / onnxruntime not installed - skipping verification.")
        print("      Install with:  pip install onnx onnxruntime")
        print(f"      Export still succeeded -> {args.output}")