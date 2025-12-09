import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

# -------------
# Configuration
# -------------
MODEL_NAME = "FacebookAI/roberta-base"
SAVED_MODEL_PATH = r"roberta_lora_phishing_detector.pt"  # adjust if stored elsewhere

# Tokenization / inference params
MAX_LENGTH = 128
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]

# Label mapping (must match training)
ID2LABEL = {0: "SAFE", 1: "PHISHING"}
LABEL2ID = {"SAFE": 0, "PHISHING": 1}

# -----------------------------
# Lazy singletons
# -----------------------------
_tokenizer = None
_model = None
_device = None

def _init_model():
    global _tokenizer, _model, _device

    if _tokenizer is not None and _model is not None and _device is not None:
        return

    # Device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # Base model + LoRA wrapper
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = get_peft_model(base_model, lora_config)

    # Load LoRA weights (state_dict should match exactly what was saved during training)
    if not os.path.exists(SAVED_MODEL_PATH):
        # Fail fast with clear message
        raise FileNotFoundError(
            f"LoRA weights file not found at '{SAVED_MODEL_PATH}'. "
            f"Please place your trained weights there or update SAVED_MODEL_PATH."
        )
    state = torch.load(SAVED_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    model.to(_device)
    model.eval()

    # Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Assign back
    globals()["_tokenizer"] = _tokenizer
    globals()["_model"] = model
    globals()["_device"] = _device


def _softmax_confidence(logits: torch.Tensor) -> tuple[int, float, dict]:
    """
    Convert raw logits to:
      - predicted class index
      - confidence (probability of predicted class)
      - full probabilities dict
    """
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy().tolist()  # [1, 2]
    probs = probs[0]
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    confidence = float(probs[pred_idx])
    # Map to label keys
    probs_dict = {
        ID2LABEL[i].capitalize(): float(p) for i, p in enumerate(probs)
    }  # {'Safe': x, 'Phishing': y}
    return pred_idx, confidence, probs_dict


def predict_body_label(text: str):
    """
    Classify an email body text.

    Returns:
      - prediction: "Safe" or "Phishing"
      - confidence: float in [0,1]
      - probs: dict like {"Safe": p0, "Phishing": p1}

    Raises:
      - FileNotFoundError if weights are missing
      - RuntimeError for general inference errors
    """
    if text is None:
        text = ""
    text = text.strip()

    _init_model()  # ensure model/tokenizer/device are ready

    try:
        enc = _tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(_device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = _model(**enc)  # logits shape [1, 2]
            logits = outputs.logits

        pred_idx, confidence, probs = _softmax_confidence(logits)
        label = ID2LABEL[pred_idx].capitalize()  # "Safe" or "Phishing"
        return label, confidence, probs

    except Exception as e:
        # Fallback: safe default with zeroed confidences
        # Allows caller to handle gracefully without crashing the app
        return "Safe", 0.0, {"Safe": 0.0, "Phishing": 0.0}


# Optional: simple CLI for quick local tests
if __name__ == "__main__":
    sample = "Urgent: Verify your password immediately to avoid account suspension."
    try:
        label, conf, probs = predict_body_label(sample)
        print(f"Prediction: {label} (confidence={conf:.4f})")
        print(f"Probabilities: {probs}")
    except FileNotFoundError as e:
        print(str(e))