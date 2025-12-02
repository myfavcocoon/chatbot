# src/model_loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from .config import MODEL_OPTIONS, HF_TOKEN, BASE_DIR

print(torch.cuda.is_available())

def load_model_with_adapter(model_key="qwen2-3b"):
    """
    Load a base model and merge with adapter from MODEL_OPTIONS.

    Returns:
        model: HuggingFace causal LM with adapter merged
        tokenizer: corresponding tokenizer
    """
    if model_key not in MODEL_OPTIONS:
        raise ValueError(f"Model key '{model_key}' not found in MODEL_OPTIONS.")

    cfg = MODEL_OPTIONS[model_key]
    base_model_path = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]

    # Nếu adapter chưa có, thông báo lỗi
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_auth_token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (4-bit quantized)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=HF_TOKEN
    )

    # Load adapter and merge
    model = PeftModel.from_pretrained(base_model, adapter_dir, device_map="auto").merge_and_unload()
    model.eval()

    return model, tokenizer

def build_pipeline(model_key="qwen2-3b", max_new_tokens=512, temperature=0.2):
    """
    Returns a HuggingFace pipeline ready for text-generation
    """
    model, tokenizer = load_model_with_adapter(model_key)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False
    )
    return gen_pipe, tokenizer

# ======= DEBUG / TEST =======
if __name__ == "__main__":
    key = "qwen2-3b"
    print(f"Loading model: {key}")

    try:
        # Test load_model_with_adapter
        model, tokenizer = load_model_with_adapter(key)
        print(f"  Base model: {MODEL_OPTIONS[key]['base_model']}")
        print(f"  Adapter dir: {MODEL_OPTIONS[key]['adapter_dir']}")
        print(f"  Tokenizer vocab size: {len(tokenizer)}")

        # Test build_pipeline
        print("Building HuggingFace pipeline...")
        gen_pipe, _ = build_pipeline(model_key=key)
        test_prompt = "Viết một câu hỏi pháp lý mẫu:"
        output = gen_pipe(test_prompt, max_new_tokens=20, do_sample=False)
        print("Pipeline test output:", output[0]["generated_text"])

    except Exception as e:
        print(f"  Error loading {key} or building pipeline: {e}")
