# GGUF Conversion Artifacts

Due to file size (quantized model ~2.2GB, base f16 ~7.6GB), these
binary artifacts are not stored in this git repository. They were
produced via the scripts in this folder and are hosted externally:

- Base f16 GGUF: converted via llama.cpp's convert_hf_to_gguf.py
- LoRA adapter GGUF: converted via convert_lora_to_gguf.py
- Quantized Q4_K_M GGUF: produced via llama-cpp-python's low-level
  quantize API (see quantize_model.py)

Conversion performed on Kaggle Notebooks (CPU, 30GB RAM) after
Colab's free-tier RAM ceiling (~12GB) proved insufficient for
ONNX Runtime GenAI's builder tool -- documented pivot from ONNX
to GGUF/llama.cpp due to this concrete resource constraint.
