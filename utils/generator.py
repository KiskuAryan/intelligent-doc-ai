import mlx_lm

MODEL_ID = "mlx-community/Phi-3-mini-4k-instruct-4bit"

# Lazy-loaded singletons
_model = None
_tokenizer = None

def get_generator_model():
    """
    Lazy singleton loader for the MLX 4-bit model and tokenizer.
    Runs locally on Apple Silicon Metal GPU with ~2.2 GB unified RAM footprint.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = mlx_lm.load(MODEL_ID)
    return _model, _tokenizer

def format_prompt(prompt_text):
    """
    Formats the prompt using the official Phi-3 instruct chat template:
    <|user|>\n{prompt}<|end|>\n<|assistant|>\n
    """
    _, tokenizer = get_generator_model()
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_answer(prompt, max_tokens=400):
    """
    Generates a full answer string (for CLI or non-streaming requests).
    Runs at 50+ tokens/sec on Apple M-series GPU via MLX.

    Args:
        prompt : full prompt string (context + question)
        max_tokens : maximum generation tokens

    Returns:
        Generated answer string
    """
    model, tokenizer = get_generator_model()
    formatted = format_prompt(prompt)
    answer = mlx_lm.generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        verbose=False
    )
    return answer.strip()

def stream_answer(prompt, max_tokens=500):
    """
    Streams answer tokens in real-time as they are produced by Apple Metal GPU.
    Yields text chunks suitable for Streamlit `st.write_stream()`.

    Args:
        prompt : full prompt string (context + question)
        max_tokens : maximum generation tokens

    Yields:
        str : newly generated token text
    """
    model, tokenizer = get_generator_model()
    formatted = format_prompt(prompt)
    for response in mlx_lm.stream_generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens):
        if response.text:
            yield response.text
