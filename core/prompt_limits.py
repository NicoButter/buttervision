"""Use the loaded CLIP tokenizer; never log text discarded by CLIP."""
import logging

logger = logging.getLogger("uvicorn.error.engine")


def limit_prompts(pipeline, prompt, negative_prompt, warn=None):
    tokenizer = pipeline.tokenizer
    maximum = min(tokenizer.model_max_length,
                  pipeline.text_encoder.config.max_position_embeddings)
    results = []
    for label, text in (("prompt", prompt), ("prompt negativo", negative_prompt)):
        tokens = tokenizer(text, truncation=False, verbose=False).input_ids
        if len(tokens) > maximum:
            logger.warning("engine prompt_truncated field=%s tokens=%s limit=%s", label, len(tokens), maximum)
            if warn:
                warn(f"El {label} supera el límite CLIP de {maximum} tokens "
                     "(incluidos los tokens especiales) y se truncará. "
                     "El texto que exceda ese límite no se procesará.")
            content_limit = maximum - tokenizer.num_special_tokens_to_add(pair=False)
            content = tokenizer(text, add_special_tokens=False,
                                truncation=False, verbose=False).input_ids[:content_limit]
            text = tokenizer.decode(content, skip_special_tokens=True)
            # Decoding and re-tokenizing must also fit, including special tokens.
            while len(tokenizer(text, truncation=False, verbose=False).input_ids) > maximum:
                content = content[:-1]
                text = tokenizer.decode(content, skip_special_tokens=True)
        results.append(text)
    return tuple(results)
