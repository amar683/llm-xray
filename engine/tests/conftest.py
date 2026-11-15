"""Shared fixtures. The tiny model is random-initialised and built on the fly,
so tests need no downloads and no binary files in the repository."""

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    path = tmp_path_factory.mktemp("tiny-llama")
    LlamaForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny(tiny_model_dir):
    from xray.models.adapters import load_model

    return load_model(str(tiny_model_dir), device="cpu", with_tokenizer=False)


@pytest.fixture
def prompts():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 128, (1, n), generator=g) for n in (5, 9, 12)]
