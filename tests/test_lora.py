"""LoRA on a tiny random Llama, so no weights are downloaded. Skipped where torch is missing."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from desk.ml import lora  # noqa: E402


@pytest.fixture
def model():
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()


X = torch.arange(1, 11).reshape(1, 10)


def out(m):
    with torch.no_grad():
        return m(X).logits


def train(m, steps=30):
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(m(X).logits[:, -1], torch.tensor([5]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def test_injection_changes_nothing_until_trained(model):
    before = out(model)
    lora.inject(model, r=4)
    torch.testing.assert_close(out(model.eval()), before)


def test_only_lora_parameters_train(model):
    n = lora.inject(model, r=4)
    # q_proj 32->32 and v_proj 32->16, rank 4, two layers
    assert n == 2 * (4 * (32 + 32) + 4 * (32 + 16))
    frozen = {k: v.clone() for k, v in model.named_parameters() if not v.requires_grad}
    train(model)
    for k, v in model.named_parameters():
        if k in frozen:
            assert torch.equal(v, frozen[k]), k


def test_training_lowers_the_loss(model):
    lora.inject(model, r=4, dropout=0.0)
    assert train(model, 60) < train(model, 1) + 1e-3
    assert out(model)[0, -1].argmax() == 5


def test_merge_gives_the_same_answers_without_extra_layers(model):
    lora.inject(model, r=4, dropout=0.0)
    train(model)
    want = out(model.eval())
    lora.merge(model)
    torch.testing.assert_close(out(model), want, atol=1e-5, rtol=1e-5)
    assert not any(isinstance(m, lora.LoRALinear) for m in model.modules())


def test_adapter_round_trip(model, tmp_path):
    import copy

    from safetensors.torch import load_file, save_file

    fresh = copy.deepcopy(model)
    lora.inject(model, r=4, dropout=0.0)
    train(model)
    save_file(lora.adapter_state(model), str(tmp_path / "a.safetensors"))
    lora.inject(fresh, r=4, dropout=0.0)
    lora.load_adapter(fresh, load_file(str(tmp_path / "a.safetensors")))
    torch.testing.assert_close(out(fresh.eval()), out(model.eval()))


def test_wrong_adapter_and_missing_layers_are_rejected(model):
    with pytest.raises(ValueError):
        lora.inject(model, names=("nothing",))
    lora.inject(model, r=4)
    with pytest.raises(ValueError):
        lora.load_adapter(model, {"x": torch.zeros(1)})
