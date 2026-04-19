import torch

from src.models.latent_residual_inputs import context_dict_shapes_match, context_dict_structure_equal


def test_context_dict_shapes_match_tensors_and_tuple_keys():
    a = {"T_c": torch.randn(2, 1, 8, 8), (4, 4): torch.randn(2, 16, 4, 4)}
    b = {"T_c": torch.randn(2, 1, 8, 8), (4, 4): torch.randn(2, 16, 4, 4)}
    ok, msg = context_dict_shapes_match(a, b)
    assert ok, msg


def test_context_dict_structure_equal_identical():
    a = {"T_c": torch.randn(2, 1, 8, 8)}
    b = {"T_c": a["T_c"].clone()}
    ok, msg = context_dict_structure_equal(a, b, rtol=0, atol=0)
    assert ok, msg
