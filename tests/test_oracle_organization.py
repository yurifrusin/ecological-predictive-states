"""Public prescribed tensors only. No optimizer, simulator, pilot or private read."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from epsbench.diagnostics.oracle_organization_contract import (
    Permission,
    decode_public,
    public_fixtures,
    read_public,
)
from epsbench.diagnostics.oracle_organization_models import Ecological, Generic
from epsbench.diagnostics.oracle_organization_readiness import (
    policy,
    readiness,
    regret,
    source_report,
)


def test_permission_before_read() -> None:
    def unread():
        raise AssertionError("supplier must not be read")

    with pytest.raises(PermissionError):
        read_public("public_oracle_software", unread)
    x = public_fixtures()[0].inputs
    assert np.array_equal(read_public(Permission.PUBLIC_ORACLE_SOFTWARE, lambda: x).masks, x.masks)
    with pytest.raises(TypeError):
        read_public(Permission.PUBLIC_ORACLE_SOFTWARE, lambda: {"depth": 1})


def test_literal_manifest() -> None:
    cases = public_fixtures()
    assert len(cases) == 8
    patterns = ((0, 0), (1, 0), (0, 1), (1, 1)) * 2
    for i, case in enumerate(cases):
        x = case.inputs
        assert x.executed == (1, 3) and x.announced == (-2, 2)
        assert case.success == (patterns[i], (1, 1))
        assert case.support == (tuple(16 * v for v in patterns[i]), (16, 16))
        assert x.masks.sum(axis=(2, 3)).tolist() == [[16, 16], [16, 16], [0, 16]]
        assert np.argwhere(x.masks[0, 0]).min(axis=0).tolist() == [2, 2 + i]
        assert np.argwhere(x.masks[1, 0]).min(axis=0).tolist() == [3, 3 + i]
        assert x.present.all() and x.available.all() and not x.contacts.any()
        assert not x.masks.flags.writeable


@pytest.mark.parametrize("factory", [Ecological, Generic])
def test_equivariance_and_common_initialization(factory) -> None:
    model = factory()
    x = public_fixtures()[2].inputs
    original = model(x)
    renamed = model(x.permuted())
    torch.testing.assert_close(renamed, original[[1, 0]], atol=2e-6, rtol=2e-6)
    again = factory()
    for p, q in zip(model.parameters(), again.parameters(), strict=True):
        assert torch.equal(p, q)
    e, g = Ecological(), Generic()
    for component in ("encoder", "head", "frame"):
        for p, q in zip(
            getattr(e, component).parameters(), getattr(g, component).parameters(), strict=True
        ):
            assert torch.equal(p, q)


@pytest.mark.parametrize("factory", [Ecological, Generic])
def test_missingness_and_relations(factory) -> None:
    x = public_fixtures()[0].inputs
    missing = x.present.copy()
    missing[2, 0] = False
    missing_available = x.available.copy()
    missing_available[2, 0, :] = missing_available[2, :, 0] = False
    absent = replace(x, present=missing, available=missing_available).checked()
    model = factory()
    assert not torch.allclose(model(x), model(absent))
    available = x.available.copy()
    available[1] = False
    unknown = replace(x, available=available).checked()
    if isinstance(model, Generic):
        assert not torch.equal(model.relations(x), model.relations(unknown))
    assert not torch.allclose(model(x), model(unknown))
    contacts = x.contacts.copy()
    contacts[1, 0, 1] = contacts[1, 1, 0] = True
    linked_masks = x.masks.copy()
    linked_masks[1, 1] = False
    linked_masks[1, 1, 3:7, 7:11] = True
    linked = replace(x, masks=linked_masks, contacts=contacts).checked()
    assert not torch.allclose(model(x), model(linked))
    with pytest.raises(ValueError):
        replace(x, contacts=contacts, available=available).checked()


@pytest.mark.parametrize("factory", [Ecological, Generic])
def test_reachable_parameters_and_analytic_gradient(factory) -> None:
    model = factory()
    x = public_fixtures()[3].inputs
    loss = model(x).square().sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Numerical central difference for a nonzero head bias gradient.
    parameter = model.head[2].bias
    analytic = parameter.grad[0].item()
    with torch.no_grad():
        parameter[0] += 0.001
        plus = model(x).square().sum().item()
        parameter[0] -= 0.002
        minus = model(x).square().sum().item()
        parameter[0] += 0.001
    assert analytic == pytest.approx((plus - minus) / 0.002, rel=0.01, abs=0.002)


def test_policy_ties_and_inert_entry() -> None:
    assert policy((0.75, 0.75)) == 2
    assert policy((1.0, 1.0)) == 0
    assert policy((0.0, 1.0)) == 1
    assert regret((0.0, 0.0), (0, 0)) == 0
    assert regret((1.0, 0.0), (0, 1)) == 4
    with pytest.raises(ValueError):
        policy((float("nan"), 0.1))
    with pytest.raises(PermissionError):
        readiness()


def test_accounting() -> None:
    report = source_report()
    assert report["relative_difference"] <= 0.05
    assert max(report["counts"].values()) <= 32768
    for profile in report["profiles"].values():
        assert profile["input_bytes"] <= 65536
        assert profile["derived_conservative_bytes"] <= 262144


def test_serialization_and_information_parity() -> None:
    e, g = Ecological(), Generic()
    for case in public_fixtures():
        x = case.inputs
        decoded = decode_public(Permission.PUBLIC_ORACLE_SOFTWARE, x.serialized)
        assert decoded.serialized() == x.serialized()
        torch.testing.assert_close(e.encode(x), g.encode(decoded), rtol=0, atol=0)
    with pytest.raises(ValueError):
        decode_public(Permission.PUBLIC_ORACLE_SOFTWARE, lambda: x.serialized() + b"privileged")


def test_fail_closed_types_and_targets() -> None:
    from epsbench.diagnostics.oracle_organization_contract import PublicFixture

    x = public_fixtures()[0].inputs
    with pytest.raises(ValueError):
        replace(x, masks=x.masks.astype(np.int32)).checked()
    with pytest.raises(ValueError):
        replace(x, executed=(True, 3)).checked()
    with pytest.raises(ValueError):
        replace(x, announced=(-2, 2, 1)).checked()
    with pytest.raises(ValueError):
        PublicFixture(x, ((0, 1), (1, 1)), ((0, 0), (16, 16)))


@pytest.mark.parametrize("factory", [Ecological, Generic])
def test_every_parameter_tensor_has_active_path(factory) -> None:
    x = public_fixtures()[2].inputs
    masks = x.masks.copy()
    masks[1, 1] = False
    masks[1, 1, 3:7, 9:13] = True
    contacts = x.contacts.copy()
    contacts[1, 0, 1] = contacts[1, 1, 0] = True
    x = replace(x, masks=masks, contacts=contacts).checked()
    model = factory()
    model(x).square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().max() > 0, name
