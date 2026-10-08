"""Bounded public N=2/3/4 source checks; no fitting or live allocation profiling."""

import struct
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest
import torch

from epsbench.diagnostics.oracle_organization_contract import (
    OracleInput,
    Permission,
    PublicFixture,
    decode_public,
    public_fixtures,
    public_region_fixtures,
)
from epsbench.diagnostics.oracle_organization_models import Ecological, Generic
from epsbench.diagnostics.oracle_organization_readiness import source_report
from tests import oracle_organization_v3_reference as old


def cases(n):
    return public_fixtures() if n == 2 else public_region_fixtures(n)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_literal_cardinality_and_serialization(n):
    original = public_fixtures()
    fixtures = cases(n)
    assert len(fixtures) == 8
    for base, case in zip(original, fixtures, strict=True):
        x = case.inputs
        assert x.region_count == n
        np.testing.assert_array_equal(x.masks[:, :2], base.inputs.masks)
        assert case.success[:2] == base.success and case.support[:2] == base.support
        assert x.executed == (1, 3) and x.announced == (-2, 2)
        assert x.storage_bytes() == {2: 6212, 3: 9320, 4: 12440}[n]
        if n > 2:
            assert x.masks[:, 2, 16:20, 4:8].all()
            assert x.masks[:, 2:].sum(axis=(2, 3)).tolist() == [[16] * (n - 2)] * 3
            assert case.success[2:] == ((1, 1),) * (n - 2)
        if n == 4:
            assert x.masks[:, 3, 16:20, 8:12].all()
            assert x.contacts.sum() == 6
            assert x.contacts[:, 2, 3].all() and x.contacts[:, 3, 2].all()
        else:
            assert not x.contacts.any()
        payload = x.serialized()
        if n == 2:
            expected = (
                b"EPS-PUBLIC-ORACLE-V3\0"
                + b"".join(
                    a.tobytes() for a in (x.masks, x.present, x.observed, x.contacts, x.available)
                )
                + struct.pack("<4q", 1, 3, -2, 2)
            )
            assert payload == expected
        else:
            assert payload.startswith(b"EPS-PUBLIC-ORACLE-N34-V1\0" + bytes([n]))
        restored = decode_public(Permission.PUBLIC_ORACLE_SOFTWARE, lambda payload=payload: payload)
        assert restored.serialized() == payload
        with pytest.raises(ValueError):
            decode_public(
                Permission.PUBLIC_ORACLE_SOFTWARE, lambda payload=payload: payload + b"privileged"
            )


@pytest.mark.parametrize("factory", [Ecological, Generic])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_all_permutations_and_information(factory, n):
    model = factory()
    case = cases(n)[2]
    x = case.inputs
    before = model(x)
    e, g = Ecological(), Generic()
    torch.testing.assert_close(e.encode(x), g.encode(x), rtol=0, atol=0)
    for order in permutations(range(n)):
        permuted = case.permuted(order)
        np.testing.assert_array_equal(permuted.inputs.masks, x.masks[:, list(order)])
        assert permuted.success == tuple(case.success[i] for i in order)
        torch.testing.assert_close(
            model(permuted.inputs), before[list(order)], atol=3e-6, rtol=3e-6
        )
    relations = g.relations(x)
    assert relations.shape == (3 * n, 3 * n, 3)
    for t in range(3):
        block = relations[t * n : (t + 1) * n, t * n : (t + 1) * n]
        torch.testing.assert_close(block[:, :, 1], torch.tensor(x.contacts[t], dtype=torch.float32))
        torch.testing.assert_close(
            block[:, :, 2], torch.tensor(x.available[t], dtype=torch.float32)
        )
    for a in range(3 * n):
        for b in range(3 * n):
            assert relations[a, b, 0].item() == float(a % n == b % n)
    assert (
        sum(p.numel() for p in model.parameters()) == {Ecological: 24209, Generic: 23881}[factory]
    )


@pytest.mark.parametrize("factory", [Ecological, Generic])
@pytest.mark.parametrize("n", [3, 4])
def test_missing_extra_nodes_keep_global_context(factory, n):
    x = cases(n)[0].inputs
    index = n - 1
    masks = x.masks.copy()
    masks[2, index] = False
    observed = x.observed.copy()
    observed[2, index] = False
    contacts = x.contacts.copy()
    contacts[2, index, :] = contacts[2, :, index] = False
    empty = replace(x, masks=masks, observed=observed, contacts=contacts).checked()
    present = empty.present.copy()
    present[2, index] = False
    available = empty.available.copy()
    available[2, index, :] = available[2, :, index] = False
    missing = replace(empty, present=present, available=available).checked()
    model = factory()
    assert model(missing).shape == (n, 2)
    assert not torch.allclose(model(empty), model(missing))
    # A is absent in the current frame. Changing an extra node's earlier mask
    # must remain available to A's global readout, even when that node is now missing.
    changed = masks.copy()
    changed[0, index] = False
    changed[0, index, 12:16, 20:24] = True
    changed_contacts = contacts.copy()
    changed_contacts[0, index, :] = changed_contacts[0, :, index] = False
    other = replace(missing, masks=changed, contacts=changed_contacts).checked()
    assert not torch.allclose(model(missing)[0], model(other)[0])


@pytest.mark.parametrize("n", [3, 4])
def test_every_pair_contact_validation_and_failures(n):
    x = cases(n)[0].inputs
    bad = x.contacts.copy()
    bad[:, 0, n - 1] = bad[:, n - 1, 0] = True
    with pytest.raises(ValueError):
        replace(x, contacts=bad).checked()
    for order in (
        (0,) * n,
        tuple(range(n - 1)),
        tuple(range(1, n + 1)),
        (True, *range(1, n)),
    ):
        with pytest.raises(ValueError):
            x.permuted(order)
    with pytest.raises(ValueError):
        PublicFixture(x, ((1, 1),), ((16, 16),))
    payload = x.serialized()
    offset = len(b"EPS-PUBLIC-ORACLE-N34-V1\0")
    for count in (0, 1, 2, 5, 255):
        malformed = payload[:offset] + bytes([count]) + payload[offset + 1 :]
        with pytest.raises(ValueError):
            decode_public(Permission.PUBLIC_ORACLE_SOFTWARE, lambda malformed=malformed: malformed)
    with pytest.raises(PermissionError):
        decode_public("public_oracle_software", lambda: pytest.fail("supplier read"))


@pytest.mark.parametrize("n", [0, 1, 5])
def test_invalid_cardinality(n):
    x = OracleInput(
        np.zeros((3, n, 32, 32), dtype=np.bool_),
        np.ones((3, n), dtype=np.bool_),
        np.zeros((3, n), dtype=np.bool_),
        np.zeros((3, n, n), dtype=np.bool_),
        np.ones((3, n, n), dtype=np.bool_),
        (1, 3),
        (-2, 2),
    )
    with pytest.raises(ValueError):
        x.checked()


@pytest.mark.parametrize("factory,legacy", [(Ecological, old.Ecological), (Generic, old.Generic)])
def test_exact_n2_untrained_reference_outputs_and_gradients(factory, legacy):
    model, reference = factory(), legacy()
    assert list(model.state_dict()) == list(reference.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key])
    for case in public_fixtures():
        model.zero_grad()
        reference.zero_grad()
        output, prior = model(case.inputs), reference(case.inputs)
        assert torch.equal(output, prior)
        output.square().sum().backward()
        prior.square().sum().backward()
        for parameter, previous in zip(model.parameters(), reference.parameters(), strict=True):
            assert torch.equal(parameter.grad, previous.grad)


@pytest.mark.parametrize("factory", [Ecological, Generic])
@pytest.mark.parametrize("n", [3, 4])
def test_new_cardinality_reachable_gradient_paths(factory, n):
    x = cases(n)[2].inputs
    masks = x.masks.copy()
    masks[1, 1] = False
    masks[1, 1, 3:7, 9:13] = True
    contacts = x.contacts.copy()
    contacts[1, 0, 1] = contacts[1, 1, 0] = True
    linked = replace(x, masks=masks, contacts=contacts).checked()
    model = factory()
    model(linked).square().sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, name


def test_forward_inventory_is_not_live_backward_qualification():
    report = source_report()
    assert set(report["additional_cardinality_profiles"]) == {"3", "4"}
    assert report["new_cardinality_live_accounting"].startswith("UNVERIFIED")
    for n, by_model in report["additional_cardinality_profiles"].items():
        for inventory in by_model.values():
            assert inventory["input_bytes"] == {3: 9320, 4: 12440}[int(n)]
            assert inventory["torch_cumulative_bytes"] > 0


@pytest.mark.parametrize("seed", [271828, 271829, 271830])
def test_bounded_seed_repeatability_and_common_parity(seed):
    e, g = Ecological(seed), Generic(seed)
    for factory, model in ((Ecological, e), (Generic, g)):
        repeated = factory(seed)
        for key, value in model.state_dict().items():
            assert torch.equal(value, repeated.state_dict()[key])
    for key, value in e.state_dict().items():
        if not key.startswith("layers."):
            assert torch.equal(value, g.state_dict()[key])
    if seed != 271828:
        assert not torch.equal(e.encoder.weight, Ecological().encoder.weight)


@pytest.mark.parametrize("factory", [Ecological, Generic])
@pytest.mark.parametrize("seed", [True, False, 271828.0, 271827, 271831, "271828", None])
def test_unadmitted_seed_rejected(factory, seed):
    with pytest.raises(ValueError):
        factory(seed)
