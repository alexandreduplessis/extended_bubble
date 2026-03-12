"""Tests for the constraint encoding and generation system."""

import numpy as np
import pytest

from src.data.constraints import (
    CONSTRAINT_DIM,
    NUM_CONSTRAINT_TYPES,
    NUM_ROOM_TYPES,
    Constraint,
    ConstraintType,
    encode_constraint,
    generate_constraints_from_bubbles,
)


# ---------------------------------------------------------------------------
# encode_constraint
# ---------------------------------------------------------------------------

class TestEncodeConstraint:
    def test_output_shape(self):
        c = Constraint(type=ConstraintType.MIN_AREA, room_type_a=0, value=5.0)
        vec = encode_constraint(c)
        assert vec.shape == (CONSTRAINT_DIM,)
        assert vec.shape == (30,)

    def test_type_onehot(self):
        for ct in ConstraintType:
            c = Constraint(type=ct)
            vec = encode_constraint(c)
            # Only the correct type index should be 1
            type_section = vec[:NUM_CONSTRAINT_TYPES]
            assert type_section[int(ct)] == 1.0
            assert type_section.sum() == 1.0

    def test_room_type_a_onehot(self):
        c = Constraint(type=ConstraintType.MUST_EXIST, room_type_a=3)
        vec = encode_constraint(c)
        a_section = vec[NUM_CONSTRAINT_TYPES : NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES]
        assert a_section[3] == 1.0
        assert a_section.sum() == 1.0

    def test_room_type_b_onehot(self):
        c = Constraint(
            type=ConstraintType.REQUIRED_ADJACENCY, room_type_a=1, room_type_b=5
        )
        vec = encode_constraint(c)
        b_section = vec[
            NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES : NUM_CONSTRAINT_TYPES + 2 * NUM_ROOM_TYPES
        ]
        assert b_section[5] == 1.0
        assert b_section.sum() == 1.0

    def test_unused_room_types_are_zero(self):
        c = Constraint(type=ConstraintType.MIN_AREA)
        vec = encode_constraint(c)
        # room_type_a and room_type_b sections should be all zero
        a_section = vec[NUM_CONSTRAINT_TYPES : NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES]
        b_section = vec[
            NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES : NUM_CONSTRAINT_TYPES + 2 * NUM_ROOM_TYPES
        ]
        assert a_section.sum() == 0.0
        assert b_section.sum() == 0.0

    def test_value_position(self):
        c = Constraint(type=ConstraintType.MIN_AREA, room_type_a=0, value=42.0)
        vec = encode_constraint(c)
        assert vec[-1] == pytest.approx(42.0)

    def test_correct_onehot_positions(self):
        """Verify exact positions in the 30-dim vector."""
        c = Constraint(
            type=ConstraintType.MAX_DISTANCE,  # index 5
            room_type_a=2,
            room_type_b=7,
            value=10.5,
        )
        vec = encode_constraint(c)
        # type at index 5
        assert vec[5] == 1.0
        # room_type_a=2 at index 11+2=13
        assert vec[13] == 1.0
        # room_type_b=7 at index 20+7=27
        assert vec[27] == 1.0
        # value at index 29
        assert vec[29] == pytest.approx(10.5)
        # total non-zero entries: 3 one-hots + 1 value = 4
        assert np.count_nonzero(vec) == 4


# ---------------------------------------------------------------------------
# generate_constraints_from_bubbles
# ---------------------------------------------------------------------------

def _make_simple_layout():
    """A small layout with 4 bubbles of 2 room types inside a square boundary."""
    bubbles = [
        (1.0, 1.0, 1.0, 0),  # type 0
        (3.5, 1.0, 1.0, 0),  # type 0
        (1.0, 3.5, 1.0, 1),  # type 1
        (3.5, 3.5, 1.5, 1),  # type 1
    ]
    boundary = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 5.0],
        [0.0, 5.0],
    ])
    return bubbles, boundary


class TestGenerateConstraints:
    def test_returns_list_of_constraints(self):
        bubbles, boundary = _make_simple_layout()
        constraints = generate_constraints_from_bubbles(bubbles, boundary, seed=42)
        assert isinstance(constraints, list)
        assert len(constraints) > 0
        for c in constraints:
            assert isinstance(c, Constraint)

    def test_reasonable_count(self):
        bubbles, boundary = _make_simple_layout()
        constraints = generate_constraints_from_bubbles(
            bubbles, boundary, n_sample=200, seed=42
        )
        # With 4 bubbles and 2 types we expect a moderate number of constraints
        assert len(constraints) >= 5
        assert len(constraints) <= 200

    def test_must_exist_present(self):
        bubbles, boundary = _make_simple_layout()
        constraints = generate_constraints_from_bubbles(
            bubbles, boundary, n_sample=500, seed=42
        )
        must_exist = [c for c in constraints if c.type == ConstraintType.MUST_EXIST]
        # Should have MUST_EXIST for types 0 and 1
        present_types = {c.room_type_a for c in must_exist}
        assert 0 in present_types
        assert 1 in present_types

    def test_empty_bubbles(self):
        boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        constraints = generate_constraints_from_bubbles([], boundary, seed=42)
        assert constraints == []

    def test_sampling_respects_n_sample(self):
        bubbles, boundary = _make_simple_layout()
        constraints = generate_constraints_from_bubbles(
            bubbles, boundary, n_sample=5, seed=42
        )
        assert len(constraints) <= 5

    def test_all_constraints_are_instances(self):
        bubbles, boundary = _make_simple_layout()
        constraints = generate_constraints_from_bubbles(
            bubbles, boundary, n_sample=100, seed=42
        )
        for c in constraints:
            assert isinstance(c, Constraint)
            assert isinstance(c.type, ConstraintType)

    def test_seed_reproducibility(self):
        bubbles, boundary = _make_simple_layout()
        c1 = generate_constraints_from_bubbles(bubbles, boundary, seed=123)
        c2 = generate_constraints_from_bubbles(bubbles, boundary, seed=123)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.type == b.type
            assert a.room_type_a == b.room_type_a
            assert a.room_type_b == b.room_type_b
            assert a.value == pytest.approx(b.value)
