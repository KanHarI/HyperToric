"""Tests for N-dimensional torus topology."""

from __future__ import annotations

from itertools import product

import pytest

from hypertoric.topology import Topology

NDIMS = [2, 3, 4]
GRID_SIZES = [2, 3, 4, 5]


@pytest.fixture(
    params=list(product(NDIMS, GRID_SIZES)),
    ids=lambda p: f"{p[0]}d_g{p[1]}",
)
def topo(request: pytest.FixtureRequest) -> Topology:
    ndim, grid_size = request.param
    return Topology(ndim=ndim, grid_size=grid_size)


class TestProperties:
    def test_num_blocks(self, topo: Topology) -> None:
        assert topo.num_blocks == topo.grid_size**topo.ndim

    def test_num_neighbors(self, topo: Topology) -> None:
        assert topo.num_neighbors == 2 * topo.ndim

    def test_strides(self, topo: Topology) -> None:
        expected = tuple(topo.grid_size**a for a in range(topo.ndim))
        assert topo.strides == expected


class TestFlatCoordConversion:
    def test_round_trip_flat(self, topo: Topology) -> None:
        """coord_to_flat(flat_to_coord(i)) == i for all flat indices."""
        for i in range(topo.num_blocks):
            assert topo.coord_to_flat(topo.flat_to_coord(i)) == i

    def test_round_trip_coord(self, topo: Topology) -> None:
        """flat_to_coord(coord_to_flat(c)) == c for all valid coordinates."""
        for coord in product(range(topo.grid_size), repeat=topo.ndim):
            assert topo.flat_to_coord(topo.coord_to_flat(coord)) == coord

    def test_all_flat_indices_distinct(self, topo: Topology) -> None:
        """All coordinates map to distinct flat indices."""
        flat_indices = set()
        for coord in product(range(topo.grid_size), repeat=topo.ndim):
            flat_indices.add(topo.coord_to_flat(coord))
        assert len(flat_indices) == topo.num_blocks


class TestNeighbors:
    def test_neighbor_count(self, topo: Topology) -> None:
        """Every block has exactly 2*ndim neighbors."""
        for i in range(topo.num_blocks):
            neighbors = [
                topo.get_neighbor_flat(i, d) for d in range(topo.num_neighbors)
            ]
            assert len(neighbors) == topo.num_neighbors

    def test_neighbor_symmetry(self, topo: Topology) -> None:
        """If neighbor(a, d) == b, then neighbor(b, d^1) == a."""
        for i in range(topo.num_blocks):
            for d in range(topo.num_neighbors):
                neighbor = topo.get_neighbor_flat(i, d)
                opposite = d ^ 1
                assert topo.get_neighbor_flat(neighbor, opposite) == i

    def test_no_self_neighbors(self, topo: Topology) -> None:
        """No block is its own neighbor (requires grid_size >= 2)."""
        for i in range(topo.num_blocks):
            for d in range(topo.num_neighbors):
                assert topo.get_neighbor_flat(i, d) != i

    def test_neighbors_in_range(self, topo: Topology) -> None:
        """All neighbors are valid flat indices."""
        for i in range(topo.num_blocks):
            for d in range(topo.num_neighbors):
                n = topo.get_neighbor_flat(i, d)
                assert 0 <= n < topo.num_blocks

    def test_periodic_wrapping(self, topo: Topology) -> None:
        """Corner block (all coords=0) wraps to grid_size-1 in negative directions."""
        origin = topo.coord_to_flat(tuple(0 for _ in range(topo.ndim)))
        for axis in range(topo.ndim):
            # Odd direction = negative offset for this axis
            neg_dir = 2 * axis + 1
            neighbor = topo.get_neighbor_flat(origin, neg_dir)
            coord = topo.flat_to_coord(neighbor)
            assert coord[axis] == topo.grid_size - 1


class TestPlane:
    def test_plane_size(self, topo: Topology) -> None:
        """Each plane has grid_size^(ndim-1) blocks."""
        expected = topo.grid_size ** (topo.ndim - 1)
        for axis in range(topo.ndim):
            for pos in range(topo.grid_size):
                plane = topo.get_plane(axis, pos)
                assert len(plane) == expected

    def test_plane_coord_correct(self, topo: Topology) -> None:
        """All blocks in plane have the correct coordinate on the sliced axis."""
        for axis in range(topo.ndim):
            for pos in range(topo.grid_size):
                plane = topo.get_plane(axis, pos)
                for flat_idx in plane:
                    assert topo.flat_to_coord(flat_idx)[axis] == pos

    def test_planes_partition(self, topo: Topology) -> None:
        """Union of all planes along an axis equals all blocks."""
        for axis in range(topo.ndim):
            all_blocks: set[int] = set()
            for pos in range(topo.grid_size):
                plane = topo.get_plane(axis, pos)
                # No overlap
                assert all_blocks.isdisjoint(plane)
                all_blocks.update(plane)
            assert all_blocks == set(range(topo.num_blocks))

    def test_plane_invalid_axis(self) -> None:
        topo = Topology(ndim=3, grid_size=4)
        with pytest.raises(ValueError, match="axis"):
            topo.get_plane(3, 0)

    def test_plane_invalid_position(self) -> None:
        topo = Topology(ndim=3, grid_size=4)
        with pytest.raises(ValueError, match="position"):
            topo.get_plane(0, 4)


class TestEdgeCases:
    def test_minimal_torus(self) -> None:
        """2D grid_size=2: 4 blocks, each with 4 neighbors."""
        topo = Topology(ndim=2, grid_size=2)
        assert topo.num_blocks == 4
        assert topo.num_neighbors == 4

    def test_invalid_ndim(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            Topology(ndim=0, grid_size=4)

    def test_invalid_grid_size(self) -> None:
        with pytest.raises(ValueError, match="grid_size"):
            Topology(ndim=2, grid_size=1)
