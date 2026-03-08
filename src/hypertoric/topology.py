"""N-dimensional torus topology with flat↔coordinate conversion."""

from __future__ import annotations


class Topology:
    """N-dimensional torus grid with periodic boundary conditions.

    Provides flat↔coord conversion, neighbor computation, and plane selection.
    All flat indices are in range [0, num_blocks).
    """

    def __init__(self, ndim: int, grid_size: int) -> None:
        if ndim < 1:
            msg = f"ndim must be >= 1, got {ndim}"
            raise ValueError(msg)
        if grid_size < 2:
            msg = f"grid_size must be >= 2, got {grid_size}"
            raise ValueError(msg)

        self.ndim = ndim
        self.grid_size = grid_size
        self.strides = tuple(grid_size**a for a in range(ndim))
        self.num_blocks = grid_size**ndim
        self.num_neighbors = 2 * ndim

        # Precompute neighbor table: _neighbors[flat_idx][direction]
        self._neighbors: list[list[int]] = [
            [self._compute_neighbor(i, d) for d in range(self.num_neighbors)]
            for i in range(self.num_blocks)
        ]

    def flat_to_coord(self, flat_idx: int) -> tuple[int, ...]:
        """Convert flat index to N-dimensional coordinate tuple."""
        return tuple((flat_idx // s) % self.grid_size for s in self.strides)

    def coord_to_flat(self, coord: tuple[int, ...]) -> int:
        """Convert N-dimensional coordinate tuple to flat index."""
        return sum(c * s for c, s in zip(coord, self.strides, strict=True))

    def get_neighbor_flat(self, flat_idx: int, direction: int) -> int:
        """Return flat index of neighbor in given direction (precomputed)."""
        return self._neighbors[flat_idx][direction]

    def _compute_neighbor(self, flat_idx: int, direction: int) -> int:
        """Compute neighbor flat index for a given direction."""
        axis = direction // 2
        offset = 1 - 2 * (direction % 2)  # +1 for even, -1 for odd
        coord = list(self.flat_to_coord(flat_idx))
        coord[axis] = (coord[axis] + offset) % self.grid_size
        return self.coord_to_flat(tuple(coord))

    def get_plane(self, axis: int, position: int) -> list[int]:
        """Return flat indices of all blocks where coord[axis] == position."""
        if axis < 0 or axis >= self.ndim:
            msg = f"axis must be in [0, {self.ndim}), got {axis}"
            raise ValueError(msg)
        if position < 0 or position >= self.grid_size:
            msg = f"position must be in [0, {self.grid_size}), got {position}"
            raise ValueError(msg)
        return [
            i for i in range(self.num_blocks) if self.flat_to_coord(i)[axis] == position
        ]
