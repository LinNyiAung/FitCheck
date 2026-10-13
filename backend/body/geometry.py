"""Mesh slicing helpers.

The mesh topology never changes, only vertex positions do. So we split every measurement
into two steps:

1. *Planning* (numpy, no gradients): decide which mesh edges a plane cuts, which connected
   piece is the torso (or neck), and which cut points lie on the convex hull (a tape
   measure bridges concavities, e.g. between the breasts).
2. *Evaluation* (torch, differentiable): rebuild those cut points as fixed linear
   combinations of two vertices each and sum the hull edge lengths.

Between re-plans the cut points ride along with the surface, like a painted tape line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

Z_UP = np.array([0.0, 0.0, 1.0])


def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Indices of the 2D convex hull, counter-clockwise (Andrew's monotone chain).

    scipy's ConvexHull goes through qhull, which creates a temp file per call on Windows
    and is ~50x slower than this for the few hundred points a body slice has.
    """
    order = np.lexsort((pts[:, 1], pts[:, 0]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(indices):
        chain: list[int] = []
        for i in indices:
            while len(chain) >= 2 and cross(pts[chain[-2]], pts[chain[-1]], pts[i]) <= 0:
                chain.pop()
            chain.append(i)
        return chain

    lower, upper = half(order), half(order[::-1])
    return np.array(lower[:-1] + upper[:-1])


def hull_perimeter(pts: np.ndarray) -> float:
    p = pts[convex_hull_2d(pts)]
    return float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).sum())


def plane_basis(normal: np.ndarray) -> np.ndarray:
    """(2, 3) orthonormal in-plane axes; for a horizontal plane these are exactly X and Y."""
    n = normal / np.linalg.norm(normal)
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u)])


@dataclass
class Loop:
    """A tape line: point i = v[a_i] + t_i * (v[b_i] - v[a_i]), in order along the tape."""

    a: np.ndarray
    b: np.ndarray
    t: np.ndarray
    closed: bool = True

    def points(self, vertices: torch.Tensor) -> torch.Tensor:
        """vertices: (V, 3) -> (N, 3)."""
        va, vb = vertices[self.a], vertices[self.b]
        t = torch.as_tensor(self.t, dtype=vertices.dtype)[:, None]
        return va + t * (vb - va)

    def length(self, vertices: torch.Tensor) -> torch.Tensor:
        p = self.points(vertices)
        if self.closed:
            return torch.linalg.norm(p - torch.roll(p, 1, dims=0), dim=-1).sum()
        return torch.linalg.norm(p[1:] - p[:-1], dim=-1).sum()


@dataclass
class Slice:
    """Cut points of one closed curve on the plane (origin, normal)."""

    a: np.ndarray
    b: np.ndarray
    t: np.ndarray
    points: np.ndarray  # (N, 3)
    uv: np.ndarray  # (N, 2) in-plane coordinates

    def hull_loop(self) -> Loop:
        order = convex_hull_2d(self.uv)
        return Loop(a=self.a[order], b=self.b[order], t=self.t[order])

    def back_arc(self) -> Loop:
        """Taut tape across the back of a horizontal slice, from its left-most (+X) to its
        right-most (-X) point. In-plane axes are (X, Y) and +Y is the back, so this is the
        counter-clockwise hull chain starting at max X."""
        order = convex_hull_2d(self.uv)
        start = int(np.argmax(self.uv[order, 0]))
        order = np.roll(order, -start)
        end = int(np.argmin(self.uv[order, 0]))
        order = order[: end + 1]
        return Loop(a=self.a[order], b=self.b[order], t=self.t[order], closed=False)

    def girth(self) -> float:
        return hull_perimeter(self.uv)


class MeshSlicer:
    def __init__(self, faces: np.ndarray):
        faces = np.asarray(faces, dtype=np.int64)
        all_edges = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        all_edges.sort(axis=1)
        self.edges, inverse = np.unique(all_edges, axis=0, return_inverse=True)
        # face_edges[f] = the 3 edge ids of face f
        self.face_edges = inverse.reshape(3, -1).T

    def slices(
        self, vertices: np.ndarray, origin: np.ndarray, normal: np.ndarray = Z_UP
    ) -> list[Slice]:
        """Cut the mesh with a plane. Returns one Slice per closed curve."""
        d = (vertices - origin) @ normal
        d = np.where(d == 0, 1e-9, d)  # avoid degenerate on-plane vertices
        da, db = d[self.edges[:, 0]], d[self.edges[:, 1]]
        crossing = (da * db) < 0
        if not crossing.any():
            return []
        edge_ids = np.flatnonzero(crossing)
        local = -np.ones(len(self.edges), dtype=np.int64)
        local[edge_ids] = np.arange(len(edge_ids))
        # A face cut by the plane has exactly two crossing edges: link them.
        cut = crossing[self.face_edges]
        cut_faces = self.face_edges[cut.sum(axis=1) == 2]
        pairs = local[cut_faces[crossing[cut_faces]].reshape(-1, 2)]
        n = len(edge_ids)
        adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(adj, directed=False)

        a, b = self.edges[edge_ids, 0], self.edges[edge_ids, 1]
        t = da[edge_ids] / (da[edge_ids] - db[edge_ids])
        pts = vertices[a] + t[:, None] * (vertices[b] - vertices[a])
        uv = (pts - origin) @ plane_basis(normal).T
        return [
            Slice(a[m], b[m], t[m], pts[m], uv[m])
            for m in (labels == k for k in range(labels.max() + 1))
        ]

    def central_slice(
        self,
        vertices: np.ndarray,
        origin: np.ndarray,
        normal: np.ndarray = Z_UP,
        max_offset: float = 0.06,
    ) -> Slice | None:
        """The largest closed curve centred on the mid-sagittal plane (x = 0).

        "Largest" matters at neck height, where small inner-mouth pieces are also centred.
        Arms and legs sit well off-centre, so the offset test excludes them.
        """
        best, best_size = None, 0.0
        for s in self.slices(vertices, origin, normal):
            if len(s.t) < 8 or abs(s.points[:, 0].mean()) > max_offset:
                continue
            size = np.ptp(s.uv[:, 0]) + np.ptp(s.uv[:, 1])
            if size > best_size:
                best, best_size = s, size
        return best

    def torso_at(self, vertices: np.ndarray, h: float) -> Slice | None:
        return self.central_slice(vertices, np.array([0.0, 0.0, h]))
