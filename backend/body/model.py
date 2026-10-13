"""Thin wrapper around the Anny parametric body model.

Only the rest (A-pose) shape is needed for sizing, so we skip skinning entirely and
evaluate the linear blend shapes directly. This is ~100x cheaper than a full forward pass
and keeps everything differentiable for the measurement fitter.
"""

from __future__ import annotations

import functools

import torch
import anny
from anny.skinning import skinning

# Phenotypes the fitter is allowed to move. Gender is fixed by the user and age is fixed
# to an adult value, the rest are free in [0, 1].
FREE_PHENOTYPES = ["muscle", "weight", "height", "proportions"]

# Anny's default age anchors are [newborn, baby, child, young, old] at
# linspace(-1/3, 1, 5), so 2/3 is "young adult" (~25y) and 1.0 is "old" (~90y).
ADULT_AGE = 0.72

# MakeHuman local modifiers that matter for clothing fit. Restricting the fitter to these
# keeps the problem well posed and stops it from "fitting" measurements with faces or toes.
FIT_LOCAL_CHANGES = [
    "measure-bust-circ-incr",
    "measure-underbust-circ-incr",
    "measure-waist-circ-incr",
    "measure-hips-circ-incr",
    "measure-shoulder-dist-incr",
    "measure-neck-circ-incr",
    "measure-upperarm-length-incr",
    "measure-lowerarm-length-incr",
    "measure-upperleg-height-incr",
    "measure-lowerleg-height-incr",
    "measure-napetowaist-dist-incr",
    "measure-waisttohip-dist-incr",
    "measure-upperarm-circ-incr",
    "measure-thigh-circ-incr",
    "torso-scale-depth-incr",
    "torso-scale-horiz-incr",
    "torso-vshape-incr",
    "stomach-pregnant-incr",
    "buttocks-volume-incr",
    "hip-scale-horiz-incr",
    "hip-scale-depth-incr",
    "breast-volume-vert-up",
    "neck-scale-horiz-incr",
    "neck-scale-depth-incr",
]


class BodyModel(torch.nn.Module):
    """Maps (phenotypes, local changes) to rest-pose vertices and bone heads, in metres, Z up."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.anny = anny.Anny(local_changes="default").to(dtype=dtype)
        self.anny.eval()
        self.faces = self.anny.faces.cpu().numpy()
        self.phenotype_labels = list(self.anny.phenotype_labels)
        self.local_change_labels = list(self.anny.local_change_labels)
        self.bone_labels = list(self.anny.bone_labels)
        self.fit_local_indices = torch.tensor(
            [self.local_change_labels.index(k) for k in FIT_LOCAL_CHANGES]
        )
        self.dtype = dtype

    def bone_index(self, label: str) -> int:
        return self.bone_labels.index(label)

    def phenotype_tensor(self, values: dict[str, torch.Tensor | float]) -> torch.Tensor:
        """Stack a {label: (B,) tensor or float} dict into a (B, P) tensor, 0.5 by default."""
        batch = 1
        for v in values.values():
            if torch.is_tensor(v) and v.dim() > 0:
                batch = v.shape[0]
        cols = []
        for label in self.phenotype_labels:
            v = values.get(label, 0.5)
            v = torch.as_tensor(v, dtype=self.dtype)
            cols.append(v.expand(batch) if v.dim() == 0 else v)
        return torch.stack(cols, dim=1)

    def local_tensor(self, fit_values: torch.Tensor) -> torch.Tensor:
        """Scatter (B, len(FIT_LOCAL_CHANGES)) values into the full (B, L) local-change tensor."""
        full = torch.zeros(
            fit_values.shape[0], len(self.local_change_labels), dtype=self.dtype
        )
        return full.index_copy(1, self.fit_local_indices, fit_values)

    def forward(self, phenotypes: torch.Tensor, local_changes: torch.Tensor):
        """phenotypes: (B, P) in [0,1]; local_changes: (B, L) full local-change tensor."""
        facial = torch.zeros(phenotypes.shape[0], 0, dtype=self.dtype)
        coeffs = self.anny._get_phenotype_blendshape_coefficients(
            phenotypes, local_changes, facial
        )
        vertices = self.anny.get_rest_vertices(coeffs)
        bone_heads = skinning.apply_linear_blendshape(
            self.anny.template_bone_heads, self.anny.bone_heads_blendshapes, coeffs
        )
        # Put the feet on the floor (z = 0) so heights read as heights above ground.
        floor = vertices[..., 2].min(dim=1).values
        offset = torch.zeros(vertices.shape[0], 1, 3, dtype=self.dtype)
        offset[..., 2] = floor[:, None]
        return vertices - offset, bone_heads - offset


class FastBody(torch.nn.Module):
    """Exact, fast re-expression of BodyModel for one gender at adult age.

    Anny's vertices are (a) a product of piecewise-linear weights over each phenotype, i.e.
    piecewise *multilinear* in (muscle, weight, height, proportions), plus (b) a sum of
    independent local-change blend shapes split into +/- halves. So we can precompute the
    meshes at every phenotype anchor corner (3x3x2x2 = 36) and the 2x24 local shapes, then
    evaluate with two small einsums. Anny's generic path loops over all 254 local changes in
    Python and is ~100x slower.
    """

    def __init__(self, model: BodyModel, gender: float):
        super().__init__()
        a = model.anny
        self.model = model
        self.anchors = {k: a.anchors[k].clone() for k in FREE_PHENOTYPES}
        sizes = [len(self.anchors[k]) for k in FREE_PHENOTYPES]
        grid = torch.cartesian_prod(*[self.anchors[k] for k in FREE_PHENOTYPES])

        zeros_local = torch.zeros(len(grid), len(model.local_change_labels), dtype=model.dtype)
        values = {"gender": torch.full((len(grid),), gender), "age": torch.full((len(grid),), ADULT_AGE)}
        values.update({k: grid[:, i] for i, k in enumerate(FREE_PHENOTYPES)})
        with torch.no_grad():
            facial = torch.zeros(len(grid), 0, dtype=model.dtype)
            coeffs = a._get_phenotype_blendshape_coefficients(
                model.phenotype_tensor(values), zeros_local, facial
            )
            corner_v = a.get_rest_vertices(coeffs)
            corner_j = skinning.apply_linear_blendshape(
                a.template_bone_heads, a.bone_heads_blendshapes, coeffs
            )
        assert corner_v.shape[0] == int(torch.tensor(sizes).prod())

        # Local changes occupy the last 2 * L coefficient slots as (+, -) pairs.
        offset = a.blendshapes.shape[0] - 2 * len(model.local_change_labels)
        idx = offset + 2 * model.fit_local_indices
        bs_v, bs_j = a.blendshapes, a.bone_heads_blendshapes
        # One (36 + 2 * 24, V * 3) basis so a forward pass is a single matrix product.
        basis_v = torch.cat([corner_v, bs_v[idx], bs_v[idx + 1]])
        basis_j = torch.cat([corner_j, bs_j[idx], bs_j[idx + 1]])
        self.register_buffer("basis_v", basis_v.reshape(basis_v.shape[0], -1).contiguous())
        self.register_buffer("basis_j", basis_j.reshape(basis_j.shape[0], -1).contiguous())

    def _weights(self, phenotypes: torch.Tensor) -> list[torch.Tensor]:
        from anny.utils.interpolation import linear_interpolation_coefficients

        return [
            linear_interpolation_coefficients(phenotypes[i : i + 1], self.anchors[k])[0]
            for i, k in enumerate(FREE_PHENOTYPES)
        ]

    def forward(self, phenotypes: torch.Tensor, local_changes: torch.Tensor):
        """phenotypes: (4,) values for FREE_PHENOTYPES; local_changes: (len(FIT_LOCAL_CHANGES),).

        Returns rest vertices (V, 3) and bone heads (J, 3), feet on the floor.
        """
        wm, ww, wh, wp = self._weights(phenotypes)
        corner_w = torch.einsum("a,b,c,d->abcd", wm, ww, wh, wp).reshape(-1)
        # Same split as Anny: a ReLU whose gradient at 0 is 1, otherwise local changes that
        # start at 0 never receive a gradient.
        x = local_changes
        local_w = torch.cat([x * (x >= 0).to(x.dtype), -x * (x < 0).to(x.dtype)])
        weights = torch.cat([corner_w, local_w])
        v = (weights @ self.basis_v).reshape(-1, 3)
        j = (weights @ self.basis_j).reshape(-1, 3)
        floor = torch.zeros(3, dtype=v.dtype)
        floor[2] = v[:, 2].min()
        return v - floor, j - floor


@functools.cache
def fast_body(gender: float) -> FastBody:
    return FastBody(load_body_model(), gender)


@functools.cache
def load_body_model() -> BodyModel:
    """The first call parses MakeHuman assets (minutes on a cold cache), later calls are fast."""
    return BodyModel()
