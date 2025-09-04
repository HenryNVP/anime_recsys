"""
ALS baseline using the 'implicit' library (Hu et al., KDD 2008).
Install: pip install implicit
"""
from typing import Optional
import numpy as np

try:
    from implicit.als import AlternatingLeastSquares
except Exception:
    AlternatingLeastSquares = None


class ALSWrapper:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 10,
        random_state: Optional[int] = 42
    ):
        if AlternatingLeastSquares is None:
            raise ImportError("Install 'implicit' to use ALS: pip install implicit")
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state
        )

    def fit(self, ui_csr):
        """
        ui_csr: scipy.sparse CSR matrix of shape (num_users, num_items),
                implicit preferences (1 for observed interactions).
        implicit expects item-user, so we pass transpose inside fit.
        """
        self.model.fit(ui_csr.T, show_progress=False)

    def score_user_candidates(self, u: int, candidates: np.ndarray) -> np.ndarray:
        # fast dot-product using learned factors
        return self.model.item_factors[candidates] @ self.model.user_factors[u]
