import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Dict, Callable, Optional
from pathlib import Path
import numpy as np
import faiss

from src.methods.GNN.schemas import ExtractedEntity
from src.methods.GNN.utils import (
    calculate_term_similarity,
    calculate_overlap_neighbors_similarity,
    calculate_l2_dist,
)
from src.utils.config_loader import ConfigLoader

_GNN_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "resources" / "config.gnn.yaml"
_cfg = ConfigLoader(str(_GNN_CONFIG_PATH))

_THRESHOLD_TERM       = _cfg.get("inter_entity_threshold.term_similarity", 0.0)
_THRESHOLD_KNN        = _cfg.get("inter_entity_threshold.common_neighbors_similarity", 0.0)
_THRESHOLD_SENIORITY  = _cfg.get("inter_entity_threshold.l2_dist", 0.0)


class CJMDataset(Dataset):
    def __init__(
        self, 
        data_list: List[Tuple[ExtractedEntity, ExtractedEntity, int]], 
        embedder: Callable[[str], torch.Tensor],
        skill_index: Optional[faiss.Index] = None,
        seniority_map: Dict[str, int] = None,
        k_neighbors: int = 10,
        p_sharpening: float = 4.0
    ):
        """
        Args:
            data_list: List of (CV, JD, Label) tuples.
            embedder: A function that takes a string and returns a 1D Tensor (e.g., 768 dims).
                      Used for building node features.
            skill_index: A FAISS index of the skill universe for kNN lookups.
            seniority_map: Dict mapping seniority text to integers (e.g. {"junior": 1, "senior": 3}).
            k_neighbors: Number of neighbors for kNN overlap calculation.
            p_sharpening: Sharpening exponent for overlap similarity.
        """
        super().__init__()
        self.data_list = data_list
        self.embedder = embedder
        self.skill_index = skill_index
        self.k = k_neighbors
        self.p = p_sharpening

        # Default ordinal mapping if none provided
        self.seniority_map = seniority_map or {
            "intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 5
        }

    @property
    def _embed_fn(self):
        """Adapts self.embedder (str → Tensor) to the EmbedFn (List[str] → np.ndarray) interface."""
        def _fn(texts: List[str]) -> np.ndarray:
            return torch.stack([self.embedder(t) for t in texts]).numpy()
        return _fn

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        cv, jd, label = self.data_list[idx]

        # Primary Nodes (Macro Context)
        cv_primary = self.embedder(cv.model_dump_json())
        jd_primary = self.embedder(jd.model_dump_json())


        def repr_list(lst: List[str]) -> torch.Tensor:
            if not lst:
                zero = torch.zeros_like(cv_primary)
                return torch.cat([zero, zero, zero])
            stacked = torch.stack([self.embedder(s) for s in lst])  # [N, dim]
            mean_ = stacked.mean(dim=0)
            sum_  = stacked.sum(dim=0)
            max_  = stacked.max(dim=0).values
            return torch.cat([mean_, sum_, max_])                    # [3*dim]

        def repr_term(s: str) -> torch.Tensor:
            e = self.embedder(s)                                     # [dim]
            return torch.cat([e, e, e])                              # [3*dim]

        cv_ent = [
            repr_term(cv.title),
            repr_list(cv.tech_stack),
            repr_list(cv.soft_skills),
            repr_term(cv.domain),
            repr_term(cv.seniority),
        ]

        jd_ent = [
            repr_term(jd.title),
            repr_list(jd.tech_stack),
            repr_list(jd.soft_skills),
            repr_term(jd.domain),
            repr_term(jd.seniority),
        ]

        # x_primary shape: [2, dim]
        # x_entities shape: [10, dim] (5 for CV, 5 for JD)
        x_primary = torch.stack([cv_primary, jd_primary])
        x_entities = torch.stack(cv_ent + jd_ent)

        # Category 0: Title (Cosine Similarity)
        w_title = calculate_term_similarity(cv.title, jd.title, self._embed_fn)

        # Category 1: Tech Stack (kNN Overlap)
        w_tech = calculate_overlap_neighbors_similarity(
            cv.tech_stack, jd.tech_stack, self.skill_index, self._embed_fn,
            k=self.k, sharpening_factor=self.p
        )

        # Category 2: Soft Skills (kNN Overlap)
        w_soft = calculate_overlap_neighbors_similarity(
            cv.soft_skills, jd.soft_skills, self.skill_index, self._embed_fn,
            k=self.k, sharpening_factor=self.p
        )

        # Category 3: Domain (Cosine Similarity)
        w_domain = calculate_term_similarity(cv.domain, jd.domain, self._embed_fn)

        # Category 4: Seniority (Ordinal L1 Distance → converted to similarity)
        w_seniority = 1.0 - calculate_l2_dist(
            cv.seniority, jd.seniority, self.seniority_map, normalize=True
        )

        semantic_weights = [w_title, w_tech, w_soft, w_domain, w_seniority]

        # Node indices map:

        
        # 0: CV Primary, 1: JD Primary
        # 2..6: CV Entities, 7..11: JD Entities
        
        edges = []
        weights = []

        def add_undirected_edge(u, v, weight):
            edges.append([u, v])
            edges.append([v, u])
            weights.extend([weight, weight])

        # A. Structural Edges (Weight = 1.0)
        add_undirected_edge(0, 1, 1.0)  # Cand <-> Job
        
        for i in range(5):
            add_undirected_edge(0, i + 2, 1.0)  # Cand <-> Cand Entities
            add_undirected_edge(1, i + 7, 1.0)  # Job <-> Job Entities

        # B. Semantic Edges (gated by inter_entity_threshold from config.gnn.yaml)
        # Order matches: [title, tech, soft, domain, seniority]
        thresholds = [_THRESHOLD_TERM, _THRESHOLD_KNN, _THRESHOLD_KNN, _THRESHOLD_TERM, _THRESHOLD_SENIORITY]
        for i, (w, threshold) in enumerate(zip(semantic_weights, thresholds)):
            if w > threshold:
                add_undirected_edge(i + 2, i + 7, w)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)

        return Data(
            x_primary=x_primary, 
            x_entities=x_entities, 
            edge_index=edge_index, 
            edge_weight=edge_weight, 
            y=torch.tensor([label], dtype=torch.float)
        )