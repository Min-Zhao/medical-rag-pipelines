"""
Knowledge Graph Builder
=======================
Constructs a biomedical knowledge graph from CLA documents.
Nodes represent entities (diseases, drugs, genes, symptoms, biomarkers).
Edges represent relationships (TREATS, CAUSES, ASSOCIATED_WITH, etc.).

Dependencies:
    pip install networkx spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from .document_processor import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLA-specific entity dictionary for rule-based NER
# ---------------------------------------------------------------------------

CLA_ENTITIES: dict[str, list[str]] = {
    "DISEASE": [
        "gorham-stout disease", "GSD", "vanishing bone disease", "massive osteolysis",
        "generalized lymphatic anomaly", "GLA",
        "central conducting lymphatic anomaly", "CCLA",
        "kaposiform lymphangiomatosis", "KLA",
        "lymphangioleiomyomatosis", "LAM",
        "complex lymphatic anomaly", "CLA",
        "chylothorax", "chylous ascites",
        "lymphatic malformation", "vascular anomaly",
        "tuberous sclerosis", "noonan syndrome",
        "pulmonary lymphatic perfusion syndrome", "PLPS",
        "protein-losing enteropathy",
    ],
    "DRUG": [
        "sirolimus", "rapamycin",
        "everolimus", "RAD001",
        "trametinib", "cobimetinib", "binimetinib",
        "alpelisib", "copanlisib",
        "bevacizumab",
        "bisphosphonate", "zoledronate", "pamidronate",
        "vincristine",
        "octreotide",
        "interferon-alpha", "interferon",
        "bleomycin", "doxycycline",
    ],
    "GENE": [
        "PIK3CA", "PIK3R1", "AKT1", "MTOR",
        "TSC1", "TSC2",
        "NRAS", "KRAS", "HRAS",
        "PTPN11", "SOS1", "RAF1", "RIT1", "BRAF",
        "VEGFR-3", "VEGFR3",
        "VEGF-C", "VEGFC",
        "VEGF-D", "VEGFD", "VEGF-A",
        "mTORC1", "mTORC2",
        "MEK1", "MEK2", "ERK1", "ERK2",
        "PROX1", "LYVE-1", "D2-40",
        "IL-6", "IL6",
    ],
    "SYMPTOM": [
        "dyspnea", "shortness of breath",
        "pleural effusion",
        "bone pain", "osteolysis",
        "thrombocytopenia", "low platelets",
        "coagulopathy", "bleeding",
        "fatigue",
        "lymphedema",
        "chylous effusion",
        "splenomegaly",
        "pneumothorax",
        "hemoptysis",
    ],
    "BIOMARKER": [
        "VEGF-D", "VEGF-C",
        "angiopoietin-2", "Ang-2",
        "D-dimer", "fibrinogen",
        "triglycerides",
        "FEV1", "FEV₁",
    ],
    "PROCEDURE": [
        "MRI", "CT scan", "X-ray", "PET scan",
        "DCMRL", "dynamic contrast-enhanced MR lymphangiography", "MR lymphangiography",
        "biopsy", "thoracentesis",
        "thoracic duct embolization", "TDE",
        "pleurodesis",
        "lung transplantation",
        "sclerotherapy",
    ],
    "PATHWAY": [
        "PI3K/AKT/mTOR", "mTOR pathway", "PI3K pathway",
        "RAS/MAPK", "MAPK/ERK", "MAPK pathway",
    ],
}

# Canonical name for normalization
ENTITY_ALIASES: dict[str, str] = {
    "rapamycin": "sirolimus",
    "rad001": "everolimus",
    "vanishing bone disease": "gorham-stout disease",
    "massive osteolysis": "gorham-stout disease",
    "vegfr3": "VEGFR-3",
    "vegfc": "VEGF-C",
    "vegfd": "VEGF-D",
    "il6": "IL-6",
    "tde": "thoracic duct embolization",
    "dcmrl": "MR lymphangiography",
    "plps": "pulmonary lymphatic perfusion syndrome",
}

RELATIONSHIP_PATTERNS: list[tuple[str, str, str]] = [
    # (trigger phrase, relation type, direction hint)
    (r"treat(?:s|ed|ment for)", "TREATS", "forward"),
    (r"used (?:to treat|for)", "TREATS", "forward"),
    (r"first.line (?:therapy|treatment) for", "TREATS", "forward"),
    (r"approved for", "TREATS", "forward"),
    (r"inhibit(?:s|or of)", "INHIBITS", "forward"),
    (r"target(?:s|ing)", "TARGETS", "forward"),
    (r"cause(?:s|d by)", "CAUSES", "forward"),
    (r"associated with", "ASSOCIATED_WITH", "bidirectional"),
    (r"manifest(?:s|ation of)", "MANIFESTS_AS", "forward"),
    (r"diagnos(?:ed|is) (?:of|by|with)", "DIAGNOSED_BY", "forward"),
    (r"upregulat(?:es|ed)", "UPREGULATES", "forward"),
    (r"downregulat(?:es|ed)", "DOWNREGULATES", "forward"),
    (r"mutation in", "MUTATION_IN", "forward"),
    (r"express(?:es|ed)", "EXPRESSES", "forward"),
    (r"biomarker for", "BIOMARKER_FOR", "forward"),
]


@dataclass
class Entity:
    name: str
    entity_type: str
    canonical: str = ""
    doc_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.canonical:
            self.canonical = ENTITY_ALIASES.get(self.name.lower(), self.name)


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    doc_id: str
    sentence: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Rule-based entity extractor
# ---------------------------------------------------------------------------

class RuleBasedNER:
    """Identifies CLA-domain entities using a curated dictionary."""

    def __init__(self):
        self._patterns: dict[str, re.Pattern] = {}
        for entity_type, terms in CLA_ENTITIES.items():
            escaped = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
            self._patterns[entity_type] = re.compile(
                r"\b(" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )

    def extract(self, text: str, doc_id: str = "") -> list[Entity]:
        entities: list[Entity] = []
        seen: set[str] = set()
        for entity_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                name = match.group(0)
                canonical = ENTITY_ALIASES.get(name.lower(), name)
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    entities.append(Entity(
                        name=name,
                        entity_type=entity_type,
                        canonical=canonical,
                        doc_ids=[doc_id] if doc_id else [],
                    ))
        return entities


# ---------------------------------------------------------------------------
# Relation extractor
# ---------------------------------------------------------------------------

class RuleBasedRelationExtractor:
    """Extracts entity pairs connected by linguistic triggers."""

    def __init__(self, window: int = 80):
        self.window = window
        self._rel_patterns = [
            (re.compile(pattern, re.IGNORECASE), rel_type)
            for pattern, rel_type, _ in RELATIONSHIP_PATTERNS
        ]

    def extract(
        self,
        text: str,
        entities: list[Entity],
        doc_id: str = "",
    ) -> list[Relation]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relations: list[Relation] = []

        for sentence in sentences:
            ents_in_sent = [
                e for e in entities
                if re.search(re.escape(e.canonical), sentence, re.IGNORECASE)
                or re.search(re.escape(e.name), sentence, re.IGNORECASE)
            ]
            if len(ents_in_sent) < 2:
                continue

            for i, e1 in enumerate(ents_in_sent):
                for e2 in ents_in_sent[i + 1 :]:
                    for rel_pattern, rel_type in self._rel_patterns:
                        if rel_pattern.search(sentence):
                            relations.append(Relation(
                                source=e1.canonical,
                                target=e2.canonical,
                                relation_type=rel_type,
                                doc_id=doc_id,
                                sentence=sentence[:200],
                                confidence=0.8,
                            ))
                            break

        return relations


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Builds and queries a NetworkX knowledge graph over CLA documents.

    Example:
        builder = KnowledgeGraphBuilder()
        builder.build_from_documents(documents)
        results = builder.query_entity_neighborhood("sirolimus", depth=2)
        context = builder.get_context_for_query("What does sirolimus treat?")
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.ner = RuleBasedNER()
        self.rel_extractor = RuleBasedRelationExtractor()
        self._entity_map: dict[str, Entity] = {}

    def build_from_documents(self, documents: list[Document]) -> None:
        """Extract entities and relations from all documents."""
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for doc in documents:
            text = doc.full_text or doc.abstract
            entities = self.ner.extract(text, doc_id=doc.doc_id)
            relations = self.rel_extractor.extract(text, entities, doc_id=doc.doc_id)
            all_entities.extend(entities)
            all_relations.extend(relations)

        self._add_entities_to_graph(all_entities)
        self._add_relations_to_graph(all_relations)

        logger.info(
            "Knowledge graph: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _add_entities_to_graph(self, entities: list[Entity]) -> None:
        for ent in entities:
            canonical = ent.canonical
            if canonical not in self._entity_map:
                self._entity_map[canonical] = ent
                self.graph.add_node(
                    canonical,
                    entity_type=ent.entity_type,
                    doc_ids=ent.doc_ids[:],
                )
            else:
                existing = self.graph.nodes[canonical]
                existing_docs = existing.get("doc_ids", [])
                for d in ent.doc_ids:
                    if d not in existing_docs:
                        existing_docs.append(d)

    def _add_relations_to_graph(self, relations: list[Relation]) -> None:
        for rel in relations:
            if rel.source in self.graph and rel.target in self.graph:
                if self.graph.has_edge(rel.source, rel.target):
                    existing = self.graph[rel.source][rel.target]
                    existing["doc_ids"] = list(set(existing.get("doc_ids", []) + [rel.doc_id]))
                    existing["count"] = existing.get("count", 1) + 1
                else:
                    self.graph.add_edge(
                        rel.source,
                        rel.target,
                        relation_type=rel.relation_type,
                        doc_ids=[rel.doc_id],
                        sentences=[rel.sentence],
                        count=1,
                        confidence=rel.confidence,
                    )

    def query_entity_neighborhood(
        self, entity_name: str, depth: int = 2
    ) -> dict[str, Any]:
        """Return all nodes within `depth` hops of the given entity."""
        canonical = ENTITY_ALIASES.get(entity_name.lower(), entity_name)
        if canonical not in self.graph:
            matches = [n for n in self.graph.nodes if entity_name.lower() in n.lower()]
            if not matches:
                return {"entity": entity_name, "nodes": [], "edges": []}
            canonical = matches[0]

        subgraph_nodes = set()
        subgraph_nodes.add(canonical)
        frontier = {canonical}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier

        sub = self.graph.subgraph(subgraph_nodes)
        nodes = [
            {"name": n, "type": sub.nodes[n].get("entity_type", "UNKNOWN")}
            for n in sub.nodes
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "relation": sub[u][v].get("relation_type", "RELATED_TO"),
                "count": sub[u][v].get("count", 1),
            }
            for u, v in sub.edges
        ]
        return {"entity": canonical, "nodes": nodes, "edges": edges}

    def get_context_for_query(
        self, query: str, max_triples: int = 15
    ) -> str:
        """
        Identify query entities then retrieve their graph neighborhood
        as a formatted string for LLM context injection.
        """
        entities = self.ner.extract(query)
        if not entities:
            return ""

        triples: list[str] = []
        seen_triples: set[tuple] = set()

        for ent in entities:
            neighborhood = self.query_entity_neighborhood(ent.canonical, depth=1)
            for edge in neighborhood["edges"]:
                triple = (edge["source"], edge["relation"], edge["target"])
                if triple not in seen_triples:
                    seen_triples.add(triple)
                    triples.append(f"({edge['source']}) --[{edge['relation']}]--> ({edge['target']})")
                if len(triples) >= max_triples:
                    break
            if len(triples) >= max_triples:
                break

        if not triples:
            return ""

        return "Knowledge Graph Context:\n" + "\n".join(triples)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Knowledge graph saved to %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        logger.info(
            "Knowledge graph loaded: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "entity_type_counts": dict(
                defaultdict(
                    int,
                    {
                        t: sum(1 for _, d in self.graph.nodes(data=True) if d.get("entity_type") == t)
                        for t in {"DISEASE", "DRUG", "GENE", "SYMPTOM", "BIOMARKER", "PROCEDURE", "PATHWAY"}
                    }
                )
            ),
            "top_degree_nodes": sorted(
                [(n, self.graph.degree(n)) for n in self.graph.nodes],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
