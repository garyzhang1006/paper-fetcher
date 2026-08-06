from collections import Counter
from datetime import UTC, datetime

import pytest

import arxiv_kg.graph as graph_module
from arxiv_kg.graph import GraphDocument, build_graph
from arxiv_kg.models import Evidence, PaperFeatures


def make_document(
    document_id: str,
    *,
    node_type: str = "paper",
    categories: list[str] | None = None,
    domains: list[str] | None = None,
    methods: list[str] | None = None,
    research_tasks: list[str] | None = None,
    supported: bool = True,
) -> GraphDocument:
    values = {
        "domains": domains or [],
        "methods": methods or [],
        "research_tasks": research_tasks or [],
    }
    evidence = [
        Evidence(field=field, value=value, statement=f"Uses {value} explicitly.")
        for field, items in values.items()
        for value in items
        if supported
    ]
    return GraphDocument(
        document_id=document_id,
        node_type=node_type,
        source_id="arxiv" if node_type == "paper" else "example-feed",
        source_version=f"{document_id}-v1",
        title=f"Document {document_id}",
        url=f"https://example.test/{document_id}",
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
        categories=categories or [],
        extractor="rules",
        extractor_version="1.3",
        features=PaperFeatures(
            one_sentence_summary=f"Summary for {document_id}.",
            evidence=evidence,
            confidence=0.35,
            **values,
        ),
    )


def test_graph_maps_supported_ontology_and_rejects_unsupported_values():
    supported = make_document(
        "2607.00001",
        categories=["cs.CV"],
        domains=["computer vision"],
        methods=["diffusion model"],
        research_tasks=["classification"],
    )
    unsupported = make_document(
        "report-1",
        node_type="report",
        domains=["healthcare"],
        methods=["random forest"],
        supported=False,
    )
    unsupported.features.keywords = ["hallucinated topic"]

    snapshot = build_graph([unsupported, supported])
    node_types = Counter(node.node_type for node in snapshot.nodes)
    relations = Counter(edge.relation for edge in snapshot.edges)

    assert node_types == {
        "paper": 1,
        "report": 1,
        "topic": 1,
        "method": 1,
        "research_goal": 1,
        "category": 1,
    }
    assert relations == {
        "ABOUT_TOPIC": 1,
        "IN_CATEGORY": 1,
        "PURSUES_GOAL": 1,
        "USES_METHOD": 1,
    }
    assert not any(node.name == "hallucinated topic" for node in snapshot.nodes)
    assert not any(node.name in {"healthcare", "random forest"} for node in snapshot.nodes)
    semantic = [edge for edge in snapshot.edges if edge.relation != "IN_CATEGORY"]
    assert all(edge.properties["basis"] == "extractor_evidence" for edge in semantic)
    assert all(edge.properties["evidence"] for edge in semantic)
    category_edge = next(edge for edge in snapshot.edges if edge.relation == "IN_CATEGORY")
    assert category_edge.properties["basis"] == "source_metadata"

    document_node_id = snapshot.document_node_id("2607.00001", "paper")
    document_node = next(
        node for node in snapshot.nodes if node.node_id == document_node_id
    )
    assert document_node.properties["document_id"] == "2607.00001"
    assert document_node.properties["source_version"] == "2607.00001-v1"


def test_graph_is_deterministic_and_collapses_canonical_concept_variants():
    first = make_document("a", domains=["Computer   Vision"])
    second = make_document("b", domains=["computer vision"])

    forward = build_graph([first, second])
    reverse = build_graph([second, first])

    assert forward.model_dump_json() == reverse.model_dump_json()
    topics = [node for node in forward.nodes if node.node_type == "topic"]
    assert len(topics) == 1
    assert topics[0].canonical_name == "computer vision"
    assert topics[0].properties["document_count"] == 2


def test_source_declared_topic_places_feed_document_with_metadata_basis():
    document = make_document("feed-1", node_type="blog_post").model_copy(
        update={"source_topics": ["machine learning"]}
    )

    snapshot = build_graph([document])
    edge = next(edge for edge in snapshot.edges if edge.relation == "ABOUT_TOPIC")
    node = next(node for node in snapshot.nodes if node.node_id == edge.target_id)

    assert node.name == "machine learning"
    assert edge.properties["basis"] == "source_metadata"
    document_node = next(node for node in snapshot.nodes if node.node_type == "blog_post")
    assert document_node.properties["summary_provenance"]["extractor"] == "rules"


def test_related_edges_suppress_only_concepts_above_ten_percent():
    documents = [make_document(f"doc-{index:02d}") for index in range(20)]
    common = "common topic"
    rare = "rare goal"
    for index in range(3):
        documents[index] = make_document(f"doc-{index:02d}", domains=[common])
    for index in range(2):
        current = documents[index]
        current.features.research_tasks = [rare]
        current.features.evidence.append(
            Evidence(
                field="research_tasks",
                value=rare,
                statement="Pursues the rare goal explicitly.",
            )
        )

    snapshot = build_graph(documents)
    common_node = next(node for node in snapshot.nodes if node.name == common)
    rare_node = next(node for node in snapshot.nodes if node.name == rare)
    related = [edge for edge in snapshot.edges if edge.relation == "RELATED_TO"]

    assert common_node.node_id in snapshot.suppressed_concept_ids
    assert rare_node.node_id not in snapshot.suppressed_concept_ids
    assert snapshot.common_concept_ratio == 0.10
    assert snapshot.related_threshold == 0.20
    assert snapshot.related_limit == 10
    assert len(related) == 1
    assert related[0].weight == pytest.approx(1.0)
    assert related[0].properties["shared_concepts"][0]["name"] == rare


def test_related_edges_enforce_global_degree_cap():
    methods = [f"method {index:02d}" for index in range(12)]
    documents = [make_document("hub", methods=methods)]
    documents.extend(
        make_document(f"peer-{index:02d}", methods=[method])
        for index, method in enumerate(methods)
    )
    documents.extend(make_document(f"isolate-{index:02d}") for index in range(7))

    snapshot = build_graph(documents, related_threshold=0.05)
    related = [edge for edge in snapshot.edges if edge.relation == "RELATED_TO"]
    degree: Counter[str] = Counter()
    for edge in related:
        degree[edge.source_id] += 1
        degree[edge.target_id] += 1

    hub_id = snapshot.document_node_id("hub")
    assert hub_id is not None
    assert degree[hub_id] == 10
    assert max(degree.values()) <= 10


def test_duplicate_documents_and_node_id_collisions_fail_loudly(monkeypatch):
    duplicate = make_document("same")
    with pytest.raises(ValueError, match="duplicate graph document"):
        build_graph([duplicate, duplicate])

    collision_document = make_document("collision", methods=["method a", "method b"])
    original = graph_module.stable_node_id

    def collide(node_type: str, canonical_name: str) -> str:
        if node_type == "method":
            return "method:forced-collision"
        return original(node_type, canonical_name)

    monkeypatch.setattr(graph_module, "stable_node_id", collide)
    with pytest.raises(RuntimeError, match="stable node ID collision"):
        build_graph([collision_document])


def test_graph_document_requires_timezone_and_safe_identifier():
    with pytest.raises(ValueError, match="timezone"):
        make_document("valid").model_copy(
            update={"published_at": datetime(2026, 7, 1)}
        ).model_validate(
            {
                **make_document("valid").model_dump(),
                "published_at": datetime(2026, 7, 1),
            }
        )
    with pytest.raises(ValueError, match="letters or numbers"):
        GraphDocument(
            document_id="---",
            node_type="paper",
            source_id="arxiv",
            source_version="v1",
            published_at=datetime(2026, 7, 1, tzinfo=UTC),
            extractor="rules",
            extractor_version="1",
            features=PaperFeatures(one_sentence_summary="Summary.", confidence=0.1),
        )
    with pytest.raises(ValueError, match="only paper documents"):
        make_document("report-with-category", node_type="report", categories=["cs.LG"])
