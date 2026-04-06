"""
example_usage.py — Demonstrates the TrifectaClient multi-modal RAG pipeline.

Ingests text documents and (optionally) an image, builds KG edges, then
queries using text-only and fused multi-modal retrieval.

Usage:
    python example_usage.py

    To test image ingestion, place a JPEG at ./sample.jpg before running.
"""

from pathlib import Path

from trifecta_client import TrifectaClient
import trifecta_py as tr


def main() -> None:
    print("=" * 60)
    print("  TrifectaRAG — Multi-Modal Retrieval Demo")
    print("=" * 60)

    # ── Initialize ────────────────────────────────────────────────────────
    client = TrifectaClient(device="cpu")
    print(f"\nClient ready: {client}\n")

    # ── Ingest text documents ─────────────────────────────────────────────
    id0 = client.add_document(
        text=(
            "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France."
        ),
        metadata={"source": "wikipedia", "topic": "landmarks"},
    )
    print(f"  Ingested text doc  -> gid={id0}")

    id1 = client.add_document(
        text=(
            "Quantum computing leverages qubits and superposition to "
            "solve problems intractable for classical computers."
        ),
        metadata={"source": "textbook", "topic": "computing"},
    )
    print(f"  Ingested text doc  -> gid={id1}")

    id2 = client.add_document(
        text=(
            "The Louvre Museum in Paris houses the Mona Lisa and "
            "thousands of other works of art."
        ),
        metadata={"source": "travel_guide", "topic": "landmarks"},
    )
    print(f"  Ingested text doc  -> gid={id2}")

    # ── Optionally ingest an image ────────────────────────────────────────
    sample_img = Path("sample.jpg")
    id3 = None

    if sample_img.exists():
        id3 = client.add_image(
            image=sample_img,
            caption="A photograph of the Eiffel Tower illuminated at night",
            metadata={"source": "photo_album", "location": "Paris"},
        )
        print(f"  Ingested image     -> gid={id3}")
    else:
        print(f"\n  [Skipping image — place a JPEG at {sample_img.resolve()} to test]")

    # ── Build knowledge graph edges ───────────────────────────────────────
    client.add_edge(id0, id2, tr.EdgeType.RELATES_TO)
    print(f"\n  KG edge: gid {id0} --RELATES_TO--> gid {id2}")
    if id3 is not None:
        client.add_edge(id0, id3, tr.EdgeType.DEPICTS)
        print(f"  KG edge: gid {id0} --DEPICTS--> gid {id3}")

    # ── Query: text-only ──────────────────────────────────────────────────
    print("\n--- Text-only query: 'Eiffel Tower in Paris' ---")
    results = client.query(text="Eiffel Tower in Paris", top_k=5)
    for gid, score in results:
        print(f"  gid={gid:3d}  score={score:.6f}")

    # ── Query: multi-modal (text + image) ─────────────────────────────────
    if id3 is not None and sample_img.exists():
        print("\n--- Multi-modal query: text='Paris landmarks' + image ---")
        results = client.query(text="Paris landmarks", image=sample_img, top_k=5)
        for gid, score in results:
            print(f"  gid={gid:3d}  score={score:.6f}")

    print(f"\nFinal state: {client}")
    print("Done.")


if __name__ == "__main__":
    main()
