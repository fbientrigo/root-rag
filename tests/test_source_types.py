from root_rag.retrieval.models import classify_source_type


def test_source_type_classification_expected_paths():
    assert classify_source_type("README.md") == "doc"
    assert classify_source_type("macro/ShipReco.py") == "macro"
    assert classify_source_type("python/shipVeto.py") == "python"
    assert classify_source_type("muonDIS/makeMuonDIS.py") == "python"
    assert classify_source_type("shipgen/MuDISGenerator.cxx") == "cpp"
    assert classify_source_type("data/indexes_fairship/run/fts.sqlite") == "artifact"


def test_source_type_precedence_artifact_over_doc_and_code():
    assert classify_source_type("data/processed/README.md") == "artifact"
    assert classify_source_type("data/indexes_fairship/macro/ShipReco.py") == "artifact"


def test_source_type_precedence_macro_over_python():
    assert classify_source_type("macro/run_simScript.py") == "macro"
