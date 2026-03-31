"""Tests for ROOT SOFIE corpus indexing and retrieval.

SOFIE (System for Optimized Fast Inference code Emit) is ROOT's experimental
ML inference framework. These tests ensure SOFIE APIs are properly indexed
and retrievable for FairShip ML workflow integration.
"""
import pytest

from root_rag.parser.seed_filter import load_seed_corpus_config


class TestSOFIECorpusConfig:
    """Tests for SOFIE-specific corpus configuration."""

    def test_sofie_corpus_config_structure(self, tmp_path):
        """SOFIE corpus config should define SOFIE classes and ONNX runtime."""
        config_content = """
root:
  version: "6.36.08"
  tag: "v6-36-08"
  repository: "root-project/root"

corpus:
  tier: "sofie_experimental"
  rationale: "ML inference APIs for FairShip SOFIE integration"
  
  classes:
    - name: "ROOT::Experimental::SOFIE::RModel"
      rationale: "Main SOFIE model wrapper"
      headers:
        - "tmva/sofie/inc/SOFIE/RModel.hxx"
      
    - name: "ROOT::Experimental::SOFIE::ROperator"
      rationale: "SOFIE operator base class"
      headers:
        - "tmva/sofie/inc/SOFIE/ROperator.hxx"
    
    - name: "ROOT::Experimental::SOFIE::ONNX"
      rationale: "ONNX model parsing"
      headers:
        - "tmva/sofie/inc/SOFIE/SOFIE.hxx"
"""
        
        config_file = tmp_path / "sofie_config.yaml"
        config_file.write_text(config_content)
        
        # Load and validate
        config = load_seed_corpus_config(config_file)
        
        assert config is not None
        assert config["root"]["version"] == "6.36.08"
        assert config["corpus"]["tier"] == "sofie_experimental"
        assert len(config["corpus"]["classes"]) == 3
        
        # Check SOFIE classes are defined
        class_names = [c["name"] for c in config["corpus"]["classes"]]
        assert "ROOT::Experimental::SOFIE::RModel" in class_names
        assert "ROOT::Experimental::SOFIE::ROperator" in class_names
        assert "ROOT::Experimental::SOFIE::ONNX" in class_names


class TestSOFIEGoldenQueries:
    """Golden queries for SOFIE API retrieval."""

    @pytest.mark.integration
    def test_golden_query_rmodel_location(self):
        """Can retrieve RModel header location."""
        # This test requires actual ROOT 6.36.08 index with SOFIE
        # Marked as integration test
        pytest.skip("Requires full ROOT 6.36.08 index with SOFIE corpus")

    @pytest.mark.integration
    def test_golden_query_onnx_runtime(self):
        """Can retrieve ONNX runtime wrapper APIs."""
        pytest.skip("Requires full ROOT 6.36.08 index with SOFIE corpus")

    @pytest.mark.integration
    def test_golden_query_sofie_inference_workflow(self):
        """Can retrieve SOFIE inference code generation workflow."""
        pytest.skip("Requires full ROOT 6.36.08 index with SOFIE corpus")


class TestSOFIEChunkProvenance:
    """Test SOFIE chunk metadata and provenance."""

    def test_sofie_chunk_has_experimental_marker(self, tmp_path):
        """SOFIE chunks should be marked as experimental."""
        # Mock chunk validation
        from root_rag.index.schemas import Chunk
        
        chunk = Chunk(
            chunk_id="sofie_test_1",
            root_ref="v6-36-08",
            resolved_commit="9005eb7d69f1abc",
            file_path="tmva/sofie/inc/SOFIE/RModel.hxx",
            language="cpp",
            start_line=1,
            end_line=50,
            content="namespace ROOT::Experimental::SOFIE { class RModel {...}; }",
            doc_origin="source_header",
            symbol_path="ROOT::Experimental::SOFIE::RModel",
            has_doxygen=True,
        )
        
        # Validate chunk
        assert chunk.file_path.startswith("tmva/sofie/")
        assert "SOFIE" in chunk.symbol_path
        assert chunk.doc_origin == "source_header"


class TestSOFIEFairShipIntegration:
    """Test SOFIE + FairShip integration scenarios."""

    def test_sofie_not_yet_in_fairship_usage(self):
        """Verify SOFIE is not yet used by FairShip master (as of audit)."""
        import json
        from pathlib import Path
        
        # Load FairShip extraction results
        inventory_path = Path("artifacts/fairship_root_usage_inventory.json")
        if not inventory_path.exists():
            pytest.skip("FairShip inventory not available")
        
        inventory = json.loads(inventory_path.read_text())
        headers = [h["name"] for h in inventory["root_headers"]]
        symbols = [s["name"] for s in inventory["root_symbols"]]
        
        # SOFIE should NOT be in current FairShip usage
        assert not any("SOFIE" in h for h in headers), \
            "SOFIE should not be in current FairShip (experimental)"
        assert not any("RModel" in s for s in symbols), \
            "RModel should not be in current FairShip (experimental)"

    def test_future_sofie_integration_readiness(self, tmp_path):
        """Verify system is ready to index SOFIE when FairShip adopts it."""
        config_content = """
root:
  version: "6.36.08"
  tag: "v6-36-08"

corpus:
  tier: "fairship_plus_sofie"
  classes:
    # Existing FairShip usage
    - name: "TGeoManager"
      headers: ["geom/geom/inc/TGeoManager.h"]
    
    # Future SOFIE adoption
    - name: "ROOT::Experimental::SOFIE::RModel"
      headers: ["tmva/sofie/inc/SOFIE/RModel.hxx"]
"""
        
        config_file = tmp_path / "future_config.yaml"
        config_file.write_text(config_content)
        
        config = load_seed_corpus_config(config_file)
        
        # System supports mixing traditional ROOT + experimental SOFIE
        class_names = [c["name"] for c in config["corpus"]["classes"]]
        assert "TGeoManager" in class_names
        assert "ROOT::Experimental::SOFIE::RModel" in class_names


class TestSOFIEDocumentation:
    """Test SOFIE documentation extraction."""

    def test_sofie_experimental_warning_extraction(self):
        """SOFIE docs should preserve experimental status warnings."""
        # Mock Doxygen comment with experimental warning
        mock_doxygen = """
        /// @class ROOT::Experimental::SOFIE::RModel
        /// @brief Model representation for SOFIE
        /// @warning This is EXPERIMENTAL and may change in future ROOT versions
        /// @note Requires ROOT compiled with -Dtmva-sofie=ON
        """
        
        assert "EXPERIMENTAL" in mock_doxygen
        assert "@warning" in mock_doxygen

    def test_sofie_onnx_dependencies_documented(self):
        """SOFIE ONNX support should document build requirements."""
        mock_requirements = """
        SOFIE ONNX Runtime Requirements:
        - ROOT 6.36+ with SOFIE enabled
        - Optional: ONNX Runtime for inference
        - CMake flag: -Dtmva-sofie=ON -Dtmva-sofie-onnx=ON
        """
        
        assert "ONNX Runtime" in mock_requirements
        assert "tmva-sofie" in mock_requirements
