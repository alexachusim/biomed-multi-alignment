"""
tests/test_plugin.py
---------------------
Unit tests for the mammal plugin.
These tests do NOT require a GPU or a running vLLM server.
"""

import pytest


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class TestPluginRegistration:
    def test_register_function_exists(self):
        from vllm_mammal_plugin import register_mammal_model
        assert callable(register_mammal_model)


# ---------------------------------------------------------------------------
# MAMMAL prompt formatting helpers (used in examples / application code)
# ---------------------------------------------------------------------------

def format_mammal_prompt(sequence: str, modality: str) -> str:
    """Reference implementation of MAMMAL prompt formatting.
    
    Note: The working example uses pre-formatted strings with <EOS> tokens.
    This helper is for testing the formatting logic only.
    """
    modality = modality.lower()
    if modality in ("protein", "aa", "amino_acid"):
        return (
            f"<@TOKENIZER-TYPE=AA>"
            f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_PROTEIN>"
            f"{sequence.strip()}"
            f"<EOS>"
        )
    if modality in ("smiles", "small_molecule", "drug"):
        return (
            f"<@TOKENIZER-TYPE=SMILES>"
            f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>"
            f"{sequence.strip()}"
            f"<EOS>"
        )
    return sequence


class TestPromptFormatting:
    def test_protein_prefix(self):
        result = format_mammal_prompt("ACDEFG", "protein")
        assert result.startswith("<@TOKENIZER-TYPE=AA>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in result
        assert "ACDEFG" in result
        assert result.endswith("<EOS>")

    def test_smiles_prefix(self):
        result = format_mammal_prompt("CC(=O)O", "smiles")
        assert result.startswith("<@TOKENIZER-TYPE=SMILES>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in result
        assert result.endswith("<EOS>")

    def test_text_passthrough(self):
        seq = "raw gene expression data"
        assert format_mammal_prompt(seq, "text") == seq

    def test_protein_aliases(self):
        for alias in ("aa", "amino_acid"):
            result = format_mammal_prompt("AAA", alias)
            assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in result
            assert result.endswith("<EOS>")

    def test_smiles_aliases(self):
        for alias in ("small_molecule", "drug"):
            result = format_mammal_prompt("CC", alias)
            assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in result
            assert result.endswith("<EOS>")

    def test_whitespace_stripped(self):
        result = format_mammal_prompt("  ACDEFG  ", "protein")
        assert "ACDEFG<EOS>" in result
