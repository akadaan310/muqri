"""Sifaat (articulation qualities): formant, HNR, spectral and timing analyzers.

Validators are registered in ``app.scoring`` (importing them here would create an import cycle
with ``app.tajweed_rules.base``, which uses ``app.sifaat.formants``).
"""
