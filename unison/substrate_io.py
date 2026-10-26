"""Classical bit/register and simulated quantum basis states remain typed apart.

Neither data structure is a model of the actual universe. Quantum amplitudes
below are a finite classical state-vector representation; measurement is a
lossy instrument, NOT a reversible bit converter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class BitWord:
    width: int
    bits: int

    def __post_init__(self):
        if type(self.width) is not int or not 1 <= self.width <= 64:
            raise ValueError("Bit width must be 1..64")
        if type(self.bits) is not int or not 0 <= self.bits < (1 << self.width):
            raise ValueError("Bit value exceeds declared width")

    def display(self) -> str:
        return format(self.bits, f"0{self.width}b")

    def flip(self, bit_index_lsb0: int) -> "BitWord":
        if not 0 <= bit_index_lsb0 < self.width:
            raise ValueError("Bit index out of range; index 0 is LSB")
        return BitWord(self.width, self.bits ^ (1 << bit_index_lsb0))

    def assign(self, bit_index_lsb0: int, value: int) -> "BitWord":
        if not 0 <= bit_index_lsb0 < self.width or value not in (0, 1):
            raise ValueError("Invalid assignment")
        mask = 1 << bit_index_lsb0
        return BitWord(self.width, (self.bits | mask) if value else (self.bits & ~mask))


@dataclass(frozen=True, slots=True)
class MaskTransition:
    before: BitWord
    after: BitWord
    operation: str
    generation: int

    def __post_init__(self):
        if self.before.width != self.after.width or self.generation < 0:
            raise ValueError("Transition width/generation mismatch")


# Exact owner's control path. Numeric successor is NOT inferred from masks.
MOTIF = ("0000", "1000", "1100", "1110", "1000")
OPERATIONS = ("ACTIVATE", "GROW", "GROW", "RETURN")


def qitt_control_trace() -> tuple[MaskTransition, ...]:
    return tuple(
        MaskTransition(BitWord(4, int(MOTIF[i], 2)),
                       BitWord(4, int(MOTIF[i+1], 2)),
                       OPERATIONS[i], 1 if i == 3 else 0)
        for i in range(len(OPERATIONS))
    )


@dataclass(frozen=True, slots=True)
class QuantumStateVector:
    """An ideal normalized vector in a DECLARED basis/endianness convention."""

    logical_qubits: int
    basis_order: str
    amplitudes: tuple[complex, ...]
    source: str

    def __post_init__(self):
        if type(self.logical_qubits) is not int or not 1 <= self.logical_qubits <= 8:
            raise ValueError("Bounded simulation allows 1..8 qubits")
        if self.basis_order not in {"q0_msb", "q0_lsb"}:
            raise ValueError("Declare qubit basis ordering")
        if len(self.amplitudes) != (1 << self.logical_qubits) or not self.source:
            raise ValueError("Incorrect basis dimension or missing model source")
        if any(not(math.isfinite(z.real) and math.isfinite(z.imag)) for z in self.amplitudes):
            raise ValueError("Nonfinite amplitude")
        norm = sum(abs(z)**2 for z in self.amplitudes)
        if abs(norm - 1.0) > 1e-9:
            raise ValueError("State is not normalized")

    def probabilities(self) -> tuple[float, ...]:
        """Born probabilities, not an observed measurement outcome."""
        return tuple(float(abs(z)**2) for z in self.amplitudes)


@dataclass(frozen=True, slots=True)
class HybridObservation:
    classical_control: BitWord
    quantum: QuantumStateVector | None
    clock_generation: int
    classical_source_id: str
    quantum_source_id: str | None = None

    def __post_init__(self):
        if self.clock_generation < 0 or not self.classical_source_id:
            raise ValueError("Missing provenance or generation")
        if (self.quantum is None) != (self.quantum_source_id is None):
            raise ValueError("Quantum representation requires independent source identity")

    def as_record(self) -> dict:
        return {
            "classical": {"bit_word": self.classical_control.display(),
                          "endianness": "display_msb_left; operators_lsb0",
                          "source_id": self.classical_source_id},
            "quantum": None if self.quantum is None else {
                "qubits": self.quantum.logical_qubits,
                "basis_order": self.quantum.basis_order,
                "born_probabilities_simulated": self.quantum.probabilities(),
                "source_id": self.quantum_source_id,
            },
            "clock_generation": self.clock_generation,
            "no_implicit_binary_quantum_conversion": True,
        }
