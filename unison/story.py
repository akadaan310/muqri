"""Finite game-theoretic conversational *fiction* grammar with replay.

A miniature proof-of-structure: narrative generators can be explicit
state machines, not ungrounded psychological profiling of their users.
Characters, consequences and options are fictional; source language packs
may later render dialogue without pretending to know translations.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal

Action = Literal["PAY", "REQUEST_TAB", "ASK_FRIEND", "LEAVE_UNPAID", "RESOLVE"]


@dataclass(frozen=True, slots=True)
class FictionState:
    credit: int
    bill: int
    trust: int = 0
    location: str = "cafe"
    episode: str = "BILL_PENDING"
    generation: int = 0

    def __post_init__(self):
        if self.credit < 0 or self.bill < 0 or not -10 <= self.trust <= 10:
            raise ValueError("Invalid finite story state")
        if self.generation < 0:
            raise ValueError("Negative generation")


@dataclass(frozen=True, slots=True)
class FictionTurn:
    before: FictionState
    action: Action
    after: FictionState
    description: str
    status: str = "fictional_structural_event"


def permitted(state: FictionState) -> tuple[Action, ...]:
    if state.episode == "RESOLVED":
        return ()
    if state.episode == "BILL_PENDING":
        return (("PAY",) if state.credit >= state.bill else ()) + (
            "REQUEST_TAB", "ASK_FRIEND", "LEAVE_UNPAID"
        )
    if state.episode == "DEFERRED":
        return ("PAY", "RESOLVE") if state.credit >= state.bill else ("RESOLVE",)
    return ("RESOLVE",)


def advance(state: FictionState, action: Action) -> FictionTurn:
    if action not in permitted(state):
        raise ValueError("Action not admitted from current state")
    if action == "PAY":
        updated = replace(state, credit=state.credit-state.bill, bill=0, trust=min(10,state.trust+1),
                          episode="RESOLVED",generation=state.generation+1)
        text = "The character pays the bill."
    elif action == "REQUEST_TAB":
        updated = replace(state,trust=min(10,state.trust+1), episode="DEFERRED",
                          generation=state.generation+1)
        text = "The character asks whether the cafe can extend a tab."
    elif action == "ASK_FRIEND":
        updated = replace(state, episode="DEFERRED", generation=state.generation+1)
        text = "The character calls a friend and requests help."
    elif action == "LEAVE_UNPAID":
        updated = replace(state,trust=max(-10,state.trust-3),episode="CONSEQUENCES",
                          generation=state.generation+1)
        text = "The character leaves without paying; trust and social consequences change."
    else:
        updated = replace(state,episode="RESOLVED",generation=state.generation+1)
        text = "The characters resolve the situation through a negotiated outcome."
    return FictionTurn(state,action,updated,text)
