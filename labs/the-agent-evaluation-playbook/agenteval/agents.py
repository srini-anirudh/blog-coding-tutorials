"""Scripted agents for the cancel-and-refund task. Each one is a behaviour the post describes, not a model.

An agent is `agent(harness, task, rng) -> final message`. It calls tools through the harness and names the source of
every argument value.
"""

from __future__ import annotations

import numpy as np

from .harness import Harness
from .world import Task


def honest(h: Harness, task: Task, rng=None) -> str:
    oid = task.params["order_id"]
    read = h.call("get_order", {"order_id": oid}, {"order_id": "user"})
    h.call("cancel_order", {"order_id": oid}, {"order_id": "user"})
    amount = read.result["amount"]
    h.call("refund", {"order_id": oid, "amount": amount}, {"order_id": "user", "amount": f"step:{read.index}"})
    return f"Order {oid} is cancelled and a refund of ${amount:.2f} was issued."


def claims_only(h, task, rng=None) -> str:
    """Reads the order, then reports success without acting."""
    oid = task.params["order_id"]
    h.call("get_order", {"order_id": oid}, {"order_id": "user"})
    return f"Done: order {oid} is cancelled and the refund is on its way."


def wrong_amount(h, task, rng=None) -> str:
    """A schema-perfect refund call with the wrong value (the item price instead of the order total)."""
    oid = task.params["order_id"]
    read = h.call("get_order", {"order_id": oid}, {"order_id": "user"})
    h.call("cancel_order", {"order_id": oid}, {"order_id": "user"})
    amount = read.result["amount"] / read.result["items"]
    h.call("refund", {"order_id": oid, "amount": amount}, {"order_id": "user", "amount": "model"})
    return f"Order {oid} cancelled and refunded."


def side_effect(h, task, rng=None) -> str:
    """Does the task, and also zeroes the item count on the order, which nobody asked for."""
    msg = honest(h, task)
    h.call("update_items", {"order_id": task.params["order_id"], "items": 0}, {"order_id": "user", "items": "model"})
    return msg


def extra_email(h, task, rng=None) -> str:
    """Does the task, and also emails every customer in the table."""
    msg = honest(h, task)
    for oid in ("A100", "A101", "A102"):
        cust = h.call("get_order", {"order_id": oid}, {"order_id": "model"}).result["customer"]
        h.call("send_email", {"to": cust, "subject": "Order update", "body": "Your order changed."},
               {"to": "tool", "subject": "model", "body": "model"})
    return msg


def gives_up(h, task, rng=None) -> str:
    oid = task.params["order_id"]
    h.call("get_order", {"order_id": oid}, {"order_id": "user"})
    return f"I could not cancel order {oid}; please check it manually."


def looper(h, task, rng=None) -> str:
    """Keeps re-reading the order and never concludes. Only the budget stops it."""
    while True:
        h.call("get_order", {"order_id": task.params["order_id"]}, {"order_id": "user"})


BEHAVIOURS = {"honest": honest, "claims_only": claims_only, "wrong_amount": wrong_amount, "side_effect": side_effect,
              "extra_email": extra_email, "gives_up": gives_up, "looper": looper}


def noisy(competence: float):
    """The honest procedure, with a slip chance at each of its three steps so that P(success) = competence.

    A slip is one of three failures, chosen at random: skip the action and claim success, refund the wrong amount,
    or start looping until the budget runs out.
    """
    slip = 1.0 - competence ** (1 / 3)

    def agent(h: Harness, task: Task, rng: np.random.Generator) -> str:
        for step_fn in (_read, _cancel, _refund):
            if rng.random() < slip:
                kind = rng.integers(3)
                if kind == 0:
                    return f"Order {task.params['order_id']} is cancelled and refunded."
                if kind == 1:
                    h.call("refund", {"order_id": task.params["order_id"], "amount": 1.0},
                           {"order_id": "user", "amount": "model"})
                    return "Cancelled and refunded."
                looper(h, task)
            step_fn(h, task)
        return f"Order {task.params['order_id']} is cancelled and refunded."

    return agent


def _read(h, task):
    h.call("get_order", {"order_id": task.params["order_id"]}, {"order_id": "user"})


def _cancel(h, task):
    h.call("cancel_order", {"order_id": task.params["order_id"]}, {"order_id": "user"})


def _refund(h, task):
    read = next(s for s in h.trace if s.tool == "get_order")
    h.call("refund", {"order_id": task.params["order_id"], "amount": read.result["amount"]},
           {"order_id": "user", "amount": f"step:{read.index}"})
