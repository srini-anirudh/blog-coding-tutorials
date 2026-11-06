"""Tool retrieval is RAG over actions, and retrieval recall caps task success.

A synthetic catalogue: every tool is (domain, verb, object), e.g. ``billing_update_invoice``, with a templated
description and JSON schema. A task needs three tools and is written the way a user would write it: with synonyms
("modify the bill in accounts") that the tool descriptions may not share. That vocabulary gap is the thing that makes
retrieval miss.

Retrievers return the top k tools for a query by cosine similarity of TF-IDF vectors (numpy only):
    whole      : one query, the whole request
    decomposed : the harness splits the request into its three clauses and retrieves ceil(k/3) per clause
A task is feasible only if all three needed tools are in the retrieved set: P(success) <= P(T* subset of R_k(q)).
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

VERBS = {"get": ["fetch", "retrieve", "look up"], "create": ["add", "make", "open"], "update": ["modify", "change",
         "edit"], "delete": ["remove", "erase", "drop"], "list": ["enumerate", "show all", "browse"], "search": ["find",
         "query", "locate"], "send": ["dispatch", "email", "deliver"], "cancel": ["void", "revoke", "abort"],
         "approve": ["accept", "sign off", "authorize"], "export": ["download", "dump", "extract"],
         "assign": ["allocate", "hand over", "delegate"], "archive": ["shelve", "store away", "retire"]}
OBJECTS = {"invoice": ["bill", "statement"], "customer": ["client", "account holder"], "order": ["purchase",
           "sale"], "ticket": ["issue", "case"], "event": ["meeting", "appointment"], "file": ["document",
           "attachment"], "user": ["member", "person"], "report": ["summary", "analysis"], "payment": ["transaction",
           "charge"], "message": ["note", "chat"], "task": ["todo", "work item"], "contract": ["agreement",
           "deal"], "shipment": ["delivery", "parcel"], "product": ["item", "sku"], "refund": ["reimbursement",
           "chargeback"], "invoice_line": ["line item", "billing row"], "project": ["initiative", "programme"],
           "lead": ["prospect", "opportunity"], "alert": ["notification", "warning"], "dataset": ["table", "data"]}
DOMAINS = ["billing", "crm", "support", "calendar", "storage", "hr", "analytics", "payments", "chat", "tasks",
           "legal", "logistics", "catalog", "finance", "projects", "sales", "monitoring", "warehouse", "marketing",
           "security", "inventory", "payroll", "travel", "procurement", "compliance", "education", "health",
           "realestate", "insurance", "telecom", "energy", "media", "gaming", "auctions", "banking", "tax",
           "recruiting", "facilities", "fleet", "events", "nonprofit", "retail", "kitchen", "library", "parking",
           "hotel", "airline", "school", "clinic", "lab"]
DOMAIN_SYN = {d: [d, f"{d} system", f"our {d} app"] for d in DOMAINS}


@dataclass(frozen=True)
class Tool:
    domain: str
    verb: str
    obj: str

    @property
    def name(self) -> str:
        return f"{self.domain}_{self.verb}_{self.obj}"

    def description(self) -> str:
        return f"{self.verb.capitalize()} a {self.obj.replace('_', ' ')} in the {self.domain} service."

    def schema(self) -> str:
        props = {f"{self.obj}_id": {"type": "string", "description": f"ID of the {self.obj.replace('_', ' ')}"},
                 "fields": {"type": "object", "description": "Fields to set or filter on"},
                 "dry_run": {"type": "boolean", "description": "Validate without applying"}}
        return json.dumps({"name": self.name, "description": self.description(),
                           "parameters": {"type": "object", "properties": props, "required": [f"{self.obj}_id"]}})


def catalogue(size: int, seed: int = 0) -> list[Tool]:
    combos = [Tool(d, v, o) for d in DOMAINS for v in VERBS for o in OBJECTS]
    return random.Random(seed).sample(combos, size)


def phrase(word: str, table: dict, rng: random.Random, p_synonym: float) -> str:
    return rng.choice(table[word]) if rng.random() < p_synonym else word.replace("_", " ")


def make_task(tools: list[Tool], rng: random.Random, p_synonym: float = 0.5) -> tuple[list[Tool], list[str]]:
    """Three tools a workflow needs, mostly from one domain, and the request as three clauses."""
    first = rng.choice(tools)
    same = [t for t in tools if t.domain == first.domain and t != first]
    need = [first] + rng.sample(same, min(2, len(same)))
    while len(need) < 3:
        t = rng.choice(tools)
        if t not in need:
            need.append(t)
    clauses = [f"{phrase(t.verb, VERBS, rng, p_synonym)} the {phrase(t.obj, OBJECTS, rng, p_synonym)} in "
               f"{rng.choice(DOMAIN_SYN[t.domain]) if rng.random() < p_synonym else t.domain}" for t in need]
    return need, clauses


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("_", " "))


class TfidfIndex:
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        docs = [words(t.name + " " + t.description()) for t in tools]
        df = Counter(w for d in docs for w in set(d))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        self.idf = np.array([math.log(len(docs) / df[w]) + 1 for w in sorted(df)])
        self.matrix = np.stack([self.embed_words(d) for d in docs])

    def embed_words(self, ws: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(ws).items():
            if w in self.vocab:
                v[self.vocab[w]] = c
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def top_k(self, text: str, k: int) -> list[int]:
        scores = self.matrix @ self.embed_words(words(text))
        return list(np.argsort(-scores, kind="stable")[:k])

    def retrieve(self, clauses: list[str], k: int, decomposed: bool) -> set[int]:
        if not decomposed:
            return set(self.top_k(" and then ".join(clauses), k))
        per = math.ceil(k / len(clauses))
        return {i for c in clauses for i in self.top_k(c, per)}


def evaluate(index: TfidfIndex, k: int, decomposed: bool, n_tasks: int, seed: int) -> dict:
    rng = random.Random(seed)
    pos = {t: i for i, t in enumerate(index.tools)}
    per_tool, joint = [], []
    for _ in range(n_tasks):
        need, clauses = make_task(index.tools, rng)
        got = index.retrieve(clauses, k, decomposed)
        hits = [pos[t] in got for t in need]
        per_tool.append(sum(hits) / 3)
        joint.append(all(hits))
    r = sum(per_tool) / n_tasks
    return {"per_tool_recall": r, "all_three_retrieved": sum(joint) / n_tasks, "independence_r3": r ** 3}
