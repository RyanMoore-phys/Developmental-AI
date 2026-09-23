"""Object slots: instances, relations and persistence.

A LIGHT PACKAGE, like `sensors` and `spatial`, so these are importable
without the Minecraft stack.

UNTESTED AS SHIPPED — deliberately. Every contract that would matter here
needs the Wave 1/2 go/no-go first: slots decode the FLOW FIELD, so if the
flow head has not been shown to learn anything on live frames, a slot model
trained on it is a slot model trained on noise, and a green contract would
only certify that the plumbing runs. SAVi's own stated limitation is that
fully-unsupervised decomposition "still fails to scale to diverse realistic
data"; that is a warning about exactly this situation.
"""
from .attention import SlotAttention, SAViSlots
from .relations import slot_relations, RELATION_FIELDS

__all__ = ["SlotAttention", "SAViSlots", "slot_relations", "RELATION_FIELDS"]
