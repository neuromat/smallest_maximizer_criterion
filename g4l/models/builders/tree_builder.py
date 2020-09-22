class ContextTreeBuilder:
  def __init__(self, A):
    self.A = A
    self.contexts = []

  def add_context(self, context, transition_probs):
    self.contexts.append((context, transition_probs))

  def build(self):
    pass


