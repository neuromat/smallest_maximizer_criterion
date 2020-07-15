class Base():

  def __init__(self, context_tree):
    self.context_tree = context_tree
    self.sample = self.context_tree.sample
    self.data = self.sample.data
    self.A = self.sample.A
