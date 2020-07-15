def persist_trees(dump_fld, bic_tree, champion_trees):
  bic_tree.save(dump_fld, 'bic_tree')
  #to load: bic_tree = g4l.tree.ContextTree.load(dump_fld, 'bic_tree')
  for i, ch in enumerate(champion_trees):
    ch.save(dump_fld, 'champion.%s' % i)
  print('BIC tree and champion trees saved')
