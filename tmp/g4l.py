import numpy as np
import pandas as pd
import g4l

class SmallestMaximizerCriterion():
  champion_trees = []
  best_tree = None
  X = None
  max_depth = None

  def __init__(self, tree_generation_method, best_tree_selection_method):
    self.tree_generation_strategy = tree_generation_strategy
    self.best_tree_selection_strategy = best_tree_selection_strategy

  def fit(self, X, max_depth=4, dump_folder=None):
    self.X = X
    self.max_depth = max_depth
    context_tree = g4l.tree.ContextTree(X, max_depth=max_depth)
    self.champion_trees = self.tree_generation_strategy.execute(context_tree)
    self.best_tree = self.best_tree_selection_strategy(self)
    if dump_folder is not None:
      self.save(dump_folder)
    #bootstrap_dataframe = self.bootstrap(self.sample, partition_string)
    #t_result = self.t_test(bootstrap_dataframe)
    #tree_idx = int(list(t_result[t_result.pvalue < alpha].tree_idx)[0])
    #self.best_tree = champion_trees[tree_idx]
    return self

  def score(self, data):
    pass

  def t_test(self, df):
    df2 = pd.DataFrame(columns=['tree_idx', 'tvalue','pvalue', 'sd_sm', 'sd_lg'])
    for tree_idx in df[df.lml_delta_div_n.notnull()].tree_idx.unique():
      delta_lml_sm = df.loc[(df.sample_size=='sm') & (df.tree_idx == tree_idx)].lml_delta_div_n
      delta_lml_lg = df.loc[(df.sample_size=='lg') & (df.tree_idx == tree_idx)].lml_delta_div_n
      ttest = stats.ttest_ind(delta_lml_sm, delta_lml_lg)
      df2.loc[len(df2)] = [tree_idx, ttest.statistic, ttest.pvalue, delta_lml_sm.std(), delta_lml_lg.std()]
    return df2

  def generate_champion_trees(self, min_c, max_c, epsilon):
    smc = g4l.estimators.SMC(self.initial_tree)
    self.champion_trees = smc.execute(min_c, max_c, epsilon)

  def save(self, dump_folder):
    # TODO: add sample_file to model_folder
    for i, ch in enumerate(self.champion_trees):
        ch.save(dump_folder, 'champion.%s' % i)
    print('saved!')

  def load(self, dump_folder):
    # TODO: remove sample_file, it should be added when saving
    # TODO: load all configs
    smc = SmallestMaximizerCriterion(sample_file, max_depth=max_depth)
    smc.champion_trees = []
    amnt_trees = len(glob.glob(model_folder + '/champion*.df.pkl'))
    for i in range(amnt_trees):
        ch = g4l.tree.ContextTree.load(dump_folder, 'champion.%s' % i)
        smc.champion_trees.append(ch)
    return smc

# workaround to temporarily persist generated trees


# Initial data



#c = 0.5
#print("Setup...")
#sample = g4l.data.Sample('./g4l/fixtures/folha.txt', A)
#initial_tree = g4l.tree.ContextTree(sample, max_depth=max_depth)
#ctm = g4l.estimators.CTM(initial_tree)
#bic_tree = ctm.execute(c)
#print("BIC tree: ", bic_tree.to_str())

#smc = g4l.estimators.SMC(initial_tree)
#champion_trees = smc.execute(0.1, 400, 0.01) # min_c, max_c, epsilon

#dump_fld = '/home/arthur/Documents/Neuromat/code/SMC/euclid.aoas.1331043393/supplementary/SCRIPTS/python/g4l/fixtures/dump'
#g4l.util.persist_trees(dump_fld, bic_tree, champion_trees)



print("Done. Run test.py now.")
# A natureza tem horror do vazio
#import code; code.interact(local=dict(globals(), **locals()))
