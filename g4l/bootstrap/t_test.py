import pandas as pd
import logging
from scipy import stats


class TTest():

    results = None
    best_tree_idx = None

    def __init__(self, resample_set_1, resample_set_2, alpha=0.01):
        self.resample_set_1 = resample_set_1
        self.resample_set_2 = resample_set_2
        self.alpha = alpha
        self.results = None
        self.best_tree_idx = None

    def evaluate(self, champion_trees):
        logging.debug("Evaluating using T-test")
        df = pd.DataFrame(columns=['sample_idx', 'tree_idx', 'c', 'resample_label', 'num_contexts', 'tree', 'lml', 'lml_next', 'lml_delta', 'lml_delta_div_n'])
        self.add_resamples_to_dataframe(self.resample_set_1, 'SET1', df, champion_trees)
        self.add_resamples_to_dataframe(self.resample_set_2, 'SET2', df, champion_trees)
        self.results = self.apply_t_test(df, 'SET1', 'SET2')
        self.best_tree_idx = int(list(self.results[self.results.pvalue < self.alpha].tree_idx)[0])

    def add_resamples_to_dataframe(self, resample_set, label, df, champion_trees):
        for resample_idx, resample in resample_set.generate():
            for tree_idx, t in enumerate(champion_trees):
                log_likelihood = t.evaluate_sample(resample).log_likelihood()
                self.__add_row(df, resample_idx, tree_idx, t, label, log_likelihood)
        self.calc_delta(df, label, resample_set.size)

    def calc_delta(self, df, resample_label, sample_length):
        df.sort_values(['resample_label', 'sample_idx', 'c'], inplace=True)
        for resample_idx in range(len(df.sample_idx.unique())):
            condition = (df.resample_label==resample_label) & (df.sample_idx==resample_idx)
            df.loc[condition, 'lml_next'] = (list(df.loc[condition].lml)[1:] + [float('NaN')])
            df.loc[condition, 'lml_delta'] = (df.loc[condition].lml - df.loc[condition].lml_next)
            df.loc[condition, 'lml_delta_div_n'] = df.loc[condition].lml_delta / sample_length**0.9

    def apply_t_test(self, df, resample_set_1_lbl, resample_set_2_lbl):
        df2 = pd.DataFrame(columns=['tree_idx', 'tvalue','pvalue', 'sd_set1', 'sd_set2'])
        for tree_idx in df[df.lml_delta_div_n.notnull()].tree_idx.unique():
            delta_lml_sm = df.loc[(df.resample_label==resample_set_1_lbl) & (df.tree_idx == tree_idx)].lml_delta_div_n
            delta_lml_lg = df.loc[(df.resample_label==resample_set_2_lbl) & (df.tree_idx == tree_idx)].lml_delta_div_n
            ttest = stats.ttest_ind(delta_lml_sm, delta_lml_lg)
            df2.loc[len(df2)] = [tree_idx, ttest.statistic, ttest.pvalue, delta_lml_sm.std(), delta_lml_lg.std()]
        return df2

    def __add_row(self, df, resample_idx, tree_idx, t, resample_label, log_likelihood):
        tree_str = t.to_str()
        num_contexts = t.num_contexts()
        logging.debug("Adding row [%s; %s; %s; %s]" %  (resample_idx, resample_label, num_contexts, log_likelihood))
        df.loc[len(df)] = [resample_idx, tree_idx, t.chosen_penalty,
                                             resample_label, num_contexts,
                                             tree_str, log_likelihood, None, None, None]
