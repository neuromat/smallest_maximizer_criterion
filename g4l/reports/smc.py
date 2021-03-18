import glob
import os
import yaml
import numpy as np
from g4l.models import ContextTree
from g4l.display import plot2
import matplotlib.pyplot as plt
import pandas as pd


class SmcReport:
    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        self.reports_folder = os.path.join(self.folder, 'reports')
        self.champion_trees = []
        self.thresholds = np.load(os.path.join(self.folder, 'bic_c.npy'))
        #self.load_summary()

    def generate(self):
        os.makedirs(self.reports_folder, exist_ok=True)
        self.champion_trees_report()

    def champion_trees_report(self):
        self.load_trees()
        ct = self.optimal_tree()
        txt = '\n'.join([t.to_str(reverse=True) for t in self.champion_trees])
        trees_file = os.path.join(self.reports_folder, 'trees.txt')
        open(trees_file, 'w').write(txt + '\n')
        #import code; code.interact(local=dict(globals(), **locals()))
        open(trees_file, 'a').write('\n\nOptimal:\n %s\n' % ct.to_str(reverse=True) )

    def trees_dataframe(self, n=10):
        self.load_trees()
        df = pd.DataFrame(columns=['c', 'ctxs', 'tree'])
        for i, t in enumerate(list(reversed(self.champion_trees))):
            df.loc[len(df)] = [self.thresholds[i], t.num_contexts(), t.to_str(reverse=True)]
        return df.head(n)

    def draw_likelihoods_boxplot(self, n_sizes, n=10):
        trees = list(reversed(self.champion_trees))[:n]
        plt.figure(figsize=(24, 18))

        for i, t in enumerate(trees):
            #qlml1 = df[(df.tree_idx==i) & (df.sample_size=='sm')].lml_delta_div_n
            #qlml2 = df[(df.tree_idx==i) & (df.sample_size=='lg')].lml_delta_div_n
            #plt.subplot(4, np.ceil(len(champion_trees)/4), i+1)
            #plt.title('$t_{%s}$ - %s'% (i, len(t.to_str().split(' '))))
            #plt.boxplot([qlml1, qlml2])
            pass

    def draw_tree(self, tree, title=None, column='symbol'):
        plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(1, 1, 1)
        plt.title(title)
        plot2(tree, column_label=column, font_size=10,
              label_offset_x=-8,
              label_offset_y=2,
              horizontal_alignment='right',
              node_size=8,
              ax=ax1,
              linewidths=1.0,
              node_color='black')
        plt.show()

    def draw_likelihoods(self):
        plt.figure(figsize=(15, 4))
        likelihoods = [t.log_likelihood() for t in reversed(self.champion_trees)]
        num_contexts = [t.num_contexts() for t in reversed(self.champion_trees)]
        plt.plot(num_contexts, likelihoods, '.', label="Champion trees")
        plt.plot([self.optimal_tree().num_contexts()], [self.optimal_tree().log_likelihood()], 'o', color='red',  label="Optimal tree")
        plt.legend()
        plt.show()

    def create_summary(self, smc_instance, sample, n_sizes, args):
        d = dict()
        d['max_depth'] = smc_instance.max_depth
        d['penalty_interval'] = smc_instance.penalty_interval
        d['epsilon'] = smc_instance.epsilon
        d['df_method'] = smc_instance.df_method
        d['callback_fn'] = smc_instance.callback_fn
        d['scan_offset'] = smc_instance.scan_offset
        d['thresholds'] = smc_instance.thresholds
        d['perl_compatible'] = smc_instance.perl_compatible
        d['bootstrap_sizes'] = n_sizes
        try:
            d['args'] = vars(args).copy()
            d['args']['sample_path'] = d['args']['sample_path'].name
        except:
            pass

        d['trees'] = []
        for i, t in enumerate(smc_instance.context_trees):
            el = {'num_contexts': t.num_contexts(),
                  'tree': t.to_str(reverse=True),
                  'log_likelihood': str(round(t.log_likelihood(), 5))}
            d['trees'].append(el)
        with open(os.path.join(self.folder, 'smc.yml'), 'w') as file:
            yaml.dump(d, file)

    def load_summary(self):
        with open(os.path.join(self.folder, 'smc.yml'), 'r') as file:
            self.summary = yaml.load(file)

    def optimal_tree(self):
        f = os.path.join(self.folder, 'optimal.tree')
        return ContextTree.load_from_file(f)

    def load_trees(self):
        champion_trees_folder = os.path.join(self.folder, 'champion_trees')
        self.champion_trees = []
        for f in sorted(glob.glob(champion_trees_folder + "/*.tree")):
            ct = ContextTree.load_from_file(f)
            self.champion_trees.append(ct)

    def likelihoods_bloxplot(self):
        from g4l.estimators.base import jit_calculate_diffs
        folder = self.folder
        L = np.load(os.path.join(folder, 'likelihoods', 'L.npy'))
        num_resample_sizes, num_trees, num_resamples = L.shape
        d1, d2 = jit_calculate_diffs(self.summary['bootstrap_sizes'],
                                     L,
                                     num_resample_sizes,
                                     num_trees,
                                     num_resamples)
        plt.figure(figsize=(24, 18))
        for i, t in enumerate(self.champion_trees):
            plt.subplot(4, np.ceil(len(self.champion_trees)/4), i+1)
            plt.boxplot([d1[i], d2[i]])

