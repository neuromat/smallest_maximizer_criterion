import jinja2
import glob
import os
import json
import numpy as np
from g4l.context_tree import ContextTree
from g4l.display import plot2
import matplotlib.pyplot as plt
import logging


class SmcReport:
    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        self.reports_folder = os.path.join(self.folder, 'reports')
        self.champion_trees = []
        self.thresholds = np.load(os.path.join(self.folder, 'bic_c.npy'))
        self.load_trees()

    def create_folders(self):
        os.makedirs(self.reports_folder, exist_ok=True)
        os.makedirs(os.path.join(self.reports_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.reports_folder, 'tables'), exist_ok=True)

    def create_summary(self, smc_instance, sample, n_sizes, args):
        self.create_folders()
        d = dict()
        d['max_depth'] = smc_instance.max_depth
        d['penalty_interval'] = smc_instance.penalty_interval
        d['epsilon'] = smc_instance.epsilon
        d['df_method'] = smc_instance.df_method
        d['callback_fn'] = smc_instance.callback_fn
        d['perl_compatible'] = smc_instance.perl_compatible
        d['bootstrap_sizes'] = n_sizes
        try:
            d['args'] = vars(args).copy()
            d['args']['sample_path'] = d['args']['sample_path'].name
        except:
            pass
        d['trees'] = []
        opt = self.optimal_tree()
        for i, t in enumerate(smc_instance.context_trees):
            el = {'num_contexts': t.num_contexts(),
                  'tree': t.to_str(reverse=True),
                  'optimal': t.to_str() == opt.to_str(),
                  'c': smc_instance.thresholds[i],
                  'log_likelihood': str(round(t.log_likelihood(), 5))}
            d['trees'].append(el)
        d['trees'] = list(reversed(d['trees']))
        with open(os.path.join(self.folder, 'smc.json'), 'w') as file:
            json.dump(d, file)

    def draw_tree(self, tree, title=None, column='symbol', filename=None):
        fig = plt.figure(figsize=(15, 6))
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
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def draw_likelihoods(self, filename=None):
        fig = plt.figure(figsize=(15, 4))
        likelihoods = [t.log_likelihood() for t in reversed(self.champion_trees)]
        num_contexts = [t.num_contexts() for t in reversed(self.champion_trees)]
        plt.plot(num_contexts, likelihoods, '.', label="Champion trees")
        plt.plot([self.optimal_tree().num_contexts()], [self.optimal_tree().log_likelihood()], 'o', color='red',  label="Optimal tree")
        plt.legend()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def generate_report(self):
        self.load_summary()
        self.create_report_images()
        self.fill_template()

    def fill_template(self):
        template_path = os.path.dirname(os.path.realpath(__file__))
        templateLoader = jinja2.FileSystemLoader(searchpath=template_path)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template("template.html")
        nodes, transitions = self.create_tables()
        tr_keys = transitions[0][0].keys()
        outputText = template.render(data=self.summary,
                                     nodes=nodes,
                                     transition_keys=list(tr_keys),
                                     transitions=transitions)
        html_file = os.path.join(self.folder, 'report.html')
        with open(html_file, 'w') as file:
            file.write(outputText + '\n')
        logging.info("Report generated at %s" % html_file)

        import code; code.interact(local=dict(globals(), **locals()))
        #print(outputText)
        pass

    def create_tables(self):
        nodes = []
        transitions = []
        for i, t in enumerate(self.champion_trees):
            dfx = t.df[['node', 'freq', 'active', 'likelihood_pen', 'v_node', 'v_node_sum', 'indicator']]
            transitions.append(list(t.node_transitions().reset_index().T.to_dict().values()))
            ns = list(dfx.T.to_dict().values())
            nodes.append(sorted(ns, key=lambda k: k['node'][::-1]))
        return nodes, transitions

    def create_report_images(self):
        for i, tree in enumerate(self.champion_trees):
            img_file = os.path.join(self.reports_folder,
                                    'images',
                                    '%s.png' % tree.num_contexts())

            self.draw_tree(tree, title=None,
                           column='symbol',
                           filename=img_file)

    def load_summary(self):
        with open(os.path.join(self.folder, 'smc.json'), 'r') as file:
            self.summary = json.load(file)

    def optimal_tree(self):
        f = os.path.join(self.folder, 'optimal.tree')
        return ContextTree.load_from_file(f)

    def load_trees(self):
        champion_trees_folder = os.path.join(self.folder, 'champion_trees')
        self.champion_trees = []
        for f in sorted(glob.glob(champion_trees_folder + "/*.tree")):
            ct = ContextTree.load_from_file(f)
            self.champion_trees.append(ct)
