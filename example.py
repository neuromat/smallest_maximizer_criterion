import g4l
from g4l.util.mat import MatSamples
from g4l.estimators.bic import BIC
from g4l.data import Sample

#fld = 'examples/simulation_study/samples'
#max_depth = 6
#X = MatSamples(fld, 'model1',
#               5000, [0, 1],
#               max_depth,
#               scan_offset=6).sample_by_idx(1)

perl_compatible = True
X = Sample('examples/linguistic_case_study/publico.txt.bkp',
           [0, 1, 2, 3, 4],
           4,
           cache_file='/tmp/asd.tmp',
           perl_compatible=perl_compatible,
           subsamples_separator='>')


c = 100
tree = BIC(c, 4, keep_data=True, perl_compatible=perl_compatible).fit(X).context_tree

print(tree.to_str())
g4l.display.plot2(tree, column_label='symbol', label_offset_y=0,
                  linewidths=2.0)
