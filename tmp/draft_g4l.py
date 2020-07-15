from . import estimators

def silaba(datafile):
  pass

def scan(c_min, c_max, epsilon):
#  if c_max - c_min < epsilon:
#    return

class SMC():
  def __init__(self, name, data, A, max_depth, nr_Ress = None, epsilon = 0.01, min_max_c=(0.1, 400), four_is_renov_point=1, propRes1=0.3, propRes2=0.9, test_level = 0.01):
    self.min_c = min_max_c[0]
    self.max_c = min_max_c[1]
    assert 0 < epsilon and epsilon < 1, "Epsilon must be between 0 and 1"
    assert 0 < self.min_c and self.min_c < self.max_c, "min_c, max_c must satisfy 0 < min_c < max_c"
    assert 0 < propRes1 and propRes1 < propRes2 and propRes2 < 1, "0 < propRes1 < propRes2 < 1"
    assert 0 < test_level < 1, "0 < test_level < 1"
    self.name = name
    self.data = data
    self.A = A
    self.max_depth = max_depth
    self.nr_ress = nr_Ress
    self.epsilon = eps
    self.prop_res_1 = propRes1
    self.prop_res_2 = propRes2
    self.n = len(data)
    self.BSS0 = int(self.prop_res_1 * self.n);
    self.BSS1 = int(self.prop_res_2 * self.n);
    self.LBSSS = 2




  def execute(self):
    BICtree = estimators.bic(0.5)
    a = self.min_c
    b = self.max_c
    tree_a = estimators.bic(a)
    tree_b = estimators.bic(b)
    arr_ch_tree.append(tree_a)
    while tree_a != tree_b:
      while b - a > self.epsilon:
        while tree_a != tree_b:
          old_b = b
          old_tree_b = tree_b
          b = (a + b)/2
          tree_b = self.ctm(b)
        a = b
        b = old_b
        tree_b = old_tree_b
      a = b
      tree_a = tree_b
      arr_ch_tree.append(tree_a)
      # print( AUX "$Ntr c=$B Tree[$B]=$chtree[$Ntr]\n" );
      # print( CyT "$ns. $Ntr c=$B Tree[$B]=@chtree[$Ntr]\n" );
      # print( NUESTRO "$Ntr:@chtree[$Ntr] ," );
      # print("$Ntr c=$A Tree[$A]=@chtree[$Ntr]\n ");
      b = self.max_c
      tree_b = self.ctm(b)
    return arr_ch_tree # ?


  def bootstrap():
    pass

  def resample():
    pass




def simulation(name, sample_size=10000, nr_sim=100, model=1):
  pass
