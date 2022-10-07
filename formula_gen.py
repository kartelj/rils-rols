from msilib.schema import Error
import sys
import copy
import math
from rils_rols.node import Node, NodePlus, NodeVariable, NodeMinus, NodeMultiply, NodeDivide, NodeSqr, NodeSqrt, NodeLn, NodeExp, NodeSin, NodeCos, NodeConstant
import random
from sympy import symbols, simplify, lambdify, I
import os
import warnings

warnings.filterwarnings("error")

def random_tree(size, allowed_nodes, rg, parent=None):
    if size<1:
        raise Exception("Random tree cannot have size smaller than 1.")
    if size==1:
        allowed_terms = list(filter(lambda x: x.arity==0 , allowed_nodes)) 
        node = copy.deepcopy(allowed_terms[rg.randrange(len(allowed_terms))])
        return node
    elif size == 2:
        interm = list(filter(lambda x: x.arity==1, allowed_nodes))
        node = copy.deepcopy(interm[rg.randrange(len(interm))])
        node.left = random_tree(size-1, allowed_nodes, rg, node)
        return node
    else:
        min_arity = 1
        if parent is not None and parent.arity==1:
            min_arity = 2 # this is to avoid unary compositions like f(g(h...)), although f(g(..)) can happen because of above elif, but no more than 2 unary compositions will happen in non-simplified form
        interm = list(filter(lambda x: x.arity>=min_arity, allowed_nodes))
        node = copy.deepcopy(interm[rg.randrange(len(interm))])
        if node.arity == 1:
            node.left = random_tree(size-1, allowed_nodes, rg, node)
        else:
            if rg.randint(0,1)==0:
                left_size = math.floor(size/2)
                right_size = size - left_size - 1
            else:
                right_size = math.floor(size/2)
                left_size = size - right_size - 1
            node.left = random_tree(left_size, allowed_nodes, rg, node)
            node.right = random_tree(right_size, allowed_nodes, rg, node)
        return node

def contains_all_vars(expr : Node, var_cnt):
    for i in range(var_cnt):
        if not expr.contains(NodeVariable(i)):
            return False
    return True

def sympy_expr_size(expr):
    size = 1
    for arg in expr.args:
        size+=sympy_expr_size(arg)
    return size

# Generates formulas with a fixed set of possible node, for a given size, number of variables.
if len(sys.argv)!=6:
    print("Usage: <number> <size> <variable count> <row count> <random seed>")
    print("Passed parameters were:")
    print(sys.argv[1:])
    sys.exit(1)

n = int(sys.argv[1])
size = int(sys.argv[2])
var_cnt = int(sys.argv[3])
row_cnt = int(sys.argv[4])
seed = int(sys.argv[5])

allowed_nodes=[NodeConstant(math.pi), NodeConstant(-1), NodeConstant(1), NodeConstant(2), NodeConstant(10)] 
symb_arr = []
for i in range(var_cnt):
    var_i = NodeVariable(i)
    allowed_nodes.append(var_i)
    symb_arr.append(str(var_i))
allowed_nodes+=[NodePlus(), NodeMinus(), NodeMultiply(), NodeDivide(), NodeSqr(), NodeSqrt(),NodeLn(), NodeExp(),NodeSin(), NodeCos()]

inst_dir_truth = "random_"+str(seed)+"_truth"
inst_dir_data = "random_"+str(seed)+"_data"
if not os.path.isdir(inst_dir_truth):
    os.mkdir(inst_dir_truth)
if not os.path.isdir(inst_dir_data):
    os.mkdir(inst_dir_data)

symbols(" ".join(symb_arr))
rg = random.Random(seed)
# generate input random matrix of size [row_cnt]  x [var_cnt]
rand_input_mat = []
for r in range(row_cnt):
    row = []
    for v in range(var_cnt):
        row.append(rg.random())
    rand_input_mat.append(row)

i = 0
all = set([])
while i<n:
    inst_name = "random_"+str(size).zfill(2)+"_"+str(var_cnt).zfill(2)+"_"+str(row_cnt).zfill(7)+"_"+str(i).zfill(2)
    rand_form = random_tree(size, allowed_nodes, rg)
    #print(rand_form.size())
    #print(rand_form)
    if not contains_all_vars(rand_form, var_cnt):
        continue
    try:
        rand_form_simp =  simplify(str(rand_form), ratio=1)
    except:
        continue
    if rand_form_simp.has(I):
        continue
    form_size = sympy_expr_size(rand_form_simp)
    if form_size != size: 
        continue
    if rand_form_simp in all:
        continue
    # lambdify for efficient calculation -- subs is too slow
    func = lambdify(symb_arr, str(rand_form_simp))
    all_rows_ok = True
    with open(inst_dir_data+"/"+inst_name+".data", "w") as f:
        for r in range(row_cnt):
            try:
                y = func(*rand_input_mat[r])
            except Warning as w:
                print("Warning happened "+str(w))
                all_rows_ok = False
                break
            except (NameError, TypeError) as e:
                print("Error happened "+str(e))
                all_rows_ok = False
                break
            f.write("\t".join([str(ri) for ri in rand_input_mat[r]])+"\t"+str(y)+"\n")
    if not all_rows_ok:
        continue
    print(inst_name+"\t" +str(rand_form_simp))
    with open(inst_dir_truth+"/"+inst_name+".f", "w") as f:
        f.write(str(rand_form_simp))
    with open(inst_dir_truth+"/all.f", "a") as f:
        f.write(inst_name+"\t"+str(rand_form_simp)+"\n")
    all.add(rand_form_simp)
    i+=1


