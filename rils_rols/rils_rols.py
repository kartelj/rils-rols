from cmath import inf
import copy
import time
import math
from random import Random, shuffle
from sklearn.base import BaseEstimator
import copy
from sympy import *

from .utils import R2, RMSE, ResidualVariance
from .node import Node, NodeConstant, NodeVariable, NodePlus, NodeMinus, NodeMultiply, NodeDivide, NodeSqr, NodeSqrt, NodeLn, NodeExp, NodeSin, NodeCos
from enum import Enum
import warnings
from .solution import Solution
from numpy.linalg import norm
from numpy import linspace, zeros

warnings.filterwarnings("ignore")


class FitnessType(Enum):
    BIC = 1 # bayesian information criterion
    SRM = 2 # structural risk minimization
    PENALTY = 3 # penalty based fitness function

class MonotonicityType(Enum):
    NON_DECREASING = 1
    NON_INCREASING = 2
    INCREASING = 3
    DECREASING = 4

class RILSROLSRegressor(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=100, fitness_type=FitnessType.PENALTY, complexity_penalty=0.001, monotonicity = (0, MonotonicityType.INCREASING, False), lipschitz_continuity=(0.01, 0), distribution_fit=False, initial_sample_size=0.01,  
                 random_perturbations_order = False, verbose=False, random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.fitness_type = fitness_type
        self.complexity_penalty = complexity_penalty
        self.initial_sample_size = initial_sample_size
        self.error_tolerance = 1e-16
        self.verbose = verbose
        self.random_perturbations_order = random_perturbations_order
        self.random_state = random_state
        self.monotoncity = monotonicity
        self.lipschitz_continuity = lipschitz_continuity
        self.distribution_fit = distribution_fit

    def __reset(self):
        self.model = None
        self.ls_it = 0
        self.main_it = 0
        self.last_improved_it = 0
        self.time_start = 0
        self.time_elapsed = 0
        self.rg = Random(self.random_state)
        self.close_points_derivatives = []
        self.true_target_dist = None
        Solution.clearStats()
        Node.reset_node_value_cache()

    def __setup_nodes(self, variableCount):
        self.allowed_nodes=[NodeConstant(-1), NodeConstant(0), NodeConstant(0.5), NodeConstant(1), NodeConstant(2), NodeConstant(math.pi), NodeConstant(10)]
        for i in range(variableCount):
            self.allowed_nodes.append(NodeVariable(i))
        self.allowed_nodes+=[NodePlus(), NodeMinus(), NodeMultiply(), NodeDivide(), NodeSqr(), NodeSqrt(),NodeLn(), NodeExp(),NodeSin(), NodeCos()]#, NodeArcSin(), NodeArcCos()]

    def empirical_cumulative_distribution(self, vals, bins=None):
        # preparing in advance bins for real target values
        tail_size = round(0.05*len(vals))
        vals_sorted = sorted(vals)[tail_size:len(vals)-tail_size]

        if bins is None:
            min = vals_sorted[0]
            max = vals_sorted[len(vals_sorted)-1]
            bins = linspace(min, max+0.0001, 20) # to include it in the the bin

        bin_cnts = zeros(len(bins))
        for v in vals_sorted:
            b = 1
            while v>=bins[b] and b<len(bins):
                b+=1
            bin_cnts[b-1]+=1
        return (bins, bin_cnts)

    def fit(self, X, y):
        self.__reset()
        self.start = time.time()

        x_all = copy.deepcopy(X)
        y_all = copy.deepcopy(y)
        # take 1% of points or at least 100 points initially 
        self.n = int(self.initial_sample_size*len(x_all))
        if self.n<100:
            self.n=min(100, len(X))
        print("Taking "+str(self.n)+" points initially.")
        X = x_all[:self.n]
        y = y_all[:self.n]

        # preparing things for lipshitz continuity
        lipschitz_eps, _ = self.lipschitz_continuity
        for i in range(len(X)):
            for j in range(i):
                dist = norm([X[i][k]-X[j][k] for k in range(len(X[0]))], 2)
                if dist<lipschitz_eps and dist!=0:
                    dist_f = abs(y[i]-y[j])
                    self.close_points_derivatives.append((i, j, dist, dist_f/dist))
        print("There are "+str(len(self.close_points_derivatives))+ " relevant partial derivatives.")

        # preparing things for target variable distribution comparison
        self.true_target_dist = self.empirical_cumulative_distribution(y)

        size_increased_main_it = 0

        if len(X) == 0:
            raise Exception("Input feature data set (X) cannot be empty.")
        if len(X)!=len(y):
            raise Exception("Numbers of feature vectors (X) and target values (y) must be equal.")
        self.__setup_nodes(len(X[0]))
        best_solution =  Solution([NodeConstant(0)], self.complexity_penalty)
        best_fitness = self.fitness(best_solution, X, y)
        self.main_it = 0
        checked_perturbations = set([])
        start_solutions = set([])
        start_solution = copy.deepcopy(best_solution)
        while self.time_elapsed<self.max_seconds and Solution.fit_calls<self.max_fit_calls: 
            start_solutions.add(str(start_solution))
            all_perturbations = self.all_perturbations(start_solution, len(X[0]))
            pret_fits = {}
            for pret in all_perturbations: 
                pret_ols = copy.deepcopy(pret)
                pret_ols = pret_ols.fit_constants_OLS(X, y)
                pret_ols_fit = self.fitness(pret_ols, X, y)
                pret_fits[pret]=pret_ols_fit[0]

            sorted_pret_fits = sorted(pret_fits.items(), key = lambda x: x[1])
            if self.random_perturbations_order:
                shuffle(sorted_pret_fits)

            impr = False
            p = 0
            for pret, r2Inv in sorted_pret_fits:
                p+=1
                self.time_elapsed = time.time()-self.start
                if self.time_elapsed>self.max_seconds:
                    break
                if str(pret) in checked_perturbations:
                    if self.verbose:
                        print(str(p)+"/"+str(len(sorted_pret_fits))+". "+str(pret)+" SKIPPED")
                    continue
                checked_perturbations.add(str(pret))
                if self.verbose:
                    print(str(p)+"/"+str(len(sorted_pret_fits))+". "+str(pret))
                new_solution = pret 
                new_solution.simplify_whole(len(X[0]))
                new_fitness = self.fitness(new_solution, X, y)
                new_solution = self.LS_best(new_solution, X, y)
                new_fitness = self.fitness(new_solution, X, y, False)
                if self.compare_fitness(new_fitness, best_fitness)<0:
                    if self.verbose:
                        print("perturbation "+str(pret)+" produced global improvement.")
                    best_solution = copy.deepcopy(new_solution)
                    best_fitness = new_fitness
                    impr = True
                    break

            start_solution = copy.deepcopy(best_solution)
            if not impr:
                # introduce randomness now and generate new starting solution that wasn't used before and is relatively close to best_solution
                all_perturbations = self.all_perturbations(best_solution, len(X[0]))
                self.rg.shuffle(all_perturbations)
                k=0
                while k<len(all_perturbations) and str(all_perturbations[k]) in start_solutions:
                    k+=1
                if k<len(all_perturbations):
                    start_solution = all_perturbations[k]
                    assert str(start_solution) not in start_solutions
                else:
                    # perform 2-consecutive perturbations -- do not check if this did happen before, it is very small chance for that
                    if len(all_perturbations)>0:
                        perturbation1 = all_perturbations[0]
                        all2_perturbations = self.all_perturbations(perturbation1, len(X[0]))
                        if len(all2_perturbations)>0:
                            start_solution = all2_perturbations[self.rg.randrange(len(all2_perturbations))]

                if self.verbose:
                    print("RANDOMIZING "+str(best_solution)+" TO "+str(start_solution))
                if self.n<len(x_all) and (self.main_it-size_increased_main_it)>=10:
                    self.n*=2
                    if self.n>len(x_all):
                        self.n = len(x_all)
                    print("Increasing data count to "+str(self.n))
                    X = x_all[:self.n]
                    y = y_all[:self.n]
                    size_increased_main_it = self.main_it
                    Node.reset_node_value_cache()

            self.time_elapsed = time.time()-self.start
            print("%d/%d. t=%.1f R2=%.7f RMSE=%.7f MONO=%.7f LIPS=%.7f DIST=%.7f size=%d fitFails=%d/%d cPerc=%.1f expr=%s"
            %(self.main_it,self.ls_it, self.time_elapsed, 1-best_fitness[0], best_fitness[1], best_fitness[6], best_fitness[7], best_fitness[8], best_solution.size(), Solution.fit_fails, Solution.fit_calls,Node.cache_hits*100.0/Node.cache_tries, best_solution))
            self.main_it+=1
            if best_fitness[0]<=self.error_tolerance and best_fitness[1] <= pow(self.error_tolerance, 0.125):
                break
        self.model_simp = simplify(str(best_solution), ratio=1)
        self.model_simp = self.round_floats(self.model_simp)
        self.model = Solution([Solution.convert_to_my_nodes(self.model_simp)], self.complexity_penalty)
        return (self.model, self.model_simp)
    
    def round_floats(self, ex1):
        round_digits = int(math.log10(1/self.error_tolerance))
        if round_digits>8:
            round_digits = 8
        #if round_digits<4:
        #    round_digits = 4
        print("Round to "+str(round_digits)+" digits.")
        ex2 = ex1
        for a in preorder_traversal(ex1):
            if isinstance(a, Float):
                if round(a, round_digits)==round(a, 0):
                    ex2 = ex2.subs(a, Integer(round(a)))
                else:
                    ex2 = ex2.subs(a, Float(round(a, round_digits)))
        return ex2

    def predict(self, X):
        Node.reset_node_value_cache()
        return self.model.evaluate_all(X, False)

    def size(self):
        if self.model is not None:
            return self.model.size()
        return math.inf

    def modelString(self):
        if self.model_simp is not None:
            return str(self.model_simp)
        return ""

    def fit_report_string(self, X, y):
        if self.model==None:
            raise Exception("Model is not build yet. First call fit().")
        fitness = self.fitness(self.model, X,y, False)
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tR2={4:.7f}\tRMSE={5:.7f}\tsize={6}\tsec={7:.1f}\tmainIt={8}\tlsIt={9}\tfitCalls={10}\texpr={11}\texprSimp={12}\tfitType={13}\tinitSampleSize={14}\trandomPert={15}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty, 1-fitness[0], fitness[1], self.complexity(self.model_simp), self.time_elapsed,self.main_it, self.ls_it,Solution.fit_calls, self.model, self.model_simp, self.fitness_type, self.initial_sample_size, self.random_perturbations_order)

    def complexity(self, sm):
        c=0
        for arg in preorder_traversal(sm):
            c += 1
        return c

    def all_perturbations(self, solution: Solution, var_cnt):
        all = []
        shaked_solution = copy.deepcopy(solution)
        shaked_solution.normalize_constants()
        shaked_solution.simplify_whole(var_cnt)
        shaked_solution.join()
        assert len(shaked_solution.factors)==1
        j = 0
        all_subtrees = shaked_solution.factors[0].all_nodes_exact()
        if len(all_subtrees)==0: # this is the case when we have constant or variable, so we just change the root
            for cand in self.perturb_candidates(shaked_solution.factors[j]):
                perturbed = copy.deepcopy(shaked_solution)
                perturbed.factors[j] = cand
                #perturbed.simplify_whole(varCnt)
                all.append(perturbed)
        else:
            for i in range(len(all_subtrees)):
                refNode = all_subtrees[i]
                if refNode==shaked_solution.factors[j]:
                    for cand in self.perturb_candidates(shaked_solution.factors[j]):
                        perturbed = copy.deepcopy(shaked_solution)
                        perturbed.factors[j] = cand
                        #perturbed.simplify_whole(varCnt)
                        all.append(perturbed)
                if refNode.arity >= 1:
                    for cand in self.perturb_candidates(refNode.left, refNode, True):
                        perturbed = copy.deepcopy(shaked_solution)
                        perturbed_subtrees = perturbed.factors[j].all_nodes_exact()
                        perturbed_subtrees[i].left = cand
                        #perturbed.simplify_whole(varCnt)
                        all.append(perturbed)
                if refNode.arity>=2:
                    for cand in self.perturb_candidates(refNode.right, refNode, False):
                        perturbed = copy.deepcopy(shaked_solution)
                        perturbed_subtrees = perturbed.factors[j].all_nodes_exact()
                        perturbed_subtrees[i].right = cand
                        #perturbed.simplify_whole(varCnt)
                        all.append(perturbed)
        return all
    
    def LS_best(self, solution: Solution, X, y):
        best_fitness = self.fitness(solution, X, y)
        best_solution = copy.deepcopy(solution)
        impr = True
        while impr or impr2:
            impr = False
            impr2 = False
            self.ls_it+=1
            self.time_elapsed = time.time()-self.start
            if self.time_elapsed>self.max_seconds or Solution.fit_calls>self.max_fit_calls:
                break
            old_best_fitness = best_fitness
            old_best_solution = copy.deepcopy(best_solution)
            impr, best_solution, best_fitness = self.LS_best_change_iteration(best_solution, X, y, True)
            if impr or impr2:
                best_solution.simplify_whole(len(X[0]))
                best_fitness = self.fitness(best_solution, X, y, False)
                if self.compare_fitness(best_fitness, old_best_fitness)>=0:
                    impr = False
                    impr2 = False
                    best_solution = old_best_solution
                    best_fitness = old_best_fitness
                    if self.verbose:
                        print("REVERTING back to old best "+str(best_solution))
                else:
                    if self.verbose:
                        print("IMPROVED with LS-change "+str(best_fitness)+"  "+str(best_solution))
                continue  
        return best_solution

    def LS_best_change_iteration(self, solution: Solution, X, y, cache, joined=False):
        best_fitness = self.fitness(solution, X, y, False)
        best_solution = copy.deepcopy(solution)
        if joined:
            print("JOINING SOLUTION IN LS")
            solution.join()
        impr = False
        for i in range(len(solution.factors)):
            factor = solution.factors[i]
            factor_subtrees = factor.all_nodes_exact()
            for j in range(len(factor_subtrees)):
                
                self.time_elapsed = time.time()-self.start
                if self.time_elapsed>self.max_seconds or Solution.fit_calls>self.max_fit_calls:
                    return (impr, best_solution, best_fitness)

                ref_node = factor_subtrees[j]

                if ref_node==factor: # this subtree is the whole factore
                    candidates = self.change_candidates(ref_node)
                    for cand in candidates:
                        new_solution = copy.deepcopy(solution)
                        new_solution.factors[i] = cand
                        if joined:
                            new_solution.expand_fast()
                        new_solution = new_solution.fit_constants_OLS(X, y)
                        new_fitness = self.fitness(new_solution, X, y, cache)
                        if self.compare_fitness(new_fitness, best_fitness)<0:
                            impr = True
                            best_fitness = new_fitness
                            best_solution = copy.deepcopy(new_solution)

                if ref_node.arity >= 1:
                    candidates = self.change_candidates(ref_node.left, ref_node, True)
                    for cand in candidates:
                        new_solution = copy.deepcopy(solution)               
                        new_factor_subtrees = new_solution.factors[i].all_nodes_exact()
                        new_factor_subtrees[j].left=cand
                        if joined:
                            new_solution.expand_fast()
                        new_solution = new_solution.fit_constants_OLS(X, y)
                        new_fitness = self.fitness(new_solution, X, y, cache)
                        if self.compare_fitness(new_fitness, best_fitness)<0:
                            impr = True
                            best_fitness = new_fitness
                            best_solution = copy.deepcopy(new_solution)

                if ref_node.arity>=2:
                    candidates = self.change_candidates(ref_node.right, ref_node, False)
                    for cand in candidates:
                        new_solution = copy.deepcopy(solution)               
                        new_factor_subtrees = new_solution.factors[i].all_nodes_exact()
                        new_factor_subtrees[j].right=cand
                        if joined:
                            new_solution.expand_fast()
                        new_solution = new_solution.fit_constants_OLS(X, y)
                        new_fitness = self.fitness(new_solution, X, y, cache)
                        if self.compare_fitness(new_fitness, best_fitness)<0:
                            impr = True
                            best_fitness = new_fitness
                            best_solution = copy.deepcopy(new_solution)

        return (impr, best_solution, best_fitness)

    # perturb_candidates is a subset of change_candidates
    def perturb_candidates(self, old_node: Node, parent=None, is_left_from_parent=None):
        candidates = set([])
        # change node to one of its subtrees -- reduces the size of expression
        if old_node.arity>=1:
            all_left_subtrees = old_node.left.all_nodes_exact()
            for ls in all_left_subtrees:
                candidates.add(copy.deepcopy(ls))
            #candidates.append(copy.deepcopy(old_node.left))
        if old_node.arity>=2:
            all_right_subtrees = old_node.right.all_nodes_exact()
            for rs in all_right_subtrees:
                candidates.add(copy.deepcopy(rs))
            #candidates.append(copy.deepcopy(old_node.right))
        # change constant to variable
        if old_node.arity==0 and type(old_node)==type(NodeConstant(0)):
            for node in filter(lambda x:type(x)==type(NodeVariable(0)) and x!=old_node, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                candidates.add(new_node)
        # change variable to unary operation applied to that variable
        if type(old_node)==type(NodeVariable(0)):
            for node in filter(lambda x:x.arity==1, self.allowed_nodes):
                if not node.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node)
                new_node.left =copy.deepcopy(old_node)
                new_node.right = None
                candidates.add(new_node)
        # change unary operation to another unary operation
        if old_node.arity == 1:
            for node in filter(lambda x:x.arity==1 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                new_node.left = copy.deepcopy(old_node.left)
                assert old_node.right==None
                candidates.add(new_node)
        # change one binary operation to another
        if old_node.arity==2:
            for nodeOp in filter(lambda x: x.arity==2 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                if (not nodeOp.is_allowed_left_argument(old_node.left)) or (not nodeOp.is_allowed_right_argument(old_node.right)):
                    continue
                new_node = copy.deepcopy(nodeOp)
                new_node.left = copy.deepcopy(old_node.left)
                new_node.right = copy.deepcopy(old_node.right)
                candidates.add(new_node) 
            # swap left and right side if not symmetric op
            if not old_node.symmetric:
                new_node = copy.deepcopy(old_node)
                new_node.left = copy.deepcopy(old_node.right)
                new_node.right = copy.deepcopy(old_node.left)
                candidates.add(new_node)
        # change variable or constant to binary operation with some variable  -- increases the model size
        if old_node.arity==0:
            node_args = list(filter(lambda x: type(x)==type(NodeVariable(0)), self.allowed_nodes))
            for node_arg in node_args:
                for node_op in filter(lambda x: x.arity==2, self.allowed_nodes):
                    if not node_op.is_allowed_right_argument(node_arg) or not node_op.is_allowed_left_argument(old_node):
                        continue
                    new_node = copy.deepcopy(node_op)
                    new_node.left = copy.deepcopy(old_node)
                    new_node.right = copy.deepcopy(node_arg)
                    candidates.add(new_node)
                    if not node_op.symmetric and node_op.is_allowed_right_argument(old_node) and node_op.is_allowed_left_argument(node_arg):
                        new_node = copy.deepcopy(node_op)
                        new_node.right = copy.deepcopy(old_node)
                        new_node.left = copy.deepcopy(node_arg)
                        candidates.add(new_node)
        # filtering not allowed candidates (because of the parent)
        filtered_candidates = []
        if parent is not None:
            for c in candidates:
                if is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                if not is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                filtered_candidates.append(c)
            candidates = filtered_candidates
        candidates = sorted(candidates, key=lambda x: str(x))
        return candidates

    # superset of perturb_candidates
    def change_candidates(self, old_node:Node, parent=None, is_left_from_parent=None):
        candidates = set([])
        # change constant to something multiplied with it
        if type(old_node)==type(NodeConstant(0)):
            for mult in [0.01, 0.1, 0.2, 0.5, 0.8, 0.9,1.1,1.2, 2, 5, 10, 20, 50, 100]:
                candidates.add(NodeConstant(old_node.value*mult))
        # change anything to one of its left subtrees
        if old_node.arity>=1:
            all_left_subtrees = old_node.left.all_nodes_exact()
            for ls in all_left_subtrees:
                candidates.add(copy.deepcopy(ls))
            #candidates.append(copy.deepcopy(old_node.left))
        # change anything to one of its right subtrees
        if old_node.arity>=2:
            all_right_subtrees = old_node.right.all_nodes_exact()
            for rs in all_right_subtrees:
                candidates.add(copy.deepcopy(rs))
            #candidates.append(copy.deepcopy(old_node.right))
        # change anything to a constant or variable
        for node in filter(lambda x:x.arity==0 and x!=old_node, self.allowed_nodes):
            candidates.add(copy.deepcopy(node))
        # change anything to unary operation applied to that -- increases the model size
        for node in filter(lambda x:x.arity==1, self.allowed_nodes):
            if not node.is_allowed_left_argument(old_node):
                continue
            new_node = copy.deepcopy(node)
            new_node.left =copy.deepcopy(old_node)
            new_node.right = None
            candidates.add(new_node)
        # change unary operation to another unary operation
        if old_node.arity == 1:
            for node in filter(lambda x:x.arity==1 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                new_node.left = copy.deepcopy(old_node.left)
                assert old_node.right==None
                candidates.add(new_node)
        # change anything to binary operation with some variable or constant -- increases the model size
        # or with some part of itself
        node_args = list(filter(lambda x: x.arity==0, self.allowed_nodes))+[copy.deepcopy(x) for x in old_node.all_nodes_exact()]
        for node_arg in node_args:
            for node_op in filter(lambda x: x.arity==2, self.allowed_nodes):
                if not node_op.is_allowed_right_argument(node_arg) or not node_op.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node_op)
                new_node.left = copy.deepcopy(old_node)
                new_node.right = copy.deepcopy(node_arg)
                candidates.add(new_node)
                if not node_op.symmetric and node_op.is_allowed_right_argument(old_node) and node_op.is_allowed_left_argument(node_arg):
                    new_node = copy.deepcopy(node_op)
                    new_node.right = copy.deepcopy(old_node)
                    new_node.left = copy.deepcopy(node_arg)
                    candidates.add(new_node)
        # change one binary operation to another
        if old_node.arity==2:
            for node_op in filter(lambda x: x.arity==2 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                if (not node_op.is_allowed_left_argument(old_node.left)) or (not node_op.is_allowed_right_argument(old_node.right)):
                    continue
                new_node = copy.deepcopy(node_op)
                new_node.left = copy.deepcopy(old_node.left)
                new_node.right = copy.deepcopy(old_node.right)
                candidates.add(new_node) 
            # swap left and right side if not symmetric op
            if not old_node.symmetric:
                new_node = copy.deepcopy(old_node)
                new_node.left = copy.deepcopy(old_node.right)
                new_node.right = copy.deepcopy(old_node.left)
                candidates.add(new_node)
        # filtering not allowed candidates (because of the parent)
        filtered_candidates = []
        if parent is not None:
            for c in candidates:
                if is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                if not is_left_from_parent and not parent.is_allowed_right_argument(c):
                    continue
                filtered_candidates.append(c)
            candidates = filtered_candidates
        candidates = sorted(candidates, key=lambda x: str(x))
        return candidates
    
    def monotonicity_score(self, yp, X):
        var_idx, monotonicity_type, _ = self.monotoncity
        # order w.r.t. var_idx and check lineary if yp follows the given monotonicity
        Xyp = zip(X, yp)
        Xyp_sorted = sorted(Xyp, key=lambda x: x[0][var_idx])

        incorrect = 0
        # now simply count how many y values do not follow given monotonicity
        for i in range(1, len(Xyp_sorted)):
            ypi_prev = Xyp_sorted[i-1][1]
            ypi = Xyp_sorted[i][1]
            if monotonicity_type==MonotonicityType.NON_DECREASING and ypi<ypi_prev:
                incorrect+=1
            elif monotonicity_type==MonotonicityType.NON_INCREASING and ypi>ypi_prev:
                incorrect+=1
            elif monotonicity_type==MonotonicityType.INCREASING and ypi<=ypi_prev:
                incorrect+=1
            elif monotonicity_type==MonotonicityType.DECREASING and ypi>=ypi_prev:
                incorrect+=1
        score = incorrect*1.0/(len(Xyp_sorted)-1)
        return score
    
    def lipschitz_continuity_score(self, yp):
        score = 0.0
        for i, j, dist, deriv in self.close_points_derivatives:
            dist_fp = abs(yp[i]-yp[j])
            derivp = dist_fp/dist 
            err = deriv-derivp
            score+=(err*err)
        score = sqrt(score/max(1, len(self.close_points_derivatives))) # to avoid division by zero
        return score
    
    def distribution_fit_score(self, yp):
        score = 0.0
        y_bins, y_cnts = self.true_target_dist
        _, yp_cnts = self.empirical_cumulative_distribution(yp, bins=y_bins)
        n = len(yp)
        #Bhattacharyya coefficient
        for b in range(len(y_bins)):
            score+=sqrt(y_cnts[b]*yp_cnts[b]/n/n)
        return 1-score

    def fitness(self,solution : Solution, X, y, cache=True):
        try:
            Solution.fit_calls+=1
            yp = solution.evaluate_all(X, cache) 
            size = solution.size()
            return (1-R2(y, yp), RMSE(y, yp), size, ResidualVariance(y, yp, size), solution.size_non_linear(), solution.size_operators_only(), self.monotonicity_score(yp,X), self.lipschitz_continuity_score(yp), self.distribution_fit_score(yp))
        except Exception as e:
            #print(e)
            Solution.math_error_count+=1
            Solution.fit_fails+=1
            return (inf, inf, inf, inf, inf, inf, inf, inf, inf)
    
    def compare_fitness(self, new_fit, old_fit):
        if self.fitness_type == FitnessType.BIC:
            return self.compare_fitness_bic(new_fit, old_fit)
        elif self.fitness_type == FitnessType.SRM:
            return self.compare_fitness_srm(new_fit, old_fit)
        elif self.fitness_type == FitnessType.PENALTY:
            return self.compare_fitness_penalty(new_fit, old_fit)
        else:
            raise Exception("Unrecognized fitness type "+str(self.fitness_type))

    def compare_fitness_bic(self, new_fit, old_fit):
        if math.isnan(new_fit[0]):
            return 1
        # BIC -- see equation (9) in https://www.researchgate.net/profile/Jose-Montana/publication/220743408_Model_selection_in_genetic_programming/links/0c960532776628e069000000/Model-selection-in-genetic-programming.pdf
        new_mse = new_fit[1]*new_fit[1]
        old_mse = old_fit[1]*old_fit[1]
        new_h = new_fit[2]
        old_h = old_fit[2]
        new_var = new_fit[3]
        old_var = old_fit[3]
        new_tot = new_mse + math.log(self.n)*new_h*new_var/self.n
        old_tot = old_mse + math.log(self.n)*old_h*old_var/self.n
        if new_tot<old_tot-0.00000001:
            return -1
        if new_tot>old_tot+0.00000001:
            return 1
        return 0
    
    def compare_fitness_srm(self, new_fit, old_fit):
        if math.isnan(new_fit[0]):
            return 1
        # SRM -- see equation (11) in https://www.researchgate.net/profile/Jose-Montana/publication/220743408_Model_selection_in_genetic_programming/links/0c960532776628e069000000/Model-selection-in-genetic-programming.pdf
        new_mse = new_fit[1]*new_fit[1]
        old_mse = old_fit[1]*old_fit[1]
        new_h = new_fit[4] # non-linear size
        old_h = old_fit[4]
        new_p = new_h/self.n
        old_p = old_h/self.n
        new_tot = new_mse/(1-math.sqrt(new_p-new_p*math.log(new_p)+math.log(self.n)/(2*self.n)))
        old_tot = old_mse/(1-math.sqrt(old_p-old_p*math.log(old_p)+math.log(self.n)/(2*self.n)))
        if new_tot<old_tot-0.00000001:
            return -1
        if new_tot>old_tot+0.00000001:
            return 1
        return 0

    def compare_fitness_penalty(self, new_fit, old_fit):
        if math.isnan(old_fit[0]) or old_fit[0]==inf:
            return -1
        if math.isnan(new_fit[0]) or new_fit[0]==inf:
            return 1
        new_fit_wo_size_pen = (1+new_fit[0])*(1+new_fit[1])
        old_fit_wo_size_pen = (1+old_fit[0])*(1+old_fit[1]) 

        if self.monotoncity[2]:
            # monotonicity is used
            new_fit_wo_size_pen*=(1+new_fit[6])
            old_fit_wo_size_pen*=(1+old_fit[6])

        _, lipschitz_penalty = self.lipschitz_continuity
        new_fit_wo_size_pen*=(1+lipschitz_penalty*new_fit[7])
        old_fit_wo_size_pen*=(1+lipschitz_penalty*old_fit[7])

        if self.distribution_fit:
            new_fit_wo_size_pen*=(1+ new_fit[8])
            old_fit_wo_size_pen*=(1+ old_fit[8])

        if new_fit_wo_size_pen*old_fit_wo_size_pen!=1: # otherwise, if they are both 1 (perfect), code bellow will return better based on the size
            if new_fit_wo_size_pen==1: # if this one is perfrect solution, return it
                return -1
            if old_fit_wo_size_pen==1: # otherwise, old is better
                return 1
        new_tot = new_fit_wo_size_pen*(1+new_fit[2]*self.complexity_penalty)
        old_tot = old_fit_wo_size_pen*(1+old_fit[2]*self.complexity_penalty)
        if new_tot<old_tot-0.00000001:
            return -1
        if new_tot>old_tot+0.00000001:
            return 1
        return 0
