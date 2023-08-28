from abc import abstractmethod
import copy
import numpy as np

class Node:
    tmp = -1
    node_value_cache = {}
    cache_hits = 0
    cache_tries = 1

    VERY_SMALL = 0.0001

    @classmethod
    def reset_node_value_cache(cls):
        cls.node_value_cache = {}
        cls.cache_hits = 0
        cls.cache_tries = 1    

    def __init__(self):
        self.arity = 0
        self.left = None
        self.right = None
        self.symmetric = True

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        pass

    def deepcopy(self, dst, memodict={}):
        dst.arity = self.arity
        if self.left is None:
            dst.left = None
        else:
            dst.left = copy.deepcopy(self.left, memodict)
        if self.right is None:
            dst.right = None
        else:
            dst.right = copy.deepcopy(self.right, memodict)
        dst.symmetric = self.symmetric

    @abstractmethod
    def evaluate_inner(self,X, a=None, b=None):
        pass

    def evaluate(self, X):
        if self.arity==0:
            return self.evaluate_inner(X, None, None)
        elif self.arity == 1:
            left_val = self.left.evaluate(X)
            return self.evaluate_inner(X,left_val, None)
        elif self.arity==2:
            left_val = self.left.evaluate(X)
            right_val = self.right.evaluate(X)
            return self.evaluate_inner(X,left_val,right_val)
        else:
            raise Exception("Arity > 2 is not allowed.")

    def evaluate_all(self, X, cache):
        key = str(self)
        Node.cache_tries+=1
        yp = []
        # do not waste time and memory to cache constants
        if cache and not isinstance(self, NodeConstant) and key in Node.node_value_cache:
            Node.cache_hits+=1
            yp = Node.node_value_cache[key]
        else:
            if self.arity==2:
                left_yp = self.left.evaluate_all(X, cache)
                right_yp = self.right.evaluate_all(X, cache)
                yp = self.evaluate_inner(X, left_yp, right_yp)
            elif self.arity==1:
                left_yp = self.left.evaluate_all(X, cache)
                yp = self.evaluate_inner(X, left_yp, None)
            elif self.arity==0:
                yp = self.evaluate_inner(X, None, None)
            if cache and not isinstance(self, NodeConstant):
                Node.node_value_cache[key]=yp
                if len(Node.node_value_cache)==5000:
                    Node.node_value_cache.clear()
        return yp

    def expand_fast(self):
        if type(self)==type(NodePlus()) or type(self)==type(NodeMinus()): # TODO: check if minus is happening
            left_fact = self.left.expand_fast()
            right = self.right
            if type(self)==type(NodeMinus()):
                right = NodeMultiply()
                right.left = NodeConstant(-1)
                right.right = copy.deepcopy(self.right)
            right_fact = right.expand_fast()
            return left_fact+right_fact
        return [copy.deepcopy(self)]

    def is_allowed_left_argument(self, node_arg):
        return True

    def is_allowed_right_argument(self, node_arg):
        return True

    def __eq__(self, object):
        if object is None:
            return False
        return str(self)==str(object)

    def __hash__(self):
        return hash(str(self))

    def all_nodes_exact(self):
        this_list = [self]
        if self.arity==0:
            return this_list
        elif self.arity==1:
            return this_list+self.left.all_nodes_exact()
        elif self.arity==2:
            return this_list+self.left.all_nodes_exact()+self.right.all_nodes_exact()
        else:
            raise Exception("Arity greater than 2 is not allowed.")

    def size(self):
        left_size = 0
        if self.left!=None:
            left_size = self.left.size()
        right_size = 0
        if self.right!=None:
            right_size = self.right.size()
        return 1+left_size+right_size
    
    def size_non_linear(self):
        left_size = 0
        if self.left!=None:
            left_size = self.left.size_non_linear()
        right_size = 0
        if self.right!=None:
            right_size = self.right.size_non_linear()
        # counting the level of non-linearity, so excluding terms and operations plus and minus
        if type(self)!=type(NodeConstant(0)) and type(self)!=type(NodeVariable(0)) and type(self)!=type(NodePlus) and type(self)!=type(NodeMinus):
            return 1+left_size+right_size
        else:
            return left_size+right_size
        
    def size_operators_only(self):
        left_size = 0
        if self.left!=None:
            left_size = self.left.size_operators_only()
        right_size = 0
        if self.right!=None:
            right_size = self.right.size_operators_only()
        # not counting terms
        if type(self)!=type(NodeConstant(0)) and type(self)!=type(NodeVariable(0)):
            return 1+left_size+right_size
        else:
            return left_size+right_size

    def contains_type(self, search_type):
        if type(self)==search_type:
            return True
        if self.left!=None and self.left.contains_type(search_type):
            return True
        if self.right!=None and self.right.contains_type(search_type):
            return True
        return False

    def normalize_constants(self, parent=None):
        if type(self)==type(NodeConstant(0)):
            if parent==None or type(parent) == type(NodeMultiply()) or type(parent)==type(NodePlus()) or type(parent)==type(NodeMinus()) or type(parent)==type(NodeDivide()):
                self.value = 1
            elif type(parent)==type(NodePow()) and self.value!=0.5 and self.value!=-0.5:
                    self.value = round(self.value)
            return
        if self.arity>=1:
            self.left.normalize_constants(self)
        if self.arity>=2:
            self.right.normalize_constants(self)

class NodeConstant(Node):
    def __init__(self, value):
        super().__init__()
        self.arity = 0
        self.value = round(value,13)

    def __deepcopy__(self, memodict={}):
        copy_object = NodeConstant(self.value)
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return self.value

    def __str__(self):
        return str(self.value)

class NodeVariable(Node):
    def __init__(self, index):
        super().__init__()
        self.arity = 0
        self.index = index
    
    def __deepcopy__(self, memodict={}):
        copy_object = NodeVariable(self.index)
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        # I am not sure if this check is important, can be removed
        if self.index >= np.shape(X)[1]:
            raise Exception("Variable with index " +
                            str(self.index)+" does not exist.")
        return X[:, self.index]

    def __str__(self):
        return "x"+str(self.index)

class NodePlus(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2

    def __deepcopy__(self, memodict={}):
        copy_object = NodePlus()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return a+b

    def is_allowed_left_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        return self.is_allowed_left_argument(node_arg)

    def __str__(self):
        return "("+str(self.left)+"+"+str(self.right)+")" 

class NodeMinus(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = False

    def __deepcopy__(self, memodict={}):
        copy_object = NodeMinus()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return a - b

    def is_allowed_left_argument(self, node_arg):
        if self.right==node_arg:
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        if self.left == node_arg:
            return False
        return True

    def normalize(self):
        if type(self.right)==type(NodeConstant(0)):
            new_left =  NodeConstant(self.right.value*(-1))
            new_right = copy.deepcopy(self.left)
            self = NodePlus()
            self.left = new_left
            self.right = new_right
            return self.normalize()
        else:
            return super().normalize()

    def __str__(self):
        return "("+str(self.left)+"-"+str(self.right)+")"


class NodeMultiply(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2

    def __deepcopy__(self, memodict={}):
        copy_object = NodeMultiply()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return a*b

    def is_allowed_left_argument(self, node_arg):
        if node_arg == NodeConstant(1):
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        return self.is_allowed_left_argument(node_arg)

    def __str__(self):
        return "("+str(self.left)+"*"+str(self.right)+")"

class NodeDivide(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = False

    def __deepcopy__(self, memodict={}):
        copy_object = NodeDivide()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        if type(b) is np.ndarray:
            _b = np.copy(b)
            _b[np.abs(_b) < Node.VERY_SMALL] = Node.VERY_SMALL
            return a / _b
        
        # I am not sure if code:
        # if b==0:
        #   b = Node.VERY_SMALL
        # its good idea, if for example b == 1e-15 its not changed
        # but if its zero it changed to 0.0001
        if np.abs(b) < Node.VERY_SMALL:
            b = Node.VERY_SMALL
        
        return a/b

    def is_allowed_left_argument(self, node_arg):
        if self.right == node_arg:
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        if self.left == node_arg:
            return False
        return True

    def __str__(self):
        return "("+str(self.left)+"/"+str(self.right)+")"

class NodeMax(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = True

    def __deepcopy__(self, memodict={}):
        copy_object = NodeMax()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.maximum(a, b)

    def __str__(self):
        return "max("+str(self.left)+","+str(self.right)+")"

class NodeMin(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = True

    def __deepcopy__(self, memodict={}):
        copy_object = NodeMin()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.minimum(a, b)

    def __str__(self):
        return "min("+str(self.left)+","+str(self.right)+")"

class NodePow(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = False
    
    def __deepcopy__(self, memodict={}):
        copy_object = NodePow()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        if a==0 and b<=0:
            a = Node.VERY_SMALL
        return np.power(a, b)

    def is_allowed_right_argument(self, node_arg):
        if type(node_arg)!=type(NodeConstant(0)):
            return False
        if node_arg.value!=0.5 and node_arg.value!=-0.5 and node_arg.value!=round(node_arg.value):
            return False
        return True

    def is_allowed_left_argument(self, node_arg):
        if node_arg.contains_type(type(NodePow())) or node_arg.contains_type(type(NodeExp())): # TODO: avoid complicated bases
            return False
        if type(node_arg)==type(NodeConstant(0)) and node_arg.value==0:
            return False
        return True

    def __str__(self):
        return "pow("+str(self.left)+","+str(self.right)+")"

class NodeCos(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeCos()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.cos(a)

    def is_allowed_left_argument(self, node_arg): # avoid complicated expression
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "cos("+str(self.left)+")"

class NodeArcCos(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeArcCos()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.arccos(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and (node_arg.value<-1 or node_arg.value>1):
            return False
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "acos("+str(self.left)+")"

class NodeSin(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeSin()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.sin(a)
    
    def is_allowed_left_argument(self, node_arg):
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "sin("+str(self.left)+")"

class NodeTan(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeTan()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and (node_arg.value<-1 or node_arg.value>1):
            return False
        return True

    def evaluate_inner(self,X, a, b):
        return np.tan(a)

    def __str__(self):
        return "tan("+str(self.left)+")"

class NodeArcSin(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeArcSin()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.arcsin(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and (node_arg.value<-1 or node_arg.value>1):
            return False
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "asin("+str(self.left)+")"

class NodeArcTan(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeArcTan()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.arctan(a)

    def __str__(self):
        return "atan("+str(self.left)+")"

class NodeExp(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeExp()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.exp(a)

    def is_allowed_left_argument(self, node_arg): # avoid complicated expressions
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())) or node_arg.contains_type(type(NodeExp())) or node_arg.contains_type(type(NodeLn())) or node_arg.contains_type(type(NodePow())):
            return False
        return True

    def __str__(self):
        return "exp("+str(self.left)+")"

class NodeLn(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeLn()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        if type(a) is np.ndarray:
            _a = np.copy(a)
            _a[np.abs(_a) < Node.VERY_SMALL] = Node.VERY_SMALL
            return np.log(_a)
        
        # I am not sure if code:
        # if a==0:
        #   a = Node.VERY_SMALL
        # its good idea, if for example a == 1e-15 its not changed
        # but if its zero it changed to 0.0001
        if np.abs(a) < Node.VERY_SMALL:
            a = Node.VERY_SMALL

        # abs?
        return np.log(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and node_arg.value<=0:
            return False
        if type(node_arg)==type(NodeLn()) or type(node_arg)==type(NodeExp()):
            return False
        return True

    def __str__(self):
        return "log("+str(self.left)+")"

class NodeInv(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeInv()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        if type(a) is np.ndarray:
            _a = np.copy(a)
            _a[np.abs(_a) < Node.VERY_SMALL] = Node.VERY_SMALL
            return 1.0/_a

        # I am not sure if code:
        # if a==0:
        #   a = Node.VERY_SMALL
        # its good idea, if for example a == 1e-15 its not changed
        # but if its zero it changed to 0.0001
        if np.abs(a) < Node.VERY_SMALL:
            a = Node.VERY_SMALL

        return 1.0/a

    def is_allowed_left_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        return True

    def __str__(self):
        return "1/"+str(self.left)

class NodeSgn(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeSgn()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        if type(a) is np.ndarray:
            res = np.copy(a)
            res[np.abs(a) < 0] = -1.0
            res[np.abs(a) > 0] = 1.0
            return res

        if a<0:
            return -1
        elif a==0:
            return 0
        else:
            return 1

    def __str__(self):
        return "sgn("+str(self.left)+")"

class NodeSqr(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeSqr()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.power(a, 2) # abs(a)

    def __str__(self):
        return "pow("+str(self.left)+",2)"

class NodeSqrt(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeSqrt()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.sqrt(a) #abs(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and node_arg.value<0:
            return False
        return True

    def __str__(self):
        return "sqrt("+str(self.left)+")"

class NodeUnaryMinus(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeUnaryMinus()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return -a

    def __str__(self):
        return "(-"+str(self.left)+")"

class NodeAbs(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeAbs()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.absolute(a)

    def __str__(self):
        return "abs("+str(self.left)+")"

class NodeTan(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeTan()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.tan(a)

    def __str__(self):
        return "tan("+str(self.left)+")"

class NodeFloor(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeFloor()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.floor(a)

    def __str__(self):
        return "floor("+str(self.left)+")"

class NodeCeil(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeCeil()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return np.ceil(a)

    def __str__(self):
        return "ceiling("+str(self.left)+")"

class NodeInc(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeInc()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return a+1

    def __str__(self):
        return "("+str(self.left)+"+1)"

class NodeDec(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def __deepcopy__(self, memodict={}):
        copy_object = NodeDec()
        super().deepcopy(copy_object, memodict)
        return copy_object

    def evaluate_inner(self,X, a, b):
        return a-1

    def __str__(self):
        return "("+str(self.left)+"-1)"
