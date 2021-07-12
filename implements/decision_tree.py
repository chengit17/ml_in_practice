import copy
import numpy as np
from abc import abstractmethod
from collections import Counter


class BaseNode:
    pass


class LeafNode(BaseNode):
    def __init__(self):
        self.label = None


class DiscreteAttrNode(BaseNode):
    def __init__(self, split_attr):
        super().__init__(split_attr)
        self.split_attr = None
        self.children = []

    def add_child(self, child, divide_val):
        self.children.append((child, divide_val))


class ContinousAttrNode(BaseNode):
    def __init__(self, split_attr, divide_val):
        super().__init__(self)
        self.split_attr = split_attr
        
        self.divide_val = divide_val

        self.left = None
        self.right = None

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node
        

def cal_entropy(y):
    n_samples = len(y)
    accum = 0
    for y_k in np.unique(y):
        p_k = y.count(y_k) / n_samples
        accum += p_k * np.log2(p_k)
    return -accum


def cal_gini(y):
    n_samples = len(y)
    ss = 0
    for y_k in np.unique(y):
        p_k = y.count(y_k) / n_samples
        ss += p_k**2
    return 1 - ss


class DecisionTreeCriterion:
    @abstractmethod
    def choose_best_attr_for_split(self, X, y, attrs, attrs_types):
        pass


class EntropyGainCriterion(DecisionTreeCriterion):
    def choose_best_attr_for_split(self, X, y, attrs, attrs_types):
        gains = []
        divide_vals = []
        for attr in attrs:
            if attrs_types[attr] == 'd':
                gain, divide_val = self.cal_discrete_attr_gain(X, y, attr), None
            elif attrs_types[attr] == 'c':
                gain, divide_val = self.cal_continous_attr_gain(X, y, attr)
            gains.append(gain)
            divide_vals.append(divide_val)
        best_attr_idx = np.asarray(gains).argmax()
        return attrs[best_attr_idx], divide_vals[best_attr_idx]
 
    def cal_discrete_attr_gain(self, X, y, attr_idx):
        raise NotImplementedError

    def cal_continous_attr_gain(self, X, y, attr_idx, ret_divice_val=True):
        raise NotImplementedError


class ID3(EntropyGainCriterion):
    @staticmethod
    def cal_discrete_attr_gain(X, y, attr_idx):
        ent_D = cal_entropy(y)

        ent_a = 0
        for v in np.unique(X[:, attr_idx]):
            mask = X[:, attr_idx] == v
            sub_y = y[mask]
            ent_a_v = cal_entropy(sub_y)
            ent_a += (np.count_nonzero(mask) / len(X)) * ent_a_v
        gain = ent_D - ent_a
        return gain

    @staticmethod
    def cal_continous_attr_gain(X, y, attr_idx, rtn_divide_val=True):
        ent_D = cal_entropy(y)
        
        attr_vals = X[:, attr_idx].sort()
        divide_vals = (attr_vals[1:] - attr_vals[:-1] / 2)
        ent_ds = []
        for d in divide_vals:
            mask_l = X[:, attr_idx] <= d
            mask_r = not mask_l
            y_l = y[mask_l]
            ent_l = cal_entropy(y_l)
            y_r = y[mask_r]
            ent_r = cal_entropy(y_r)
            ent_mean = (np.count_nonzero(mask_l) * ent_l + np.count_nonzero(mask_r) * ent_r) / len(X)
            ent_ds.append(ent_mean)
        min_idx = np.argmin(ent_ds)
        
        best_gain = ent_D - ent_ds[min_idx]
        if rtn_divide_val:
            best_divide_val = divide_vals[min_idx]
            return (best_gain, best_divide_val)
        return best_gain


class GiniCriterion(DecisionTreeCriterion):
    def choose_best_attr_for_split(self, X, y, attrs, attrs_types):
        ginis = []
        divide_vals = []
        for attr in attrs:
            if attrs_types[attr] == 'd':
                gini, divide_val = self.cal_discrete_attr_gini(X, y, attr), None
            elif attrs_types[attr] == 'c':
                gini, divide_val = self.cal_continous_attr_gini(X, y, attr)
            ginis.append(gini)
            divide_vals.append(divide_val)
        best_attr_idx = np.asarray(ginis).argmin()
        return attrs[best_attr_idx], divide_vals[best_attr_idx]
 
    def cal_discrete_attr_gini(self, X, y, attr_idx):
        raise NotImplementedError

    def cal_continous_attr_gini(self, X, y, attr_idx, ret_divice_val=True):
        raise NotImplementedError


class C45(EntropyGainCriterion):
    def cal_discrete_attr_gain(self, X, y, attr_idx):
        ent_D = cal_entropy(y)

        ent_a = 0
        n_vs = []
        for v in np.unique(X[:, attr_idx]):
            mask = X[:, attr_idx] == v
            sub_y = y[mask]
            n_v = np.count_nonzero(mask)
            ent_a += (n_v / len(X)) * cal_entropy(sub_y)
            n_vs.append(n_v)

        gain_a = ent_D - ent_a
        IV_a = cal_entropy(n_vs)
        penalty_gain = gain_a / IV_a
        return penalty_gain


class CART(GiniCriterion):
    def cal_discrete_attr_gini(self, X, y, attr_idx):
        gini_a = 0
        for v in np.unique(X[:, attr_idx]):
            mask = X[:, attr_idx] == v
            sub_y = y[mask]
            n_v = np.count_nonzero(mask)
            gini_a_v = cal_gini(sub_y)
            gini_a += (n_v / len(X)) * gini_a_v
        return gini_a

    def cal_continous_attr_gini(self, X, y, attr_idx, rtn_divide_val=True):
        attr_vals = X[:, attr_idx].sort()
        divide_vals = (attr_vals[1:] - attr_vals[:-1] / 2)
        gini_ds = []
        for d in divide_vals:
            mask_l = X[:, attr_idx] <= d
            mask_r = not mask_l
            y_l = y[mask_l]
            gini_l = cal_gini(y_l)
            y_r = y[mask_r]
            gini_r = cal_gini(y_r)
            gini_mean = (np.count_nonzero(mask_l) * gini_l + np.count_nonzero(mask_r) * gini_r) / len(X)
            gini_ds.append(gini_mean)
        min_idx = np.argmin(gini_ds)

        best_gini = gini_ds[min_idx]
        if rtn_divide_val:
            best_divide_val = divide_vals[min_idx]
            return (best_gini, best_divide_val)
        return best_gini


class DecistionTree:
    def __init__(self, attrs_types, criterion='C45', prune=None):
        self._attr_types = attrs_types
        self._root_node = None

        if criterion == 'C45':
            self._criterion = C45()
        elif criterion == 'ID3':
            self._criterion = ID3()
        elif criterion == 'CART':
            self._criterion = CART()

    def fit(self, X, y):
        num_attrs = X.shape[1]
        attrs = list(range(num_attrs))
        self._root_node = self._generate_tree(X, y, attrs, self._attr_types)

    def predict(self, X):
        preds = []
        for _, x in enumerate(X):
            preds.append(self._predict_x(x, self._root_node))    
        return np.asarray(preds)

    def _predict_x(self, x, node):
        if isinstance(node, LeafNode):
            return node.label

        if isinstance(node, ContinousAttrNode):
            if x[node.split_attr] <= divide_val:
                return self._predict_x(x, node.left)
            else:
                return self._predict_x(x, node.right)
        elif isinstance(node, DiscreteAttrNode):
            for child, divide_val in node.children:
                if x[node.split_attr] == divide_val:
                    return self._predict_x(x, child)

    def _generate_tree(self, X, y, attrs, attr_types):
        # 如果样本全属于同一类别,则将该类别赋给该node
        unique_labels = np.unique(y).flatten()
        if len(unique_labels) == 1:
            node = LeafNode()
            node.label = unique_labels[0]
            return node

        # 如果attrs为空
        if not any(attrs):
            # 将节点标记为叶节点,然后将取值最多的类别赋给该节点，然后返回
            node = LeafNode()
            node.label = self._find_the_label_with_most_samples(y)
            return node

        # 寻找最优划分属性
        split_attr, divide_val = self._criterion.choose_best_attr_for_split(X, y, attrs, attr_types)
        attrs = copy.copy(attrs)
        rest_attrs = attrs.remove(split_attr)

        # 遍历最优划分属性在样本中的每个取值
        if attr_types[split_attr] == 'd':
            node = DiscreteAttrNode(split_attr)

            for attr_val in X[:, split_attr]:
                sub_mask = X[:, split_attr] == attr_val
                sub_X = X[sub_mask, :]
                sub_y = y[sub_mask]
                if not np.any(sub_y):
                    child = LeafNode()
                    child.label = self._find_the_label_with_most_samples(y)
                    node.add_child(child, attr_val)
                else:
                    child = self._generate_tree(sub_X, sub_y, rest_attrs, attr_types)
                    node.add_child(child, attr_val)
            return node
        elif attr_types[split_attr] == 'c':
            node = ContinousAttrNode(split_attr, attr_val)

            mask_l = X[:, split_attr] <= attr_val
            for left in [True, False]:
                mask = mask_l if left else not mask_l

                sub_X = X[mask, :]
                sub_y = y[mask]

                if not np.any(sub_y):
                    child = LeafNode()
                    child.label = self._find_the_label_with_most_samples(y)
                else:
                    child = self._generate_tree(sub_X, sub_y, rest_attrs, attr_types)
                if left:
                    node.set_left(child)
                else:
                    node.set_right(child)
            return node

    def _find_the_label_with_most_samples(self, y):
        y_counted = Counter(y)
        y_counted = sorted(y_counted.items(), key=lambda kv : (kv[1], kv[0]), reverse=True)
        return y_counted[0][1]

    # def _check_samples_consistent_in_attrs(self, X, attrs):
    #     for attr in attrs:
    #         if len(np.unique(X[:, attr])) != 1:
    #             return False
    #     return True