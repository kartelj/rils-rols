#include "node.h"
#include "eigen/Eigen/Dense"
#include <cassert>

Eigen::ArrayXd node::evaluate_all(const vector<Eigen::ArrayXd>& X) {
	int n = X.size();
	Eigen::ArrayXd yp(n), left_vals, right_vals;
	switch (this->arity) {
	case 0:
		break;
	case 1:
		left_vals = this->left->evaluate_all(X);
		break;
	case 2:
		right_vals = this->right->evaluate_all(X);
		left_vals = this->left->evaluate_all(X);
		break;
	default:
		throw new exception("Arity > 2 is not allowed.");
	}
	yp = this->evaluate_inner(X, left_vals, right_vals);
	return yp;
}

vector<node*> node::all_subtrees_references(node* root) {
	if (root == NULL)
		return vector<node*>();
	vector<node*> queue;
	queue.push_back(root);
	int pos = 0;
	while (pos < queue.size()) {
		// adding children of current element
		node* curr = queue[pos];
		if (curr->left != NULL)
			queue.push_back(curr->left);
		if (curr->right != NULL)
			queue.push_back(curr->right);
		pos++;
	}
	assert(queue.size() == root->size());
	return queue;
}

vector<node*> node::extract_constants_references() {
	if (this == NULL)
		return vector<node*>();
	vector<node*> all_cons;
	if (this->type == node_type::CONST) {
		all_cons.push_back(this);
	}
	if (this->arity >= 1) {
		vector<node*> left_cons = this->left->extract_constants_references();
		for (int i = 0; i < left_cons.size(); i++)
			all_cons.push_back(left_cons[i]);
	}
	if (this->arity >= 2) {
		vector<node*> right_cons = this->right->extract_constants_references();
		for (int i = 0; i < right_cons.size(); i++)
			all_cons.push_back(right_cons[i]);
	}
	return all_cons;
}



vector<node*> node::extract_non_constant_factors() {
	vector<node*> all_factors;
	if (type == node_type::PLUS || type == node_type::MINUS) {
		vector<node*> left_factors = left->extract_non_constant_factors();
		vector<node*> right_factors = right->extract_non_constant_factors();
		for (auto n : left_factors)
			all_factors.push_back(n);
		for (auto n : right_factors)
			all_factors.push_back(n);
	}
	else if (type != node_type::CONST)
		all_factors.push_back(this);
	return all_factors;
}

int node::size() {
	if (this == NULL)
		return 0;
	int size = 1;
	if (this->arity >= 1)
		size += this->left->size();
	if (this->arity >= 2)
		size += this->right->size();
	return size;
}

// Perform trivial simplifications: 0*a => 0, 1*a => a, a/1 => a,
//    a+0 => a, a-0 => a  c1 op c2 => c3  where a is any expression,
//    the ci are constants, and op is any of * / + -
void node::simplify()
{
	if (arity == 0) {
		return;
	}
	else if (arity == 1)
		left->simplify();
	else {
		left->simplify();
		right->simplify();
		if (left->type == node_type::CONST && right->type == node_type::CONST)
		{
			// Both operands are constants, evaluate the operation
			// and change this expression to a constant for the result.
			if (type == node_type::PLUS)
				const_value = left->const_value + right->const_value;
			else if (type == node_type::MINUS)
				const_value = left->const_value - right->const_value;
			else if (type == node_type::MULTIPLY)
				const_value = left->const_value * right->const_value;
			else if (type == node_type::DIVIDE)
				const_value = left->const_value / right->const_value;
			arity = 0;
			type = node_type::CONST;
			left = right = NULL;
		}
		else if (left->type == node_type::CONST)
		{
			if (type == node_type::PLUS) {
				if(left->const_value == 0)
					*this = *right;
			}
			else if (type == node_type::MULTIPLY) {
				if (left->const_value == 0)
					*this = *node::node_constant(0);
				else if(left->const_value == 1)
					*this = *right;
				else {
					// some more exotic variants
					if (right->type == node_type::MULTIPLY) {
						if (right->left->type == node_type::CONST) {
							// c1*c2*expr = c3*expr [c3 = c1*c2]
							left->const_value *= right->left->const_value;
							*right = *right->right;
						}
						else if (right->right->type == node_type::CONST) {
							// c1*expr*c2 = c3*expr [c3 = c1*c2]
							left->const_value *= right->right->const_value;
							*right = *right->left;
						}
					}
				}
			}
			else if (type == node_type::DIVIDE && left->const_value == 0)
				*this = *node::node_constant(0);
		}
		else if (right->type == node_type::CONST)
		{
			if (type == node_type::PLUS && right->const_value == 0)
				*this = *left;
			else if (type == node_type::MINUS && right->const_value == 0)
				*this = *left;
			else if (type == node_type::MULTIPLY) {
				if(right->const_value == 1)
					*this = *left;
				else if (right->const_value == 0)
					*this = *node::node_constant(0);
				else {
					// some more exotic variants  
					if (left->type == node_type::MULTIPLY) {
						if (left->left->type == node_type::CONST) {
							// c1*expr*c2 = expr*c3 [c3 = c1*c2]
							right->const_value *= left->left->const_value;
							*left = *left->right;
						}
						else if (left->right->type == node_type::CONST) {
							// expr*c1*c2 = expr*c3 [c3 = c1*c2]
							right->const_value *= left->right->const_value;
							*left = *left->left;
						}
					}
				}
			}
		}
	}
}

void node::expand() {
	// applies distributive laws recursively to expand expressions
	if (arity == 0) {
		return;
	}
	else if (arity == 1)
		left->expand(); // TODO: expand sqr
	else {
		left->expand();
		right->expand();
		if (this->type == node_type::MULTIPLY) {
			if (left->type == node_type::PLUS || left->type == node_type::MINUS) {

				if (right->type == node_type::PLUS || right->type == node_type::MINUS) {
					// (t1+-t2)*(t3+-t4) = t1*t3+-t1*t4+-t2*t3+-t2*t4
				}
				else {
					// (t1+-t2)*t3 = t1*t3+-t2*t3
					node* new_left = node::node_multiply();
					new_left->left = node::node_copy(*left->left);
					new_left->right = node::node_copy(*right);
					node* new_right = node::node_multiply();
					new_right->left = node::node_copy(*left->right);
					new_right->right = node::node_copy(*right);
					//delete left;
					//delete right;
					if (left->type == node_type::PLUS)
						*this = *node::node_plus();
					else
						*this = *node::node_minus();
					left = new_left;
					right = new_right;
				}
			}
			else if (right->type == node_type::PLUS || right->type == node_type::MINUS) {
				// t1*(t2+-t3) to t1*t2+-t1*t3
				node* new_left = node::node_multiply();
				new_left->left = node::node_copy(*left);
				new_left->right = node::node_copy(*right->left);
				node* new_right = node::node_multiply();
				new_right->left = node::node_copy(*left);
				new_right->right = node::node_copy(*right->right);
				//delete left;
				//delete right;
				if (right->type == node_type::PLUS)
					*this = *node::node_plus();
				else
					*this = *node::node_minus();
				left = new_left;
				right = new_right;
			}
		}
	}
}

