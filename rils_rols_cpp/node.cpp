#include "node.h"
#include "eigen/Eigen/Dense"
#include <cassert>

Eigen::ArrayXd node::evaluate_all(const vector<Eigen::ArrayXd>& X) {
	if (arity < 0 || arity>2) {
		cout << "Arity > 2 or <0 is not allowed." << endl;
		exit(1);
	}
	int n = X.size();
	Eigen::ArrayXd yp(n), left_vals, right_vals;

	if (this->arity >= 1) 
		left_vals = this->left->evaluate_all(X);
	if (this->arity >= 2) {
		right_vals = this->right->evaluate_all(X);
	}
	
	yp = this->evaluate_inner(X, left_vals, right_vals);
	return yp;
}

vector< shared_ptr<node>> node::all_subtrees_references(shared_ptr<node> root) {
	if (root == NULL)
		return vector<shared_ptr<node>>();
	vector<shared_ptr<node>> queue;
	queue.push_back(root);
	int pos = 0;
	while (pos < queue.size()) {
		// adding children of current element
		shared_ptr<node> curr = queue[pos];
		if (curr->left != NULL)
			queue.push_back(curr->left);
		if (curr->right != NULL)
			queue.push_back(curr->right);
		pos++;
	}
	assert(queue.size() == root->size());
	return queue;
}

vector<shared_ptr<node>> node::extract_constants_references() {
	if (this == NULL)
		return vector<shared_ptr<node>>();
	vector<shared_ptr<node>> all_cons;
	if (this->type == node_type::CONST) {
		all_cons.push_back(shared_from_this());
	}
	if (this->arity >= 1) {
		vector<shared_ptr<node>> left_cons = this->left->extract_constants_references();
		for (int i = 0; i < left_cons.size(); i++)
			all_cons.push_back(left_cons[i]);
	}
	if (this->arity >= 2) {
		vector<shared_ptr<node>> right_cons = this->right->extract_constants_references();
		for (int i = 0; i < right_cons.size(); i++)
			all_cons.push_back(right_cons[i]);
	}
	return all_cons;
}



vector<shared_ptr<node>> node::extract_non_constant_factors() {
	vector<shared_ptr<node>> all_factors;
	if (type == node_type::PLUS || type == node_type::MINUS) {
		vector<shared_ptr<node>> left_factors = left->extract_non_constant_factors();
		vector<shared_ptr<node>> right_factors = right->extract_non_constant_factors();
		for (int i = 0; i < left_factors.size(); i++)
			all_factors.push_back(left_factors[i]);
		for (int i=0; i<right_factors.size(); i++)
			all_factors.push_back(right_factors[i]);
	}
	else if (type != node_type::CONST)
		all_factors.push_back(shared_from_this());
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
			else if (type == node_type::POW)
				const_value = pow(left->const_value, right->const_value);
			else {
				cout << "Simplification is not supported for this binary operator!" << endl;
				exit(1);
			}
			arity = 0;
			type = node_type::CONST;
			left = NULL;
			right = NULL;
		}
		else if (left->type == node_type::CONST)
		{
			if (type == node_type::PLUS || type==node_type::MINUS) {
				if(value_zero(left->const_value))
					this->update_with(right); // 0+t = t || 0-t = -t
				else if (right->type == node_type::PLUS || right->type == node_type::MINUS) {
					if (right->left->type == node_type::CONST) {
						// c1+-(c2+-t) = c3+-t [c3 = c1+-c2]
						if (type == node_type::PLUS)
							left->const_value += right->left->const_value;
						else
							left->const_value -= right->left->const_value;
						right->update_with(right->right);
					}
					else if (right->right->type == node_type::CONST) {
						// c1+-(t+-c2) = c3+-t [c3 = c1+-c2]
						if (type == node_type::PLUS) {
							if (right->type == node_type::PLUS)
								left->const_value += right->right->const_value;
							else
								left->const_value -= right->right->const_value;
						}
						else {
							if (right->type == node_type::PLUS)
								left->const_value -= right->right->const_value;
							else
								left->const_value += right->right->const_value;
						}
						right->update_with(right->left);
					}
				}
			}
			else if (type == node_type::MULTIPLY) {
				if (value_zero(left->const_value))
					*this = *node::node_constant(0);
				else if (value_one(left->const_value))
					this->update_with(right);
				else if(right->type == node_type::MULTIPLY) {
					if (right->left->type == node_type::CONST) {
						// c1*c2*expr = c3*expr [c3 = c1*c2]
						left->const_value *= right->left->const_value;
						right->update_with(right->right);
					}
					else if (right->right->type == node_type::CONST) {
						// c1*expr*c2 = c3*expr [c3 = c1*c2]
						left->const_value *= right->right->const_value;
						right->update_with(right->left);
					}
				}
			}
			else if (type == node_type::DIVIDE && value_zero(left->const_value))
				this->update_with(node::node_constant(0));
		}
		else if (right->type == node_type::CONST)
		{
			if (type == node_type::PLUS || type == node_type::MINUS) {
				if (value_zero(right->const_value))
					this->update_with(left);
				else if (left->type == node_type::PLUS || left->type == node_type::MINUS) {
					if (left->left->type == node_type::CONST) {
						// (c1+-t)+-c2 = c3+-t+-0 [c3 = c1+-c2]  // this 0 will be further removed with next simplification call
						if (type == node_type::PLUS)
							left->left->const_value += right->const_value;
						else
							left->left->const_value -= right->const_value;
						right->const_value = 0;
					}
					else if (left->right->type == node_type::CONST) {
						// (t+-c1)+-c2 = (t+-0)+-c3 [c3 = c1+-c2] 
						if (left->type == node_type::PLUS)
							right->const_value += left->right->const_value;
						else
							right->const_value -= left->right->const_value;
						left->right->const_value = 0;
					}
				}
			}else if (type == node_type::MULTIPLY) {
				if(value_one(right->const_value))
					this->update_with(left);
				else if (value_zero(right->const_value))
					this->update_with(node::node_constant(0));
				else {
					// some more exotic variants  
					if (left->type == node_type::MULTIPLY) {
						if (left->left->type == node_type::CONST) {
							// c1*expr*c2 = expr*c3 [c3 = c1*c2]
							right->const_value *= left->left->const_value;
							left->update_with(left->right);
						}
						else if (left->right->type == node_type::CONST) {
							// expr*c1*c2 = expr*c3 [c3 = c1*c2]
							right->const_value *= left->right->const_value;
							left->update_with(left->left);
						}
					}
				}
			}
		}
	}
}

void node::normalize_constants(node_type parent_type) {
	if (type == node_type::CONST) {
		if (parent_type == node_type::NONE || parent_type == node_type::MULTIPLY || parent_type == node_type::DIVIDE || parent_type == node_type::PLUS || parent_type == node_type::MINUS)
			const_value = 1;
		else if (parent_type == node_type::POW && type == node_type::CONST && const_value != 0.5 && const_value != -0.5)
			const_value = round(const_value);
		return;
	}
	if (arity >= 1)
		left->normalize_constants(type);
	if (arity >= 2)
		right->normalize_constants(type);
}

void node::normalize_factor_constants(node_type parent_type, bool inside_factor) {
	if (type == node_type::CONST) 
		const_value = 1;
	else if (!inside_factor && (type == node_type::PLUS || type == node_type::MINUS)) {
		left->normalize_factor_constants(type, false);
		right->normalize_factor_constants(type, false);
	}
	else if (!inside_factor) {
		if (type == node_type::MULTIPLY) {
			left->normalize_factor_constants(type, true);
			right->normalize_factor_constants(type, true);
		}
		else if (type == node_type::DIVIDE)
			right->normalize_factor_constants(type, true);
	}
}

shared_ptr<node> binomial_mult(shared_ptr<node> left, shared_ptr<node> right) {
	// (t1+-t2)*(t3+-t4) = t1*t3+-t1*t4+-t2*t3+-t2*t4
	shared_ptr<node> f1 = node::node_multiply();
	f1->left = node::node_copy(*left->left);
	f1->right = node::node_copy(*right->left);
	shared_ptr<node> f2 = node::node_multiply();
	f2->left = node::node_copy(*left->left);
	f2->right = node::node_copy(*right->right);
	shared_ptr<node> f3 = node::node_multiply();
	f3->left = node::node_copy(*left->right);
	f3->right = node::node_copy(*right->left);
	shared_ptr<node> f4 = node::node_multiply();
	f4->left = node::node_copy(*left->right);
	f4->right = node::node_copy(*right->right);
	shared_ptr<node> new_left = NULL;
	shared_ptr<node> new_right = NULL;
	shared_ptr<node> result = NULL;
	if (left->type == node_type::PLUS)
		result = node::node_plus();
	else
		result = node::node_minus();
	if (right->type == node_type::PLUS) {
		new_left = node::node_plus();
		new_right = node::node_plus();
	}
	else {
		new_left = node::node_minus();
		new_right = node::node_minus();
	}
	new_left->left = f1;
	new_left->right = f2;
	new_right->left = f3;
	new_right->right = f4;
	result->left = new_left;
	result->right = new_right;
	return result;
}

void node::expand() {
	// applies distributive laws recursively to expand expressions
	if (arity == 0) {
		return;
	}
	else if (arity == 1) {
		left->expand();
		if (type == node_type::SQR && left->arity==2){
			*this = *binomial_mult(left, left);
		}
	}
	else {
		left->expand();
		right->expand();
		if (this->type == node_type::MULTIPLY) {
			if (left->type == node_type::PLUS || left->type == node_type::MINUS) {
				if (right->type == node_type::PLUS || right->type == node_type::MINUS) {
					*this = *binomial_mult(left, right);
				}
				else {
					// (t1+-t2)*t3 = t1*t3+-t2*t3
					shared_ptr<node> new_left = node::node_multiply();
					new_left->left = node::node_copy(*left->left);
					new_left->right = node::node_copy(*right);
					shared_ptr<node> new_right = node::node_multiply();
					new_right->left = node::node_copy(*left->right);
					new_right->right = node::node_copy(*right);
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
				shared_ptr<node> new_left = node::node_multiply();
				new_left->left = node::node_copy(*left);
				new_left->right = node::node_copy(*right->left);
				shared_ptr<node> new_right = node::node_multiply();
				new_right->left = node::node_copy(*left);
				new_right->right = node::node_copy(*right->right);
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

