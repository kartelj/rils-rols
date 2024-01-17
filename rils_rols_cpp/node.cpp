#include "node.h"
#include "eigen/Eigen/Dense"
#include <cassert>

Eigen::ArrayXd node::evaluate_all(const vector<Eigen::ArrayXd>& X) {
	if (arity < 0 || arity>2) {
		cout << "Arity > 2 or <0 is not allowed." << endl;
		exit(1);
	}
	const auto n = X.size();
	Eigen::ArrayXd yp(n), left_vals, right_vals;

	if (this->arity >= 1) 
		left_vals = this->left->evaluate_all(X);
	if (this->arity >= 2) {
		right_vals = this->right->evaluate_all(X);
	}
	
	yp = this->evaluate_inner(X, left_vals, right_vals);
	return yp;
}

Eigen::ArrayXd node::evaluate_inner(const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& a, const Eigen::ArrayXd& b) noexcept(false) {
	switch (type) {
	case node_type::CONST: {
		Eigen::ArrayXd const_arr(X[0].size());
		const_arr.fill(const_value);
		return const_arr;
	}
	case node_type::VAR: {
		return X[var_index];
	}
	case node_type::PLUS:
		return  a + b;
	case node_type::MINUS:
		return a - b;
	case node_type::MULTIPLY:
		return a * b;
	case node_type::DIVIDE:
		return a / b;
	case node_type::SIN:
		return a.sin();
	case node_type::COS:
		return a.cos();
	case node_type::LN:
		return a.log();
	case node_type::EXP:
		return a.exp();
	case node_type::SQRT:
		return a.sqrt();
	case node_type::SQR:
		return a*a;
	case node_type::POW:
		return a.pow(b);
	default:
		cout << "ERROR: Unrecognized operation.";
		exit(1);
	}
};

bool node::is_allowed_left(const node& node) const noexcept {
	const auto t = node.type;
	switch (type) {
	case node_type::EXP:
	case node_type::LN:
		if (t == node_type::EXP || t == node_type::LN)
			return false;
		break;
	case node_type::POW:
		if (t == node_type::POW)
			return false;
		break;
	case node_type::COS:
	case node_type::SIN:
		if (t == node_type::COS || t == node_type::SIN)
			return false;
		break;
	default:
		return true;
	}
	return true;
}

vector<shared_ptr<node>> node::extract_constants_references() {
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



void node::extract_non_constant_factors(vector<node*>& all_factors) {
	if (type == node_type::PLUS || type == node_type::MINUS) {
		left->extract_non_constant_factors(all_factors);
		right->extract_non_constant_factors(all_factors);
	}
	else if (type != node_type::CONST)
		all_factors.push_back(this);
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
			left = nullptr;
			right = nullptr;
		}
		else if (left->type == node_type::CONST)
		{
			if (type == node_type::PLUS || type == node_type::MINUS) {
				if (value_zero(left->const_value))
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
					set_const_value(0.0);
				else if (value_one(left->const_value))
					this->update_with(right);
				else if (right->type == node_type::MULTIPLY) {
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
				set_const_value(0.0);
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
				if (value_one(right->const_value))
					this->update_with(left);
				else if (value_zero(right->const_value))
					set_const_value(0.0);
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

static auto binomial_mult(const node* left, const node* right) {
	// (t1+-t2)*(t3+-t4) = t1*t3+-t1*t4+-t2*t3+-t2*t4
	const auto f1 = create_node_ptr(node_type::MULTIPLY, left->get_left(), right->get_left());
	const auto f2 = create_node_ptr(node_type::MULTIPLY, left->get_left(), right->get_right());
	const auto f3 = create_node_ptr(node_type::MULTIPLY, left->get_right(), right->get_left());
	const auto f4 = create_node_ptr(node_type::MULTIPLY, left->get_right(), right->get_right());
	const auto new_left = make_shared<node>(right->get_type(), f1, f2);
	const auto new_right = make_shared<node>(right->get_type(), f3, f4);
	return make_shared<node>(left->get_type(), new_left, new_right);
}

void node::expand() {
	// applies distributive laws recursively to expand expressions
	if (arity == 0) {
		return;
	}
	else if (arity == 1) {
		left->expand();
		//if (type == node_type::SQR){
		//	update_with(binomial_mult(get_left(), get_left()));
		//}
	}
	else {
		left->expand();
		right->expand();
		if (this->type == node_type::MULTIPLY) {
			if (left->type == node_type::PLUS || left->type == node_type::MINUS) {
				if (right->type == node_type::PLUS || right->type == node_type::MINUS) {
					update_with(binomial_mult(get_left(), get_right()));
				}
				else {
					// (t1+-t2)*t3 = t1*t3+-t2*t3
					left = create_node_ptr(node_type::MULTIPLY, left->get_left(), right.get());
					right = create_node_ptr(node_type::MULTIPLY, left->get_right(), right.get());
					set_type(left->type);
				}
			}
			else if (right->type == node_type::PLUS || right->type == node_type::MINUS) {
				// t1*(t2+-t3) to t1*t2+-t1*t3
				left = create_node_ptr(node_type::MULTIPLY, left.get(), right->get_left());
				right = create_node_ptr(node_type::MULTIPLY, left.get(), right->get_right());
				set_type(right->type);
			}
		}
	}
}

