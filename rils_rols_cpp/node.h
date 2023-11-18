#pragma once
#include <iostream>
#include <vector>
#include <ostream>
#include <string>
#include "eigen/Eigen/Dense"

using namespace std;

#define M_PI 3.14159265358979323846

# define PRECISION 12
# define EPS pow(10, -PRECISION)

enum class node_type{
	NONE,
	CONST, 
	VAR, 
	PLUS, 
	MINUS, 
	MULTIPLY, 
	DIVIDE,
	SIN, 
	COS, 
	LN, 
	EXP, 
	SQRT, 
	SQR, 
	POW
};

class node : public std::enable_shared_from_this<node>
{

public:
	shared_ptr<node> left = NULL;
	shared_ptr<node> right = NULL;
	int arity;
	bool symmetric;
	node_type type;
	int var_index;
	double const_value;

	node() {
		left = NULL;
		right = NULL;
		arity = 0;
		const_value = 0;
		var_index = -1;
		symmetric = false;
		type = node_type::NONE;
	}

	void update_with(shared_ptr<node> src) {
		left = src->left;
		right = src->right;
		arity = src->arity;
		const_value = src->const_value;
		var_index = src->var_index;
		symmetric = src->symmetric;
		type = src->type;
	}

	node(node_type type) {
		this->type = type;
		switch (this->type) {
		case node_type::CONST:
		case node_type::VAR:
			this->arity = 0;
			break;
		case node_type::SIN:
		case node_type::COS:
		case node_type::LN:
		case node_type::EXP:
			case node_type::SQRT:
			case node_type::SQR:
			this->arity = 1;
			break;
		default:
			this->arity = 2;
		}
		switch (this->type) {
		case node_type::MINUS:
		case node_type::DIVIDE:
		case node_type::POW: // pow is set to be unary because the unary changes will take care of base, and the specific method is needed to take care for exponents -- add_pow_finetune
			this->symmetric = false;
			break;
		default:
			this->symmetric = true;
		}
		this->left = NULL;
		this->right = NULL;
		this->var_index = -1;
		this->const_value = 0;
	}

	static shared_ptr<node> node_copy(const node &n) {
		shared_ptr<node> nc = make_shared<node>(n.type);
		nc->type = n.type;
		nc->arity = n.arity;
		nc->symmetric = n.symmetric;
		nc->const_value = n.const_value;
		nc->var_index = n.var_index;
		if(n.left != NULL)
			nc->left = node_copy(*n.left);
		if(n.right!=NULL)
			nc->right = node_copy(*n.right);
		return nc;
	}

	static shared_ptr < node> node_internal(node_type type) {
		shared_ptr < node> n = make_shared< node>(type);
		return n;
	}

	static shared_ptr < node> node_constant(double const_value) {
		shared_ptr < node> n = make_shared< node>(node_type::CONST);
		n->const_value = const_value;
		return n;
	}

	static shared_ptr < node> node_variable(int var_index) {
		shared_ptr < node> n = make_shared< node>(node_type::VAR);
		n->var_index = var_index;
		return n;
	}

	static shared_ptr < node> node_minus() { return node_internal(node_type::MINUS); }

	static shared_ptr < node> node_plus() { return node_internal(node_type::PLUS); }

	static shared_ptr < node> node_multiply() { return node_internal(node_type::MULTIPLY); }

	static shared_ptr < node> node_divide() { return node_internal(node_type::DIVIDE); }

	static shared_ptr < node> node_sin() { return node_internal(node_type::SIN); }

	static shared_ptr < node> node_cos() { return node_internal(node_type::COS); }

	static shared_ptr < node> node_ln() { return node_internal(node_type::LN); }

	static shared_ptr < node> node_exp() { return node_internal(node_type::EXP); }

	static shared_ptr < node> node_sqrt() { return node_internal(node_type::SQRT); }

	static shared_ptr < node> node_sqr() { return node_internal(node_type::SQR); }

	static shared_ptr < node> node_pow() { return node_internal(node_type::POW); }

	Eigen::ArrayXd evaluate_inner(const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& a, const Eigen::ArrayXd& b) {
		switch (type) {
		case node_type::CONST: {
			Eigen::ArrayXd const_arr(X.size());
			const_arr.fill(const_value);
			return const_arr; 
		}
		case node_type::VAR: {
			Eigen::ArrayXd var_arr(X.size());
			for (int i = 0; i < X.size(); i++)
				var_arr[i] = X[i][var_index];
			return var_arr;
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
			return a * a;
		case node_type::POW:
			return a.pow(b);
		default:
			throw exception("Unrecognized operation.");
		}
	};

	inline string to_string() const {
		switch (type) {
		case node_type::CONST:
			return std::to_string(const_value);
		case node_type::VAR:
			return "x" + std::to_string(var_index);
		case node_type::PLUS:
			return  "(" + left->to_string() + "+" + right->to_string() + ")";
		case node_type::MINUS:
			return  "(" + left->to_string() + "-" + right->to_string() + ")";
		case node_type::MULTIPLY:
			return  left->to_string() + "*" + right->to_string();
		case node_type::DIVIDE:
			return  left->to_string() + "/" + right->to_string();
		case node_type::SIN:
			return "sin(" + left->to_string() + ")";
		case node_type::COS:
			return "cos(" + left->to_string() + ")";
		case node_type::LN:
			return "ln(" + left->to_string() + ")";
		case node_type::EXP:
			return "exp(" + left->to_string() + ")";
		case node_type::SQRT:
			return "sqrt(" + left->to_string() + ")";
		case node_type::SQR:
			return left->to_string() + "*" + left->to_string();
		case node_type::POW:
			return "pow("+left->to_string() + "," + right->to_string()+")";
		default:
			return "*****UNKNOWN*****";
		}
	};

	bool is_allowed_left(const node& node) const{
		node_type t = node.type;
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
	}

	Eigen::ArrayXd evaluate_all(const vector<Eigen::ArrayXd>& X);

	static vector< shared_ptr<node>> all_subtrees_references(shared_ptr < node> root);

	vector< shared_ptr<node>> extract_constants_references();

	vector< shared_ptr<node>> extract_non_constant_factors();

	int size();

	void simplify();

	void normalize_constants(node_type parent_type);

	void normalize_factor_constants(node_type parent_type, bool inside_factor);

	void expand();
};

// TODO: define these two as macros to speed up
inline bool value_zero(double val) {
	return abs(val) < EPS;
}

inline bool value_one(double val) {
	return abs(val - 1) < EPS;
}
