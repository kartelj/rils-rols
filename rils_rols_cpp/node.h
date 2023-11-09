#pragma once
#include <vector>
#include <ostream>
#include <string>

using namespace std;

# define M_PI 3.14159265358979323846

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
	SQR
};

class node
{
private:

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

public:
	node* left;
	node* right;
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

	static node* node_copy(const node &n) {
		node* nc = new node(n.type);
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

	static node* node_internal(node_type type) {
		node* n = new node(type);
		return n;
	}

	static node* node_constant(double const_value) {
		node* n = new node(node_type::CONST);
		n->const_value = const_value;
		return n;
	}

	static node* node_variable(int var_index) {
		node* n = new node(node_type::VAR);
		n->var_index = var_index;
		return n;
	}

	static node* node_minus() { return node_internal(node_type::MINUS); }

	static node* node_plus() { return node_internal(node_type::PLUS); }

	static node* node_multiply() { return node_internal(node_type::MULTIPLY); }

	static node* node_divide() { return node_internal(node_type::DIVIDE); }

	static node* node_sin() { return node_internal(node_type::SIN); }

	static node* node_cos() { return node_internal(node_type::COS); }

	static node* node_ln() { return node_internal(node_type::LN); }

	static node* node_exp() { return node_internal(node_type::EXP); }

	static node* node_sqrt() { return node_internal(node_type::SQRT); }

	static node* node_sqr() { return node_internal(node_type::SQR); }

	const vector<double>& evaluate_inner(const vector < vector< double> > & X, const vector<double>& a, const vector<double>& b) {
		int n = X.size();
		vector<double> yp(n);
		switch (type) {
		case node_type::CONST:
			for (int i = 0; i < n; i++)
				yp[i] = const_value;
			break;
		case node_type::VAR:
			for (int i = 0; i < n; i++)
				yp[i] = X[i][var_index];
			break;
		case node_type::PLUS:
			for (int i = 0; i < n; i++)
				yp[i] = a[i] + b[i];
			break;
		case node_type::MINUS:
			for (int i = 0; i < n; i++)
				yp[i] = a[i] - b[i];
			break;
		case node_type::MULTIPLY:
			for (int i = 0; i < n; i++)
				yp[i] = a[i] * b[i];
			break;
		case node_type::DIVIDE:
			for (int i = 0; i < n; i++)
				yp[i] = a[i] / b[i];
			break;
		case node_type::SIN:
			for (int i = 0; i < n; i++)
				yp[i] = sin(a[i]);
			break;
		case node_type::COS:
			for (int i = 0; i < n; i++)
				yp[i] = cos(a[i]);
			break;
		case node_type::LN:
			for (int i = 0; i < n; i++)
				yp[i] = log(a[i]);
			break;
		case node_type::EXP:
			for (int i = 0; i < n; i++)
				yp[i] = exp(a[i]);
			break;
		case node_type::SQRT:
			for (int i = 0; i < n; i++)
				yp[i] = sqrt(a[i]);
			break;
		case node_type::SQR:
			for (int i = 0; i < n; i++)
				yp[i] = a[i]*a[i];
			break;
		default:
			throw exception("Unrecognized operation.");
		}
		return yp;
	};

	string to_string() const {
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
		default:
			throw exception("Unrecognized operation.");
		}
	};

	vector<double> evaluate_all(const vector<vector<double>> &X);

	static vector<node*> all_subtrees_references(node* root);

	vector<node*> extract_constants_references();

	vector<node*> expand();

	int size();
};
