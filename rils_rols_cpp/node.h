#pragma once
#include <vector>
#include <ostream>
#include <string>

using namespace std;

# define M_PI 3.14159265358979323846

enum node_type {
	CONST, 
	VAR, 
	PLUS, 
	MINUS, 
	MULTIPLY, 
	DIVIDE,
	SIN, 
	COS, 
	LN
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
	}

public:
	node* left;
	node* right;
	int arity;
	bool symmetric;
	node_type type;
	int var_index;
	double const_value;

	static node node_copy(const node& n) {
		node nc(n.type);
		nc.type = n.type;
		nc.arity = n.arity;
		nc.symmetric = n.symmetric;
		nc.const_value = n.const_value;
		nc.var_index = n.var_index;
		if(n.left != NULL)
			nc.left = new node(*n.left);
		if(n.right!=NULL)
			nc.right = new node(*n.right);
	}

	static node node_constant(double const_value) {
		node n(node_type::CONST);
		n.const_value = const_value;
		return n;
	}

	static node node_variable(int var_index) {
		node n(node_type::VAR);
		n.var_index = var_index;
		return n;
	}

	static node node_minus() { return node(node_type::MINUS); }

	static node node_plus() { return node(node_type::PLUS); }

	static node node_multiply() { return node(node_type::MULTIPLY); }

	static node node_divide() { return node(node_type::DIVIDE); }

	static node node_sin() { return node(node_type::SIN); }

	static node node_cos() { return node(node_type::COS); }

	static node node_ln() { return node(node_type::LN); }

	double evaluate_inner(vector<double> X, double a, double b) { 
		switch (type) {
		case node_type::CONST:
			return const_value;
		case node_type::VAR:
			return X[var_index];
		case node_type::PLUS:
			return  a + b;
		case node_type::MINUS:
			return a - b;
		case node_type::MULTIPLY:
			return a * b;
		case node_type::DIVIDE:
			return a / b;
		case node_type::SIN:
			return sin(a);
		case node_type::COS:
			return cos(a);
		case node_type::LN:
			return log(a);
		default:
			throw exception("Unrecognized operation.");
		}
	};

	string to_string() {
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
			return  "(" + left->to_string() + "*" + right->to_string() + ")";
		case node_type::DIVIDE:
			return  "(" + left->to_string() + "/" + right->to_string() + ")";
		case node_type::SIN:
			return "sin(" + left->to_string() + ")";
		case node_type::COS:
			return "cos(" + left->to_string() + ")";
		case node_type::LN:
			return "ln(" + left->to_string() + ")";
		default:
			throw exception("Unrecognized operation.");
		}
	};

	vector<double> evaluate_all(vector<vector<double>>);

	static vector<node*> all_subtrees_references(node* root);

	vector<node*> extract_constants_references();

	int size();
};
