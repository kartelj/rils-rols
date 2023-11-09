#include "node.h"
#include <cassert>

vector<double> node::evaluate_all(const vector<vector<double>> &X) {
	int n = X.size();
	vector<double> left_vals, right_vals;
	if(arity>=1)
		left_vals = this->left->evaluate_all(X);
	if(arity==2)
		right_vals = this->right->evaluate_all(X);
	if(arity>2)
		throw new exception("Arity > 2 is not allowed.");
	return this->evaluate_inner(X, left_vals, right_vals);;
}

vector<node*> node::all_subtrees_references(node* root) {
	if (root == NULL)
		return vector<node*>();
	vector<node*> queue;
	queue.push_back(root);
	int pos = 0;
	while (pos<queue.size()) {
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

vector<node*> node::expand(){
	vector<node*> all_factors;
	if (type == node_type::PLUS || type == node_type::MINUS) {
		vector<node*> left_factors = left->expand();
		vector<node*> right_factors = right->expand();
		for (auto n : left_factors)
			all_factors.push_back(n);
		for (auto n : right_factors)
			all_factors.push_back(n);
	}
	else if(type!=node_type::CONST)
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
