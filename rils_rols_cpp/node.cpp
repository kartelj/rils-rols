#include "node.h"
#include <cassert>

vector<double> node::evaluate_all(vector<vector<double>> X) {
	int n = X.size();
	vector<double> yp(n), left_vals, right_vals;
	switch (this->arity) {
	case 0:
		for(int i=0; i<n; i++)
			yp[i] = this->evaluate_inner(X[i], NULL, NULL);
		break;
	case 1:
		left_vals = this->left->evaluate_all(X);
		for (int i = 0; i < n; i++) 
			yp[i] = this->evaluate_inner(X[i], left_vals[i], NULL);
		break;
	case 2:
		left_vals = this->left->evaluate_all(X);
		right_vals = this->right->evaluate_all(X);
		for(int i = 0; i < n; i++)
			yp[i] = this->evaluate_inner(X[i], left_vals[i], right_vals[i]);
		break;
	default:
		throw new exception("Arity > 2 is not allowed.");
	}
	return yp;
}

/*
vector<node*> node::all_subtrees_references() {
	if (this == NULL)
		return vector<node*>();
	vector<node*> all_st;
	all_st.push_back(this);
	if (this->arity == 0)
		return all_st;
	if (this->arity >= 1) {
		vector<node*> left_st = this->left->all_subtrees_references();
		for(int i=0; i<left_st.size(); i++)
			all_st.push_back(left_st[i]);
	}
	if (this->arity >= 2) {
		vector<node*> right_st = this->right->all_subtrees_references();
		for (int i = 0; i < right_st.size(); i++)
			all_st.push_back(right_st[i]);
	}
	return all_st;
}*/

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
