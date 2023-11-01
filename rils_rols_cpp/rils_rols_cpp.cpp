// rils_rols_cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <tuple>
#include <algorithm>
#include "node.h"
#include <cassert>

using namespace std;

enum fitness_type
{
	BIC = 1,
	SRM = 2,
	PENALTY = 3,
	JACCARD = 4
};

template<int M, template<typename> class F = std::less>
struct TupleCompare
{
	template<typename T>
	bool operator()(T const& t1, T const& t2)
	{
		return F<typename tuple_element<M, T>::type>()(std::get<M>(t1), std::get<M>(t2));
	}
};

class rils_rols {

private:
	// control parameters
	int max_fit_calls, max_seconds, random_state;
	double complexity_penalty, initial_sample_size;
	bool random_perturbations_order, verbose;
	fitness_type fitness_type;

	// internal stuff
	int ls_it, main_it, last_improved_it, time_start, time_elapsed;

	void reset() {
		ls_it = 0;
		main_it = 0;
		last_improved_it = 0;
		time_start = 0;
		time_elapsed = 0;
		srand(random_state);
	}

	vector<node> allowed_nodes;

	void setup_nodes(int var_cnt) {
		allowed_nodes.push_back(node::node_plus());
		allowed_nodes.push_back(node::node_minus());
		allowed_nodes.push_back(node::node_multiply());
		allowed_nodes.push_back(node::node_divide());
		allowed_nodes.push_back(node::node_sin());
		allowed_nodes.push_back(node::node_cos());
		allowed_nodes.push_back(node::node_ln());
		double constants[] = {-1, 0, 0.5, 2, M_PI, 10};
		for (auto c : constants)
			allowed_nodes.push_back(node::node_constant(c));
		for (int i = 0; i < var_cnt; i++)
			allowed_nodes.push_back(node::node_variable(i));
	}

	vector<node> perturb_candidates(node* old_node, node* parent, bool is_left_from_parent) {
		vector<node> candidates;
		if (old_node->arity >= 1) {
			// change node to its left subtree
			node* left_c = new node(*old_node->left);
			candidates.push_back(*left_c);
		}

		if (old_node->arity >= 2) {
			// change node to its right subtree
			node* right_c = new node(*old_node->right);
			candidates.push_back(*right_c);
		}

		if (old_node->arity==0) {
			// change constant or variable to variable
			for (auto &n : allowed_nodes) {
				if (n.arity != 0)
					continue;
				node* n_c = new node(n);
				candidates.push_back(*n_c);
			}
		}
		if (old_node->type == node_type::VAR) {
			// change variable to unary operation applied to that variable
			for (auto& n_un : allowed_nodes) {
				if (n_un.arity != 1)
					continue;
				node* n_un_c = new node(n_un);
				node* old_node_c = new node(*old_node);
				n_un_c->left = old_node_c;
				candidates.push_back(*n_un_c);
			}
		}
		if (old_node->arity == 1) {
			// change unary operation to another unary operation
			for (auto& n_un : allowed_nodes) {
				if (n_un.arity != 1 || n_un.type==old_node->type)
					continue;
				node* n_un_c = new node(n_un);
				node* old_left_c = new node(*old_node->left);
				n_un_c->left = old_left_c;
				candidates.push_back(*n_un_c);
			}
		}
		if (old_node->arity == 2) {
			// change one binary operation to another
			for (auto& n_bin : allowed_nodes) {
				if (n_bin.arity != 2 || n_bin.type==old_node->type)
					continue;
				node* n_bin_c = new node(n_bin);
				node* old_left_c = new node(*old_node->left);
				node* old_right_c = new node(*old_node->right);
				n_bin_c->left = old_left_c;
				n_bin_c->right = old_right_c;
				candidates.push_back(*n_bin_c);
				if (!n_bin.symmetric) {
					node* n_bin_c = new node(n_bin);
					node* old_left_c = new node(*old_node->left);
					node* old_right_c = new node(*old_node->right);
					n_bin_c->right = old_left_c;
					n_bin_c->left = old_right_c;
					candidates.push_back(*n_bin_c);
				}
			}
		}
		if (old_node->arity == 0) {
			// change variable or constant to binary operation with some variable  -- increases the model size
			for (auto& n_bin : allowed_nodes) {
				if (n_bin.arity != 2)
					continue;
				for (auto& n_var : allowed_nodes) {
					if (n_var.type != node_type::VAR)
						continue;
					node* n_bin_c = new node(n_bin);
					node* old_node_c = new node(*old_node);
					node* n_var_c = new node(n_var);
					n_bin_c->left = old_node_c;
					n_bin_c->right = n_var_c;
					candidates.push_back(*n_bin_c);
					if (!n_bin.symmetric) {
						n_bin_c = new node(n_bin);
						old_node_c = new node(*old_node);
						n_var_c = new node(n_var);
						n_bin_c->right = old_node_c;
						n_bin_c->left = n_var_c;
						candidates.push_back(*n_bin_c);
					}
				}
			}
		}
		// TODO: check for duplicates
		return candidates;
	}

	vector<node> all_perturbations(node passed_solution) {
		node* solution = new node(passed_solution);
		vector<node> all_pert;
		vector<node*> all_subtrees = node::all_subtrees_references(solution);
		//TODO: implemente BFS here and traverse subtrees during that
		for (int i = 0; i < all_subtrees.size(); i++) {
			if (all_subtrees[i]->size() == solution->size()) {
				// the whole tree is being changed
				vector<node> candidates = perturb_candidates(all_subtrees[i], NULL, false);
				for (auto& cand : candidates)
					all_pert.push_back(cand);
			}
			if (all_subtrees[i]->arity >= 1) {
				// the left subtree is being changed
				vector<node> candidates = perturb_candidates(all_subtrees[i]->left, all_subtrees[i], true);
				node* old_left = new node(*(all_subtrees[i]->left));
				int old_arity = all_subtrees[i]->arity;
				for (auto& cand : candidates) {
					node* cand_c = new node(cand);
					all_subtrees[i]->left = cand_c;
					all_subtrees[i]->arity = 10;
					all_subtrees[i]->arity = old_arity;
					node* solution_copy = new node(*solution);
					all_pert.push_back(*solution_copy);
				}
				all_subtrees[i]->left = old_left;
			}
			if (all_subtrees[i]->arity >= 2) {
				// the right subtree is being changed
				vector<node> candidates = perturb_candidates(all_subtrees[i]->right, all_subtrees[i], false);
				node* old_right = new node(*(all_subtrees[i]->right));
				for (auto& cand : candidates) {
					node* cand_c = new node(cand);
					all_subtrees[i]->right = cand_c;
					node* solution_copy = new node(*solution);
					all_pert.push_back(*solution_copy);
				}
				all_subtrees[i]->right = old_right;
			}
		}
		return all_pert;
	}

	node local_search(node solution, vector<vector<double>> X, vector<double> y) {
		node* best_solution = new node(solution);
		tuple<double, double, int> best_fit = fitness(*best_solution, X, y);
		bool impr = true;
		while (impr) {
			impr = false;

		}
		return solution;
	}

	node tune_constants(node solution, vector<vector<double>> X, vector<double> y) {
		node* best_solution = new node(solution);
		tuple<double, double, int> best_fitness = fitness(*best_solution, X, y);
		double multipliers[] = {0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 2, 5, 10, 20, 50, 100};
		bool impr = true;
		if (solution.size() > 6)
			cout << "bla" << endl;
		while (impr) {
			impr = false;
			vector<node*> constants = best_solution->extract_constants_references();
			int best_i = -1;
			double best_value = 1;
			for (int i = 0; i < constants.size(); i++) {
				double old_value = constants[i]->const_value;
				for (auto m : multipliers) {
					// TODO: multiplier does not make sense if old_value is zero
					constants[i]->const_value = old_value * m;
					tuple<double, double, int> new_fitness = fitness(*best_solution, X, y);
					if (compare_fitness(new_fitness, best_fitness)<0) {
						best_fitness = new_fitness;
						best_i = i;
						best_value = constants[i]->const_value;
					}
				}
				constants[i]->const_value = old_value;
			}
			if (best_i != -1) {
				impr = true;
				constants[best_i]->const_value = best_value;
				tuple<double, double, int> test_fitness = fitness(*best_solution, X, y);
				assert(compare_fitness(test_fitness, best_fitness) == 0);
				//cout << "Improved to " << best_solution->to_string() << endl;
			}
		}
		return *best_solution;
	}

	double R2(vector<double> y, vector<double> yp) {
		double y_avg = 0;
		for (auto yi : y)
			y_avg += yi;
		y_avg /= y.size();
		double ssr = 0, sst = 0;
		for (int i = 0; i < y.size(); i++) {
			ssr += pow(y[i] - yp[i], 2);
			sst += pow(y[i] - y_avg, 2);
		}
		return 1 - ssr / sst;
	}

	double RMSE(vector<double> y, vector<double> yp) {
		double rmse = 0;
		for (int i = 0; i < y.size(); i++)
			rmse += pow(y[i] - yp[i], 2);
		return sqrt(rmse / y.size());
	}

	tuple<double, double, int> fitness(node solution, vector<vector<double>> X, vector<double> y) {
		vector<double> yp = solution.evaluate_all(X);
		double r2 = R2(y, yp);
		double rmse = RMSE(y, yp);
		int size = solution.size();
		return tuple<double, double, int>{1-r2, rmse, size};
	}

public:
	rils_rols(int max_fit_calls, int max_seconds, enum fitness_type fitness_type, double complexity_penalty, double initial_sample_size, bool random_perturbations_order, bool verbose, int random_state) {
		this->max_fit_calls = max_fit_calls;
		this->max_seconds = max_seconds;
		this->fitness_type = fitness_type;
		this->complexity_penalty = complexity_penalty;
		this->initial_sample_size = initial_sample_size;
		this->random_perturbations_order = random_perturbations_order;
		this->verbose = verbose;
		this->random_state = random_state;
	}

	double penalty_fitness(tuple<double, double, int> fit) {
		return (1 + get<0>(fit)) * (1 + get<1>(fit)) * (1 + get<2>(fit) * this->complexity_penalty);
	}

	int compare_fitness(tuple<double, double, int> fit1, tuple<double, double, int> fit2) {
		double fit1_tot, fit2_tot;
		switch (this->fitness_type)
		{
		case fitness_type::PENALTY:
			fit1_tot = penalty_fitness(fit1);
			fit2_tot = penalty_fitness(fit2);
			break;
		default:
			throw exception("Not implemented fitness type.");
		}
		if (fit1_tot < fit2_tot)
			return -1;
		if (fit1_tot > fit2_tot)
			return 1;
		return 0;
	}

	void fit(vector<vector<double>> X, vector<double> y) {
		reset();
		setup_nodes(X[0].size());
		node best_solution = node::node_constant(0);
		tuple<double, double, int> best_fitness = fitness(best_solution, X, y);
		// main loop
		while (true) {
			vector<node> all_perts = all_perturbations(best_solution);
			vector<tuple<double, node>> r2_by_perts;
			for (auto& pert : all_perts) {
				// TODO: OLS fitting here before taking fitness
				tuple<double, double, int> fit = fitness(pert, X, y);
				r2_by_perts.push_back(tuple<double, node>{get<0>(fit), pert});
				//cout << pert.to_string() << "\tR2="<<1-get<0>(fit)<< "\tRMSE="<< get<1>(fit) <<"\tsize="<< get<2>(fit) << endl;
			}
			sort(r2_by_perts.begin(), r2_by_perts.end(), TupleCompare<0>());
			for (auto r2_by_pert : r2_by_perts) {
				node pert = get<1>(r2_by_pert);
				cout << get<0>(r2_by_pert) << "\t" << pert.to_string() << endl;
				//node new_solution = local_search(pert, X, y);
				node new_solution = tune_constants(pert, X, y);
				tuple<double, double, int> new_fitness = fitness(new_solution, X, y);
				//cout << get<0>(fit) <<"\t" << pert_pert.to_string() << endl;
				if (compare_fitness(new_fitness, best_fitness) < 0) {
					best_fitness = new_fitness;
					best_solution = new_solution;
					cout << "New best:\t" << get<0>(best_fitness) << "\t" << best_solution.to_string() << endl;
				}
				/*
				vector<node> all_perts_perts = all_perturbations(pert);
				for (int i = 0; i < all_perts_perts.size(); i++) {
					node pert_pert = all_perts_perts[i];
					//cout << pert_pert.to_string() << endl;
					tuple<double, double, int> fit = fitness(pert_pert, X, y);
					//cout << get<0>(fit) <<"\t" << pert_pert.to_string() << endl;
					if (compare_fitness(fit, best_fitness) < 0) {
						best_fitness = fit;
						best_solution = pert_pert;
						cout << "New best:\t"<< get<0>(best_fitness) << "\t" << best_solution.to_string() << endl;
					}
				} */
			}
			//best_solution = get<1>(r2_by_perts[0]);
			//break;
		}
	}
};

int main()
{
	rils_rols rr(10000, 100, fitness_type::PENALTY, 0.001, 0.01, false, true, 12345);
	vector<vector<double>> X = { {6,3}, {4,4}, {3,5}, {1, 6}, {3,2} };
	vector<double> y = { 18.5, 17, 16.67, 12, 6.67 };
	rr.fit(X, y);
}
