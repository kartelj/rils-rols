#include <iostream>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <random>
#include <chrono>
#include "node.h"
#include "utils.h"
#include "eigen/Eigen/Dense"

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

# define PRECISION 12

enum class fitness_type
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
	int ls_it, main_it, last_improved_it, time_elapsed, fit_calls;
	node* final_solution;
	tuple<double, double, int> final_fitness;
	double best_time;
	double total_time;
	double early_exit_eps = pow(10,-PRECISION);

	void reset() {
		ls_it = 0;
		main_it = 0;
		last_improved_it = 0;
		time_elapsed = 0;
		fit_calls = 0;
		srand(random_state);
	}

	vector<node> allowed_nodes;

	void setup_nodes(int var_cnt) {
		allowed_nodes.push_back(*node::node_plus());
		allowed_nodes.push_back(*node::node_minus());
		allowed_nodes.push_back(*node::node_multiply());
		allowed_nodes.push_back(*node::node_divide());
		allowed_nodes.push_back(*node::node_sin());
		allowed_nodes.push_back(*node::node_cos());
		allowed_nodes.push_back(*node::node_ln());
		allowed_nodes.push_back(*node::node_exp());
		//allowed_nodes.push_back(*node::node_sqrt());
		//allowed_nodes.push_back(*node::node_sqr());
		node* sqr = node::node_pow();
		sqr->right = node::node_constant(2);
		allowed_nodes.push_back(*sqr);
		node* sqrt = node::node_pow();
		sqrt->right = node::node_constant(0.5);
		allowed_nodes.push_back(*sqrt);
		double constants[] = { -1, 0, 0.5, 1, 2, M_PI, 10};
		for (auto c : constants)
			allowed_nodes.push_back(*node::node_constant(c));
		for (int i = 0; i < var_cnt; i++)
			allowed_nodes.push_back(*node::node_variable(i));
	}

	void call_and_verify_simplify(node& solution, const vector<Eigen::ArrayXd> &X, const Eigen::ArrayXd &y) {
		node* solution_before = node::node_copy(solution);
		double r2_before = get<0>(fitness(&solution, X, y));
		solution.simplify();
		double r2_after = get<0>(fitness(&solution, X, y));
		if (abs(r2_before - r2_after) > 0.0001 && abs(r2_before - r2_after)/abs(max(r2_before, r2_after))> 0.1) {
			solution_before->simplify();
				cout << "Error in simplification logic -- non acceptable difference in R2 before and after simplification "<< r2_before<< " "<<r2_after << endl;
				exit(1);
		}
	}

	void add_const_finetune(const node& old_node, vector<node> &candidates) {
		if (old_node.type == node_type::CONST) {
			//finetune constants
			if (old_node.const_value == 0) {
				candidates.push_back(*node::node_constant(-1));
				candidates.push_back(*node::node_constant(1));
			}
			else {
				double multipliers[] = { -1, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0, 1, 1.1, 1.2, 2, M_PI, 5, 10, 20, 50, 100 };
				for (auto mult : multipliers)
					candidates.push_back(*node::node_constant(old_node.const_value * mult));
				//double adders[] = { -1, -0.5, 0.5, 1 };
				//for (auto add : adders)
				//	candidates.push_back(*node::node_constant(old_node.const_value + add));
			}
		}
	}

	void add_pow_exponent_increase_decrease(const node& old_node, vector<node>& candidates) {
		if (old_node.type == node_type::POW) {
			if (old_node.right->type != node_type::CONST) {
				cout << "Only constants are allowed in power exponents." << endl;
				exit(1);
			}
			node* nc_dec = node::node_copy(old_node);
			nc_dec->right->const_value -= 0.5;
			if (nc_dec->right->const_value == 0) // avoid exponent 0
				nc_dec->right->const_value -= 0.5;
			node* nc_inc = node::node_copy(old_node);
			nc_inc->right->const_value += 0.5;
			if (nc_inc->right->const_value == 0) // avoid exponent 0
				nc_inc->right->const_value += 0.5;
			candidates.push_back(*nc_dec);
			candidates.push_back(*nc_inc);
		}
	}

	void add_change_to_subtree(const node& old_node, vector<node>& candidates) {
		if (old_node.arity >= 1) {
			// change node to one of its left subtrees
			vector<node*> subtrees = node::all_subtrees_references(old_node.left);
			for (auto n : subtrees) {
				node* n_c = node::node_copy(*n);
				candidates.push_back(*n_c);
			}
		}
		if (old_node.arity >= 2) {
			// change node to one of its right subtrees
			vector<node*> subtrees = node::all_subtrees_references(old_node.right);
			for (auto n : subtrees) {
				node* n_c = node::node_copy(*n);
				candidates.push_back(*n_c);
			}
		}
	}

	void add_change_to_var_const(const node& old_node, vector<node>& candidates) {
		// change anything to variable or constant
		for (auto& n : allowed_nodes) {
			if (n.arity != 0)
				continue;
			if (old_node.type == node_type::VAR && old_node.var_index == n.var_index)
				continue; // avoid changing to same variable
			node* n_c = node::node_copy(n);
			candidates.push_back(*n_c);
		}
	}

	void add_change_const_to_var(const node& old_node, vector<node>& candidates) {
		// change constant to variable
		if (old_node.type == node_type::CONST) {
			for (auto& n : allowed_nodes) {
				if (n.type != node_type::VAR)
					continue;
				node* n_c = node::node_copy(n);
				candidates.push_back(*n_c);
			}
		}
	}

	void add_change_unary_applied(const node& old_node, vector<node>& candidates) {
		// change anything to unary operation applied to it
		for (auto& n_un : allowed_nodes) {
			if (n_un.arity != 1)
				continue;
			node* n_un_c = node::node_copy(n_un);
			node* old_node_c = node::node_copy(old_node);
			n_un_c->left = old_node_c;
			candidates.push_back(*n_un_c);
		}
	}
	void add_change_variable_to_unary_applied(const node& old_node, vector<node>& candidates) {
		if (old_node.type == node_type::VAR) 
			add_change_unary_applied(old_node, candidates);
	}

	void add_change_unary_to_another(const node& old_node, vector<node>& candidates) {
		if (old_node.arity == 1) {
			// change unary operation to another unary operation
			for (auto& n_un : allowed_nodes) {
				if (n_un.arity != 1 || n_un.type == old_node.type)
					continue;
				node* n_un_c = node::node_copy(n_un);
				node* old_left_c = node::node_copy(*old_node.left);
				n_un_c->left = old_left_c;
				candidates.push_back(*n_un_c);
			}
		}
	}

	void add_change_binary_applied(const node& old_node, vector<node>& candidates) {
		// change anything to binary operation with some variable or constant  -- increases the model size
		for (auto& n_bin : allowed_nodes) {
			if (n_bin.arity != 2)
				continue;
			for (auto& n_var_const : allowed_nodes) {
				if (n_var_const.type != node_type::VAR && n_var_const.type != node_type::CONST)
					continue;
				node* n_bin_c = node::node_copy(n_bin);
				node* old_node_c = node::node_copy(old_node);
				node* n_var_c = node::node_copy(n_var_const);
				n_bin_c->left = old_node_c;
				n_bin_c->right = n_var_c;
				candidates.push_back(*n_bin_c);
				if (!n_bin.symmetric) {
					n_bin_c = node::node_copy(n_bin);
					old_node_c = node::node_copy(old_node);
					n_var_c = node::node_copy(n_var_const);
					n_bin_c->right = old_node_c;
					n_bin_c->left = n_var_c;
					candidates.push_back(*n_bin_c);
				}
			}
		}
	}

	void add_change_variable_constant_to_binary_applied(const node& old_node, vector<node>& candidates) {
		if (old_node.type == node_type::VAR || old_node.type == node_type::CONST)
			add_change_binary_applied(old_node, candidates);
	}

	void add_change_binary_to_another(const node& old_node, vector<node>& candidates) {
		if (old_node.arity == 2) {
			// change one binary operation to another
			for (auto& n_bin : allowed_nodes) {
				if (n_bin.arity != 2 || n_bin.type == old_node.type)
					continue;
				node* n_bin_c = node::node_copy(n_bin);
				node* old_left_c = node::node_copy(*old_node.left);
				node* old_right_c = node::node_copy(*old_node.right);
				n_bin_c->left = old_left_c;
				n_bin_c->right = old_right_c;
				candidates.push_back(*n_bin_c);
				if (!n_bin.symmetric) {
					node* n_bin_c = node::node_copy(n_bin);
					node* old_left_c = node::node_copy(*old_node.left);
					node* old_right_c = node::node_copy(*old_node.right);
					n_bin_c->right = old_left_c;
					n_bin_c->left = old_right_c;
					candidates.push_back(*n_bin_c);
				}
			}
		}
	}

	vector<node> perturb_candidates(const node& old_node) {
		vector<node> candidates;
		//add_pow_exponent_increase_decrease(old_node, candidates);
		add_change_to_subtree(old_node, candidates);
		add_change_const_to_var(old_node, candidates); // in Python version this was just change of const to var, but maybe it is ok to change anything to var
		add_change_variable_to_unary_applied(old_node, candidates); 
		add_change_unary_to_another(old_node, candidates);
		add_change_variable_constant_to_binary_applied(old_node, candidates);
		add_change_binary_to_another(old_node, candidates);
		cout << "Returning " << candidates.size() << " perturb candidates" << endl;
		return candidates;
	}

	vector<node> change_candidates(const node& old_node) {
		vector<node> candidates;
		add_const_finetune(old_node, candidates);
		//add_pow_exponent_increase_decrease(old_node, candidates);
		add_change_to_subtree(old_node, candidates);
		add_change_to_var_const(old_node, candidates);
		add_change_unary_applied(old_node, candidates);
		add_change_unary_to_another(old_node, candidates);
		add_change_binary_applied(old_node, candidates);
		add_change_binary_to_another(old_node, candidates);
		return candidates;
	}

	vector<node> all_candidates(const node& passed_solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, bool local_search) {
		node* solution = node::node_copy(passed_solution);
		vector<node> all_cand;
		vector<node*> all_subtrees;
		all_subtrees.push_back(solution);
		int i = 0;
		while (i < all_subtrees.size()) {
			if (all_subtrees[i]->size() == solution->size()) {
				// the whole tree is being changed
				vector<node> candidates;
				if (local_search)
					candidates = change_candidates(*all_subtrees[i]);
				else
					candidates = perturb_candidates(*all_subtrees[i]);
				for (auto& cand : candidates)
					all_cand.push_back(cand);
			}
			if (all_subtrees[i]->arity >= 1) {
				// the left subtree is being changed
				vector<node> candidates;
				if (local_search)
					candidates = change_candidates(*all_subtrees[i]->left);
				else
					candidates = perturb_candidates(*all_subtrees[i]->left);
				node* old_left = node::node_copy(*all_subtrees[i]->left);
				for (auto& cand : candidates) {
					node* cand_c = node::node_copy(cand);
					all_subtrees[i]->left = cand_c;
					node* solution_copy = node::node_copy(*solution);
					all_cand.push_back(*solution_copy);
				}
				all_subtrees[i]->left = old_left;
				all_subtrees.push_back(all_subtrees[i]->left);
			}
			if (all_subtrees[i]->arity >= 2) {
				// the right subtree is being changed
				vector<node> candidates;
				if (local_search)
					candidates = change_candidates(*all_subtrees[i]->right);
				else
					candidates = perturb_candidates(*all_subtrees[i]->right);
				node* old_right = node::node_copy(*all_subtrees[i]->right);
				for (auto& cand : candidates) {
					node* cand_c = node::node_copy(cand);
					all_subtrees[i]->right = cand_c;
					node* solution_copy = node::node_copy(*solution);
					all_cand.push_back(*solution_copy);
				}
				all_subtrees[i]->right = old_right;
				all_subtrees.push_back(all_subtrees[i]->right);
			}
			i++;
		}
		// TODO: simplify
		for (int i = 0; i < all_cand.size(); i++) {
			node node = all_cand[i];
			//cout << i<<"\t"<< node.to_string() << "\tBEFORE" << endl;
			call_and_verify_simplify(node, X, y);
			//cout <<i<<"\t"<< node.to_string() << "\tAFTER" << endl;
		}

		// TODO: normalize

		// eliminate duplicates
		unordered_set<string> filtered_cand_strings;
		vector<node> filtered_candidates;
		for (auto& node : all_cand) {
			string node_str = node.to_string();
			if (filtered_cand_strings.contains(node_str))
				continue;
			filtered_cand_strings.insert(node_str);
			filtered_candidates.push_back(node);
		}
		return filtered_candidates;
	}

	/// <summary>
	/// OLS based tunning on expanded expression
	/// </summary>
	/// <param name="solution"></param>
	/// <param name="X"></param>
	/// <param name="y"></param>
	/// <returns></returns>
	node* tune_constants(node *solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		node* solution_copy = node::node_copy(*solution);
		// TODO: extract non constant factors followed by expression normalization and avoiding tuning already tuned expressions should be done earlier in the all_perturbations phase
		solution->expand();
		solution->simplify();
		vector<node*> all_factors = solution->extract_non_constant_factors();
		vector<node*> factors;
		for (auto f : all_factors) {
			if (f->type == node_type::CONST)
				continue;
			if (f->arity==2 && f->left->type == node_type::CONST && f->right->type == node_type::CONST)
				continue; // this is also constant so ignore it
			if (f->type == node_type::MULTIPLY || f->type == node_type::PLUS || f->type == node_type::MINUS) { // exactly one of the terms is constant so just take another one into account because the constant will go to free term
				if (f->left->type == node_type::CONST) {
					factors.push_back(f->right);
					continue;
				}
				else if (f->right->type == node_type::CONST) {
					factors.push_back(f->left);
					continue;
				}
			}
			if (f->type == node_type::DIVIDE && f->right->type == node_type::CONST) { // divider is constant so just ignore it
				factors.push_back(f->left);
				continue;
			}
			factors.push_back(f);
		}
		factors.push_back(node::node_constant(1)); // add free term
		
		Eigen::MatrixXd A(X.size(), factors.size());
		Eigen::VectorXd b(X.size());

		for (int i = 0; i < X.size(); i++)
			b(i) = y[i];

		for (int i = 0; i < factors.size(); i++) {
			Eigen::ArrayXd factor_values = factors[i]->evaluate_all(X);
			for (int j = 0; j < X.size(); j++) 
				A(j, i) = factor_values[j];
		}

		auto coefs = A.colPivHouseholderQr().solve(b).eval();
		//for (auto coef : coefs)
		//	cout << coef << endl;

		node* ols_solution  = NULL;
		int i = 0;
		for (auto coef: coefs) {
			//cout << coefs[i] << "*"<< factors[i]->to_string()<<"+";
			node* new_fact = NULL;
			if (factors[i]->type == node_type::CONST)
				new_fact = node::node_constant(coef * factors[i]->const_value);
			else {
				new_fact = node::node_multiply();
				new_fact->left = node::node_constant(coef);
				new_fact->right = node::node_copy(*factors[i]);
			}
			if (ols_solution == NULL)
				ols_solution = new_fact;
			else {
				node* tmp = ols_solution;
				ols_solution = node::node_plus();
				ols_solution->left = tmp;
				ols_solution->right = new_fact;
			}
			i++;
		}
		return ols_solution;
	}

	tuple<double, double, int> fitness(node* solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		fit_calls++;
		Eigen::ArrayXd yp = solution->evaluate_all(X);
		double r2 = utils::R2(y, yp);
		double rmse = utils::RMSE(y, yp);
		int size = solution->size();
		tuple<double, double, int> fit = tuple<double, double, int>{ 1 - r2, rmse, size };
		return fit;
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

	void local_search(node& curr_solution, const vector<Eigen::ArrayXd> &X, const Eigen::ArrayXd &y) {
		bool improved = true;
		tuple<double, double, int> curr_fitness = fitness(&curr_solution, X, y);
		while (improved && !finished()) {
			improved = false;
			vector<node> ls_perts = all_candidates(curr_solution,X,y, true);
			for (int j = 0; j < ls_perts.size(); j++) {
				node ls_pert = ls_perts[j];
				if (finished())
					break;
				//cout << ls_pert.to_string() << endl;
				node* ls_pert_tuned = tune_constants(&ls_pert, X, y);
				tuple<double, double, int> ls_pert_tuned_fitness = fitness(ls_pert_tuned, X, y);
				//cout << get<0>(pert_pert_tuned_fitness) <<"\t" << pert_pert.to_string() << endl;
				if (compare_fitness(ls_pert_tuned_fitness, curr_fitness) < 0) {
					compare_fitness(ls_pert_tuned_fitness, curr_fitness);
					improved = true;
					curr_solution = *ls_pert_tuned;
					curr_fitness = ls_pert_tuned_fitness;
					//cout << "New improvement in phase 2:\t" << get<0>(curr_fitness) << "\t"<<get<1>(curr_fitness)<<"\t"<<get<2>(curr_fitness) << "\t" << curr_solution.to_string() << endl;
				}
			}
		}
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
		reset();
	}

	bool finished() {
		return fit_calls >= max_fit_calls || get<0>(final_fitness) < early_exit_eps;
	}

	void fit(vector<Eigen::ArrayXd> X_all, Eigen::ArrayXd y_all) {
		auto start = high_resolution_clock::now();
		reset();
		// take sample only assuming X_all and y_all are already shuffled
		int sample_size = int(initial_sample_size * X_all.size());
		vector<Eigen::ArrayXd> X;
		Eigen::ArrayXd y(sample_size);
		for (int i = 0; i < sample_size; i++) {
			Eigen::ArrayXd x(X_all[i].size());
			for (int j = 0; j < x.size(); j++)
				x[j] = X_all[i][j];
			X.push_back(x);
			y[i] = y_all[i];
		}
		setup_nodes(X[0].size());
		final_solution = node::node_constant(0);
		final_fitness = fitness(final_solution, X, y);
		// main loop
		bool improved = true;
		while (!finished()) {
			main_it += 1;
			node* start_solution = final_solution;
			if (!improved) {
				// if there was no change in previous iteration, then the search is stuck in local optima so we make two consecutive random perturbations on the final_solution (best overall)
				vector<node> all_perts = all_candidates(*final_solution,X, y, false);
				vector<node> all_2_perts = all_candidates(all_perts[rand() % all_perts.size()],X, y, false);
				start_solution = node::node_copy(all_2_perts[rand() % all_2_perts.size()]);
				cout << "Randomized to " << start_solution->to_string() << endl;
			}
			improved = false;
			//if(main_it%100==0)
			cout << main_it << ". " << fit_calls << "\t" << get<0>(final_fitness) << "\t" << final_solution->to_string() << endl;
			vector<node> all_perts = all_candidates(*start_solution,X, y, false);
			vector<tuple<double, node>> r2_by_perts;
			//taking the best 1-pert change
			for (int i = 0;i < all_perts.size(); i++) {
				if (finished())
					break;
				node pert = all_perts[i];
				//cout << pert.to_string() << endl;
				node* pert_tuned = tune_constants(&pert, X, y);
				tuple<double, double, int> pert_tuned_fitness = fitness(pert_tuned, X, y);
				r2_by_perts.push_back(tuple<double, node>{get<0>(pert_tuned_fitness), *pert_tuned});
				if (compare_fitness(pert_tuned_fitness, final_fitness) < 0) {
					improved = true;
					call_and_verify_simplify(*pert_tuned, X, y);
					final_solution = node::node_copy(*pert_tuned);
					final_fitness = pert_tuned_fitness;
					cout << "New best in phase 1:\t" << get<0>(final_fitness) << "\t" << final_solution->to_string() << endl;
					auto stop = high_resolution_clock::now();
					best_time = duration_cast<seconds>(stop - start).count();
				}
			}
			if (improved)
				continue;
			//continue;
			sort(r2_by_perts.begin(), r2_by_perts.end(), TupleCompare<0>());
			// local search on each of these perturbations
			for (int i = 0;i < r2_by_perts.size(); i++) {
				if (finished())
					break;
				node ls_pert = get<1>(r2_by_perts[i]);
				double ls_pert_r2 = get<0>(r2_by_perts[i]);
				local_search(ls_pert, X, y);
				tuple<double, double, int> ls_pert_fitness = fitness(&ls_pert, X, y);
				//cout << "LS:\t" << i << "/" << r2_by_perts.size()<<".\t"<< get<0>(ls_pert_fitness) << "\t" << ls_pert.to_string() << endl;
				if(compare_fitness(ls_pert_fitness, final_fitness)<0) {
					improved = true;
					call_and_verify_simplify(ls_pert, X, y);
					final_solution = node::node_copy(ls_pert);
					final_fitness = ls_pert_fitness;
					cout << "New best in phase 2:\t" << get<0>(final_fitness) << "\t" << final_solution->to_string() << endl;
					auto stop = high_resolution_clock::now();
					best_time = duration_cast<milliseconds>(stop - start).count() / 1000.0;
					break;
				}
			}
		}
		auto stop = high_resolution_clock::now();
		total_time = duration_cast<milliseconds>(stop - start).count()/1000.0;
	}

	Eigen::ArrayXd predict(const vector<Eigen::ArrayXd> &X) {
		return final_solution->evaluate_all(X);
	}

	string get_model_string() {
		return final_solution->to_string();
	}

	int get_best_time() {
		return best_time;
	}

	double get_total_time() {
		return total_time;
	}

	double get_fit_calls() {
		return fit_calls;
	}
};

int main()
{
	int random_state = 23654;
	int max_fit = 300000;
	int max_time = 300;
	double complexity_penalty = 0.001;
	double sample_size = 0.01;
	double train_share = 0.75;
	string dir_path = "../paper_resources/random_12345_data";
	for (const auto& entry :  fs::directory_iterator(dir_path)) {
		//if (entry.path().compare("../paper_resources/random_12345_data\\random_10_01_0010000_00.data") != 0)
		//	continue;
		std::cout << entry.path() << std::endl;
		ifstream infile(entry.path());
		string line;
		vector<string> lines;
		while (getline(infile, line))
			lines.push_back(line);
		// shuffling for later split between training and test set
		shuffle(lines.begin(), lines.end(), default_random_engine(random_state));
		int train_cnt = int(train_share * lines.size());
		vector<Eigen::ArrayXd> X_train, X_test;
		Eigen::ArrayXd y_train(train_cnt), y_test(lines.size() - train_cnt);
		for (int i = 0; i < lines.size(); i++) {
			string line = lines[i];
			stringstream ss(line);
			vector<string> tokens;
			string tmp;
			while (getline(ss, tmp, '\t'))
				tokens.push_back(tmp);
			Eigen::ArrayXd x(tokens.size());
			for (int i = 0; i < tokens.size() - 1; i++) {
				string str(tokens[i]);
				x[i] = stod(str);
			}
			string str(tokens[tokens.size() - 1]);
			if (i < train_cnt) {
				y_train[X_train.size()] = stod(str);
				X_train.push_back(x);
			}
			else {
				y_test[X_test.size()] = stod(str);
				X_test.push_back(x);
			}
		}
		rils_rols rr(max_fit, max_time, fitness_type::PENALTY, complexity_penalty, sample_size, false, true, random_state);
		rr.fit(X_train, y_train);
		Eigen::ArrayXd yp_train = rr.predict(X_train);
		double r2_train = utils::R2(y_train, yp_train);
		double rmse_train = utils::RMSE(y_train, yp_train);
		Eigen::ArrayXd yp_test = rr.predict(X_test);
		double r2 = utils::R2(y_test, yp_test);
		double rmse = utils::RMSE(y_test, yp_test);
		ofstream out_file;
		stringstream ss;
		ss << setprecision(PRECISION) <<  entry << "\tR2=" << r2 << "\tRMSE=" << rmse << "\tR2_tr=" << r2_train << "\tRMSE_tr=" << rmse_train << "\ttotal_time="<<rr.get_total_time() << "\tbest_time="<<rr.get_best_time() << "\tfit_calls="<< rr.get_fit_calls() << "\tmodel = " << rr.get_model_string() << endl;
		cout << ss.str();
		out_file.open("out.txt", ios_base::app);
		out_file << ss.str();
		out_file.close();
	}
}
