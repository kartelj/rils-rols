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
//#include <boost/python.hpp>

using namespace std;
using namespace std::chrono;
//using namespace boost::python;
namespace fs = std::filesystem;

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
	double complexity_penalty, initial_sample_size, max_complexity;
	bool random_perturbations_order, verbose;
	fitness_type fitness_type;

	// internal stuff
	int ls_it, main_it, last_improved_it, time_elapsed, fit_calls, skipped_perts, total_perts;
	unordered_set<string> checked_perts;
	chrono::time_point<chrono::high_resolution_clock> start_time;
	shared_ptr < node> final_solution;
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
		start_time = high_resolution_clock::now();
		checked_perts.clear();
		skipped_perts = 0;
		total_perts = 0;
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
		allowed_nodes.push_back(*node::node_sqrt());
		allowed_nodes.push_back(*node::node_sqr());
		/*shared_ptr<node> sqr = node::node_pow();
		sqr->right = node::node_constant(2);
		allowed_nodes.push_back(*sqr);
		shared_ptr < node> sqrt = node::node_pow();
		sqrt->right = node::node_constant(0.5);
		allowed_nodes.push_back(*sqrt);*/
		double constants[] = { -1, 0, 0.5, 1, 2, M_PI, 10};
		for (auto c : constants)
			allowed_nodes.push_back(*node::node_constant(c));
		for (int i = 0; i < var_cnt; i++)
			allowed_nodes.push_back(*node::node_variable(i));
		cout << "Finished creating allowed nodes"<<endl;
	}

	void call_and_verify_simplify(shared_ptr<node> solution, const vector<Eigen::ArrayXd> &X, const Eigen::ArrayXd &y) {
		shared_ptr<node> solution_before = node::node_copy(*solution);
		double r2_before = get<0>(fitness(solution, X, y));
		solution->simplify();
		double r2_after = get<0>(fitness(solution, X, y));
		if (abs(r2_before - r2_after) > 0.0001 && abs(r2_before - r2_after)/abs(max(r2_before, r2_after))> 0.1) {
			solution_before->simplify();
				std::cout << "Error in simplification logic -- non acceptable difference in R2 before and after simplification "<< r2_before<< " "<<r2_after << endl;
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
				std::cout << "Only constants are allowed in power exponents." << endl;
				exit(1);
			}
			shared_ptr <node> nc_dec = node::node_copy(old_node);
			nc_dec->right->const_value -= 0.5;
			if (nc_dec->right->const_value == 0) // avoid exponent 0
				nc_dec->right->const_value -= 0.5;
			shared_ptr <node> nc_inc = node::node_copy(old_node);
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
			vector< shared_ptr<node>> subtrees = node::all_subtrees_references(old_node.left);
			for (auto n : subtrees) {
				shared_ptr<node> n_c = node::node_copy(*n);
				candidates.push_back(*n_c);
			}
		}
		if (old_node.arity >= 2) {
			// change node to one of its right subtrees
			vector< shared_ptr<node>> subtrees = node::all_subtrees_references(old_node.right);
			for (auto n : subtrees) {
				shared_ptr < node> n_c = node::node_copy(*n);
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
			shared_ptr < node> n_c = node::node_copy(n);
			candidates.push_back(*n_c);
		}
	}

	void add_change_to_var_or_1(const node& old_node, vector<node>& candidates) {
		// change anything to variable or constant
		for (auto& n : allowed_nodes) {
			if (n.type!=node_type::VAR)
				continue;
			if (old_node.type == node_type::VAR && old_node.var_index == n.var_index)
				continue; // avoid changing to same variable
			shared_ptr < node> n_c = node::node_copy(n);
			candidates.push_back(*n_c);
		}
		candidates.push_back(*node::node_constant(1));
	}


	void add_change_const_to_var(const node& old_node, vector<node>& candidates) {
		// change constant to variable
		if (old_node.type == node_type::CONST) {
			for (auto& n : allowed_nodes) {
				if (n.type != node_type::VAR)
					continue;
				shared_ptr < node> n_c = node::node_copy(n);
				candidates.push_back(*n_c);
			}
		}
	}

	void add_change_unary_applied(const node& old_node, vector<node>& candidates) {
		// change anything to unary operation applied to it
		for (auto& n_un : allowed_nodes) {
			if (n_un.arity != 1)
				continue;
			if (!n_un.is_allowed_left(old_node))
				continue;
			shared_ptr < node> n_un_c = node::node_copy(n_un);
			shared_ptr < node> old_node_c = node::node_copy(old_node);
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
				if (!n_un.is_allowed_left(*old_node.left))
					continue;
				shared_ptr < node> n_un_c = node::node_copy(n_un);
				shared_ptr < node> old_left_c = node::node_copy(*old_node.left);
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
				if (n_bin.is_allowed_left(old_node)) {
					shared_ptr < node> n_bin_c = node::node_copy(n_bin);
					shared_ptr < node> old_node_c = node::node_copy(old_node);
					shared_ptr < node> n_var_c = node::node_copy(n_var_const);
					n_bin_c->left = old_node_c;
					n_bin_c->right = n_var_c;
					candidates.push_back(*n_bin_c);
				}
				if (!n_bin.symmetric && n_bin.is_allowed_left(n_var_const)) {
					shared_ptr < node> n_bin_c = node::node_copy(n_bin);
					shared_ptr < node> old_node_c = node::node_copy(old_node);
					shared_ptr < node> n_var_c = node::node_copy(n_var_const);
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
				if (n_bin.is_allowed_left(*old_node.left)) {
					shared_ptr < node> n_bin_c = node::node_copy(n_bin);
					shared_ptr < node> old_left_c = node::node_copy(*old_node.left);
					shared_ptr < node> old_right_c = node::node_copy(*old_node.right);
					n_bin_c->left = old_left_c;
					n_bin_c->right = old_right_c;
					candidates.push_back(*n_bin_c);
				}
				if (!n_bin.symmetric && n_bin.is_allowed_left(*old_node.right)) {
					shared_ptr < node> n_bin_c = node::node_copy(n_bin);
					shared_ptr < node> old_left_c = node::node_copy(*old_node.left);
					shared_ptr < node> old_right_c = node::node_copy(*old_node.right);
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
		//add_change_const_to_var(old_node, candidates); // in Python version this was just change of const to var, but maybe it is ok to change anything to var
		add_change_to_var_or_1(old_node, candidates);
		add_change_variable_to_unary_applied(old_node, candidates); 
		add_change_unary_to_another(old_node, candidates);
		add_change_variable_constant_to_binary_applied(old_node, candidates);
		add_change_binary_to_another(old_node, candidates);
		return candidates;
	}

	vector<node> change_candidates(const node& old_node) {
		vector<node> candidates;
		add_const_finetune(old_node, candidates);
		//cout << "---------------------" << endl;
		//cout<< candidates.size() << " After finetune " << endl;
		//add_pow_exponent_increase_decrease(old_node, candidates);
		add_change_to_subtree(old_node, candidates);
		//cout << candidates.size() << " After subtree " << endl;
		//add_change_to_var_const(old_node, candidates);
		add_change_to_var_or_1(old_node, candidates);
		//cout << candidates.size() << " After to var const " << endl;
		//add_change_unary_applied(old_node, candidates);
		add_change_variable_to_unary_applied(old_node, candidates);
		//cout << candidates.size() << " After unary applied " << endl;
		add_change_unary_to_another(old_node, candidates);
		//cout << candidates.size() << " After unary another " << endl;
		//add_change_binary_applied(old_node, candidates);
		add_change_variable_constant_to_binary_applied(old_node, candidates);
		//cout << candidates.size() << " After binary applied " << endl;
		add_change_binary_to_another(old_node, candidates);
		//cout << candidates.size() << " After binary another " << endl;
		return candidates;
	}

	vector<node> all_candidates(shared_ptr<node> passed_solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, bool local_search) {
		shared_ptr < node> solution = node::node_copy(*passed_solution);
		vector<node> all_cand;
		unordered_set<string> all_cand_str;
		vector<shared_ptr<node>> all_subtrees;
		all_subtrees.push_back(solution);
		int i = 0;
		//cout << "Subtrees of " << passed_solution->to_string() << "\n--------------------------------" << endl;
		while (i < all_subtrees.size()) {
			//cout << all_subtrees[i]->to_string() << endl;
			if (all_subtrees[i]->size() == solution->size()) {
				// the whole tree is being changed
				vector<node> candidates;
				if (local_search)
					candidates = change_candidates(*all_subtrees[i]);
				else
					candidates = perturb_candidates(*all_subtrees[i]);
				for (auto& cand : candidates) {
					string cand_str = cand.to_string();
					if (all_cand_str.find(cand_str) != all_cand_str.end())
						continue;
					all_cand.push_back(cand);
					all_cand_str.insert(cand_str);
				}
			}
			if (all_subtrees[i]->arity >= 1) {
				// the left subtree is being changed
				vector<node> candidates;
				if (local_search)
					candidates = change_candidates(*all_subtrees[i]->left);
				else
					candidates = perturb_candidates(*all_subtrees[i]->left);
				shared_ptr < node> old_left = node::node_copy(*all_subtrees[i]->left);
				for (auto& cand : candidates) {
					shared_ptr < node> cand_c = node::node_copy(cand);
					all_subtrees[i]->left = cand_c;
					shared_ptr < node> solution_copy = node::node_copy(*solution);
					string cand_str = solution_copy->to_string();
					if (all_cand_str.find(cand_str) != all_cand_str.end())
						continue;
					all_cand.push_back(*solution_copy);
					all_cand_str.insert(cand_str);
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
				shared_ptr < node> old_right = node::node_copy(*all_subtrees[i]->right);
				for (auto& cand : candidates) {
					shared_ptr < node> cand_c = node::node_copy(cand);
					all_subtrees[i]->right = cand_c;
					shared_ptr < node> solution_copy = node::node_copy(*solution);
					string cand_str = solution_copy->to_string();
					if (all_cand_str.find(cand_str) != all_cand_str.end())
						continue;
					all_cand.push_back(*solution_copy);
					all_cand_str.insert(cand_str);
				}
				all_subtrees[i]->right = old_right;
				all_subtrees.push_back(all_subtrees[i]->right);
			}
			i++;
		}
		for (auto &node: all_cand) {
			//cout << "Before: " << node.to_string() << endl;
			int it_max = 5;
			while(it_max>0) {
				int size = node.size();
				node.expand();
				node.normalize_factor_constants(node_type::NONE, false);
				node.simplify();
				if (node.size() == size)
					break;
				it_max--;
			}
			//cout << "After: " << node.to_string() << endl;
			//if (node.to_string().compare(before) == 0)
			//	break; // no further change
			//}
			//cout << node.to_string() << "\tAFTER" << endl;
			//cout << endl;
		}
		
		// eliminate duplicates -- TODO: to_string() is slow, fix this
		unordered_set<string> filtered_cand_strings;
		vector<node> filtered_candidates;
		for (auto& node : all_cand) {
			string node_str = node.to_string();
			if (filtered_cand_strings.contains(node_str)) {
				//cout << node_str << " already exists." << endl;
				continue;
			}
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
	shared_ptr < node> tune_constants(shared_ptr<node> solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		shared_ptr < node> solution_copy = node::node_copy(*solution);
		// TODO: extract non constant factors followed by expression normalization and avoiding tuning already tuned expressions should be done earlier in the all_perturbations phase
		solution->expand();
		solution->simplify();
		vector<shared_ptr<node>> all_factors = solution->extract_non_constant_factors();
		vector<shared_ptr<node>> factors;
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

		shared_ptr < node> ols_solution  = NULL;
		int i = 0;
		for (auto coef: coefs) {
			//cout << coefs[i] << "*"<< factors[i]->to_string()<<"+";
			if (value_zero(coef)) {
				i++;
				continue;
			}
			shared_ptr < node> new_fact = NULL;
			if (factors[i]->type == node_type::CONST)
				new_fact = node::node_constant(coef * factors[i]->const_value);
			else {
				if (value_one(coef))
					new_fact = node::node_copy(*factors[i]);
				else {
					new_fact = node::node_multiply();
					new_fact->left = node::node_constant(coef);
					new_fact->right = node::node_copy(*factors[i]);
				}
			}
			if (ols_solution == NULL)
				ols_solution = new_fact;
			else {
				shared_ptr < node> tmp = ols_solution;
				ols_solution = node::node_plus();
				ols_solution->left = tmp;
				ols_solution->right = new_fact;
			}
			i++;
		}
		if (ols_solution == NULL)
			ols_solution = node::node_constant(0);
		//ols_solution->simplify();
		return ols_solution;
	}

	tuple<double, double, int> fitness(shared_ptr < node> solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		fit_calls++;
		Eigen::ArrayXd yp = solution->evaluate_all(X);
		double r2 = utils::R2(y, yp);
		double rmse = utils::RMSE(y, yp);
		int size = solution->size();
		if (r2 != r2 || rmse != rmse) // true only for NaN values
			return make_tuple<double, double, int>(1000, 1000, 1000);
		tuple<double, double, int> fit = tuple<double, double, int>{ 1 - r2, rmse, size };
		return fit;
	}

	double penalty_fitness(tuple<double, double, int> fit) {
		return (1 + get<0>(fit)) * (1 + get<1>(fit)) * (1 + get<2>(fit) * this->complexity_penalty);
	}

	int compare_fitness(tuple<double, double, int> fit1, tuple<double, double, int> fit2) {
		// if one of the models is too large, do not accept it
		int size1 = get<2>(fit1);
		int size2 = get<2>(fit2);
		// if at least one of the complexities is to high and they are different, this is a clear criterion
		if ((size1 > max_complexity || size2 > max_complexity) && size1 != size2)
			return size1 - size2; 
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

	shared_ptr<node> local_search(shared_ptr<node> passed_solution, const vector<Eigen::ArrayXd> &X, const Eigen::ArrayXd &y) {
		bool improved = true;
		shared_ptr<node> curr_solution = node::node_copy(*passed_solution);
		curr_solution = tune_constants(curr_solution, X, y);
		tuple<double, double, int> curr_fitness = fitness(curr_solution, X, y);
		while (improved && !finished()) {
			improved = false;
			//cout << "Doing LS on " << curr_solution->to_string() <<endl<< "----------------------------"<<endl;
			vector<node> ls_perts = all_candidates(curr_solution,X,y, true);
			for (int j = 0; j < ls_perts.size(); j++) {
				shared_ptr<node> ls_pert = make_shared<node>(ls_perts[j]);
				if (finished())
					break;
				//string pert_str = ls_pert->to_string();
				//if (check_skip(pert_str)) {
				//	cout << "SKIPPED" << endl;
				//	continue;
				//}
				//cout << j<<".\t"<< pert_str << endl;
				shared_ptr < node> ls_pert_tuned = tune_constants(ls_pert, X, y);
				tuple<double, double, int> ls_pert_tuned_fitness = fitness(ls_pert_tuned, X, y);
				//cout << get<0>(pert_pert_tuned_fitness) <<"\t" << pert_pert.to_string() << endl;
				if (compare_fitness(ls_pert_tuned_fitness, curr_fitness) < 0) {
					improved = true;
					int it_max = 5;
					while (it_max>0) {
						int size = ls_pert_tuned->size();
						ls_pert_tuned->expand();
						ls_pert_tuned->simplify();
						if (size == ls_pert_tuned->size())
							break;
						it_max--;
					}
					curr_solution = ls_pert_tuned;
					curr_fitness = ls_pert_tuned_fitness;
					//cout << fit_calls << " New improvement in phase 2:\t" << get<0>(curr_fitness) << "\t"<<get<1>(curr_fitness)<<"\t"<<get<2>(curr_fitness) << "\t" << curr_solution->to_string() << endl;
				}
			}
		}
		return curr_solution;
	}

public:
	rils_rols(int max_fit_calls, int max_seconds, enum fitness_type fitness_type, double complexity_penalty, int max_complexity, double initial_sample_size, bool random_perturbations_order, bool verbose, int random_state) {
		this->max_fit_calls = max_fit_calls;
		this->max_seconds = max_seconds;
		this->fitness_type = fitness_type;
		this->complexity_penalty = complexity_penalty;
		this->max_complexity = max_complexity;
		this->initial_sample_size = initial_sample_size;
		this->random_perturbations_order = random_perturbations_order;
		this->verbose = verbose;
		this->random_state = random_state;
		reset();
	}

	bool finished() {
		return fit_calls >= max_fit_calls || (get<0>(final_fitness) < early_exit_eps && get<1>(final_fitness)<early_exit_eps) || duration_cast<seconds>(high_resolution_clock::now() - start_time).count()>max_seconds;
	}

	bool check_skip(const string& pert_str) {
		// checking if pert was already checked
		total_perts++;
		if (checked_perts.find(pert_str) != checked_perts.end()) {
			skipped_perts++;
			return true;
		}
		checked_perts.insert(pert_str);
		return false;
	}

	void fit(vector<Eigen::ArrayXd> X_all, Eigen::ArrayXd y_all) {
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

			shared_ptr < node> start_solution = final_solution;
			if (!improved) {
				// if there was no change in previous iteration, then the search is stuck in local optima so we make two consecutive random perturbations on the final_solution (best overall)
				vector<node> all_perts = all_candidates(final_solution,X, y, false);
				vector<node> all_2_perts = all_candidates(make_shared<node>(all_perts[rand() % all_perts.size()]),X, y, false);
				start_solution = node::node_copy(all_2_perts[rand() % all_2_perts.size()]);
				std::cout << "Randomized to " << start_solution->to_string() << endl;
			}
			improved = false;
			//if(main_it%100==0)
			std::cout << main_it << ". " << fit_calls << "\t" << get<0>(final_fitness) << "\t" << final_solution->to_string() << "\tchecks skipped: "<<skipped_perts<<"/"<<total_perts << endl;
			vector<node> all_perts = all_candidates(start_solution,X, y, false);
			std::cout << "Checking " << all_perts.size() << " perturbations of starting solution." << endl;
			vector<tuple<double, shared_ptr<node>>> r2_by_perts;
			for (int i = 0;i < all_perts.size(); i++) {
				if (finished())
					break;
				node pert = all_perts[i];
				string pert_str = pert.to_string();
				if (check_skip(pert_str))
					continue;
				//cout << pert_str << endl;
				//pert = local_search(pert, X, y);
				shared_ptr < node> pert_tuned = node::node_copy(pert); // do nothing
				//shared_ptr < node> pert_tuned = tune_constants(make_shared<node>(pert), X, y);
				tuple<double, double, int> pert_tuned_fitness = fitness(pert_tuned, X, y);
				r2_by_perts.push_back(tuple<double, shared_ptr<node>>{get<0>(pert_tuned_fitness), pert_tuned});
				/*
				if (compare_fitness(pert_tuned_fitness, final_fitness) < 0) {
					improved = true;
					call_and_verify_simplify(*pert_tuned, X, y);
					final_solution = node::node_copy(*pert_tuned);
					final_fitness = pert_tuned_fitness;
					std::cout << "New best in phase 1:\t" << get<0>(final_fitness) << "\t" << final_solution->to_string() << endl;
					auto stop = high_resolution_clock::now();
					best_time = duration_cast<seconds>(stop - start).count();
				}*/
			}
			//if (improved)
			//	continue;
			//continue;
			std::cout << "Preserved " << all_perts.size() << " perturbations after removing already done." << endl;
			sort(r2_by_perts.begin(), r2_by_perts.end(), TupleCompare<0>());
			// local search on each of these perturbations
			for (int i = 0;i < r2_by_perts.size(); i++) {
				if (finished())
					break;
				shared_ptr<node> ls_pert = get<1>(r2_by_perts[i]);
				double ls_pert_r2 = get<0>(r2_by_perts[i]);
				string pert_str = ls_pert->to_string();
				//cout << pert_str << endl;
				checked_perts.insert(pert_str);
				ls_pert = local_search(ls_pert, X, y);
				tuple<double, double, int> ls_pert_fitness = fitness(ls_pert, X, y);
				//cout << "LS:\t" << i << "/" << r2_by_perts.size()<<".\t"<< get<0>(ls_pert_fitness) << "\t" << ls_pert->to_string() << endl;
				int cmp = compare_fitness(ls_pert_fitness, final_fitness);
				if(cmp<0) {
					improved = true;
					call_and_verify_simplify(ls_pert, X, y);
					final_solution = node::node_copy(*ls_pert);
					final_fitness = ls_pert_fitness;
					std::cout << "New best in phase 2:\t" << get<0>(final_fitness) << "\t" << final_solution->to_string() << endl;
					auto stop = high_resolution_clock::now();
					best_time = duration_cast<milliseconds>(stop - start_time).count() / 1000.0;
					//break;
				}
			}
		}
		auto stop = high_resolution_clock::now();
		total_time = duration_cast<milliseconds>(stop - start_time).count()/1000.0;
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
	int max_fit = 10000000;
	int max_time = 300;
	double complexity_penalty = 0.001;
	int max_complexity = 200;
	double sample_size = 0.01;
	double train_share = 0.75;
	string dir_path = "../paper_resources/random_12345_data";
	bool started = false;
	for (const auto& entry :  fs::directory_iterator(dir_path)) {
		//if (entry.path().compare("../paper_resources/random_12345_data\\random_15_04_0010000_03.data") != 0)
		//	continue;
		//if (started || entry.path().compare("../paper_resources/random_12345_data\\random_06_01_0010000_00.data") == 0)
		//	started = true;
		//else
		//	continue;
		std::cout << entry.path() << std::endl;
		ifstream infile(entry.path());
		string line;
		vector<string> lines;
		while (getline(infile, line))
			lines.push_back(line);
		srand(random_state);
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
		rils_rols rr(max_fit, max_time, fitness_type::PENALTY, complexity_penalty, max_complexity, sample_size, false, true, random_state);
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
		std::cout << ss.str();
		out_file.open("out.txt", ios_base::app);
		out_file << ss.str();
		out_file.close();
	}
}

// python integration stuff
/*
BOOST_PYTHON_MODULE(rils_rols_ext) {
	class_<rils_rols>("rils_rols")
		.def(init<int, int, int, fitness_type, double, int, double, bool, bool, int>())
		.def("fit", &rils_rols::fit)
		;
}
*/
