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
#include <numeric>
#include "node.h"
#include "utils.h"
#include "eigen/Eigen/Dense"

#define PYTHON_WRAPPER 1 // comment this to run pure CPP

#ifdef PYTHON_WRAPPER

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#endif

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

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
	int max_fit_calls, max_seconds, random_state, max_feat = 200;
	double complexity_penalty, sample_size, max_complexity;
	bool classification, verbose;

	// internal stuff
	int  main_it, last_improved_it, time_elapsed, fit_calls, ls_calls, skipped_perts, total_perts;
	unordered_set<string> checked_perts;
	chrono::time_point<chrono::high_resolution_clock> start_time;
	shared_ptr < node> final_solution;
	tuple<double, double, int> final_fitness;
	double best_time;
	double total_time;
	double early_exit_eps = pow(10, -PRECISION);

	void reset() noexcept {
		main_it = 0;
		last_improved_it = 0;
		time_elapsed = 0;
		fit_calls = 0;
		ls_calls = 0;
		start_time = high_resolution_clock::now();
		checked_perts.clear();
		skipped_perts = 0;
		total_perts = 0;
		srand(random_state);
	}

	vector<node> allowed_nodes;

	void setup_nodes(vector<int> rel_feat) {
		allowed_nodes.push_back(node(node_type::PLUS));
		allowed_nodes.push_back(node(node_type::MINUS));
		allowed_nodes.push_back(node(node_type::MULTIPLY));
		allowed_nodes.push_back(node(node_type::DIVIDE));
		allowed_nodes.push_back(node(node_type::SIN));
		allowed_nodes.push_back(node(node_type::COS));
		allowed_nodes.push_back(node(node_type::LN));
		allowed_nodes.push_back(node(node_type::EXP));
		allowed_nodes.push_back(node(node_type::SQRT));
		allowed_nodes.push_back(node(node_type::SQR));
		double constants[] = { -1., 0., 0.5, 1., 2., M_PI, 10. };
		for (const auto c : constants)
			allowed_nodes.push_back(node(c));
		for (int i = 0; i < rel_feat.size(); i++)
			allowed_nodes.push_back(node(rel_feat[i]));
		if(verbose)
			cout << "Finished creating allowed nodes" << endl;
	}

	/*
	void call_and_verify_simplify(shared_ptr<node> solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		shared_ptr<node> solution_before = node::node_copy(*solution);
		double r2_before = get<0>(fitness(solution, X, y));
		solution->simplify();
		double r2_after = get<0>(fitness(solution, X, y));
		if (abs(r2_before - r2_after) > 0.0001 && abs(r2_before - r2_after) / abs(max(r2_before, r2_after)) > 0.1) {
			solution_before->simplify();
			std::cout << "Error in simplification logic -- non acceptable difference in R2 before and after simplification " << r2_before << " " << r2_after << endl;
			exit(1);
		}
	}*/

	void add_const_finetune(const node& old_node, vector<node>& candidates) {
		if (old_node.is<node_type::CONST>()) {
			//finetune constants
			if (old_node.get_const_value() == 0.0) {
				candidates.push_back(node(-1.0));
				candidates.push_back(node(1.0));
			}
			else {
				double multipliers[] = { -1., 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0., 1, 1.1, 1.2, 2., M_PI, 5., 10., 20., 50., 100. };
				for (const auto mult : multipliers)
					candidates.push_back(node(old_node.get_const_value() * mult));
				//double adders[] = { -1, -0.5, 0.5, 1 };
				//for (auto add : adders)
				//	candidates.push_back(*node::node_constant(old_node.const_value + add));
			}
		}
	}

	void add_pow_exponent_increase_decrease(const node& old_node, vector<node>& candidates) {
		if (old_node.is<node_type::POW>()) {
			if (old_node.get_right()->is_not<node_type::CONST>()) {
				std::cout << "Only constants are allowed in power exponents." << endl;
				exit(1);
			}
			auto nc_dec = node::deep_copy(old_node);
			nc_dec.get_right()->add_const(-0.5);
			if (nc_dec.get_right()->get_const_value() == 0.0) // avoid exponent 0
				nc_dec.get_right()->add_const(-0.5);
			auto nc_inc = node::deep_copy(old_node);
			nc_inc.get_right()->add_const(0.5);
			if (nc_inc.get_right()->get_const_value() == 0.0) // avoid exponent 0
				nc_inc.get_right()->add_const(0.5);
			candidates.push_back(nc_dec);
			candidates.push_back(nc_inc);
		}
	}

	void add_change_to_subtree(const node& old_node, vector<node>& candidates) {
		if (old_node.get_arity() >= 1) {
			// change node to one of its left subtrees
			get_all_subtrees<true>(*old_node.get_left(), candidates);
		}
		if (old_node.get_arity() >= 2) {
			// change node to one of its right subtrees
			get_all_subtrees<true>(*old_node.get_right(), candidates);
		}
	}

	void add_change_to_var_const(const node& old_node, vector<node>& candidates) {
		// change anything to variable or constant
		for (auto& n : allowed_nodes) {
			if (n.get_arity() != 0)
				continue;
			if (old_node.is<node_type::VAR>() && old_node.get_var_index() == n.get_var_index())
				continue; // avoid changing to same variable
			candidates.push_back(node::deep_copy(n));
		}
	}

	void add_change_to_var_or_1(const node& old_node, vector<node>& candidates) {
		// change anything to variable or constant
		for (auto& n : allowed_nodes) {
			if (n.is_not<node_type::VAR>())
				continue;
			if (old_node.is<node_type::VAR>() && old_node.get_var_index() == n.get_var_index())
				continue; // avoid changing to same variable
			candidates.push_back(node::deep_copy(n));
		}
		candidates.push_back(node(1.0));
	}

	void add_change_const_to_var(const node& old_node, vector<node>& candidates) {
		// change constant to variable
		if (old_node.is<node_type::CONST>()) {
			for (auto& n : allowed_nodes) {
				if (n.is_not<node_type::VAR>())
					continue;
				candidates.push_back(node::deep_copy(n));
			}
		}
	}

	void add_change_unary_applied(const node& old_node, vector<node>& candidates) {
		// change anything to unary operation applied to it
		for (const auto& n_un : allowed_nodes) {
			if (n_un.get_arity() != 1 || !n_un.is_allowed_left(old_node))
				continue;
			candidates.push_back(create_node(n_un.get_type(), &old_node));
		}
	}
	void add_change_variable_to_unary_applied(const node& old_node, vector<node>& candidates) {
		if (old_node.is<node_type::VAR>())
			add_change_unary_applied(old_node, candidates);
	}

	void add_change_unary_to_another(const node& old_node, vector<node>& candidates) {
		if (old_node.get_arity() == 1)
		{
			// change unary operation to another unary operation
			for (const auto& n_un : allowed_nodes) {
				if (n_un.get_arity() != 1 || n_un.get_type() == old_node.get_type())
					continue;
				if (!n_un.is_allowed_left(*old_node.get_left()))
					continue;
				candidates.push_back(create_node(n_un.get_type(), old_node.get_left()));
			}
		}
	}

	void add_change_binary_applied(const node& old_node, vector<node>& candidates) {
		// change anything to binary operation with some variable or constant  -- increases the model size
		vector<node> args;
		args.reserve(old_node.size()+ allowed_nodes.size());
		// not neead a full copy, create_node do it.
		get_all_subtrees<false>(old_node, args);

		for (auto& n_var_const : allowed_nodes) {
			if (n_var_const.is_not<node_type::VAR>() && n_var_const.is_not<node_type::CONST>())
				continue;
			args.push_back(n_var_const);
		}

		for (auto& n_bin : allowed_nodes) {
			if (n_bin.get_arity() != 2)
				continue;
			for (auto& n_arg: args)
			{
				if (n_bin.is_allowed_left(old_node))
					candidates.push_back(create_node(n_bin.get_type(), &old_node, &n_arg));
				if (!n_bin.is_symmetric() && n_bin.is_allowed_left(n_arg))
					candidates.push_back(create_node(n_bin.get_type(), &n_arg, &old_node));
			}
		}
	}

	void add_change_variable_constant_to_binary_applied(const node& old_node, vector<node>& candidates) {
		if (old_node.is<node_type::VAR>() || old_node.is<node_type::CONST>())
			add_change_binary_applied(old_node, candidates);
	}

	void add_change_binary_to_another(const node& old_node, vector<node>& candidates) {
		if (old_node.get_arity() == 2) {
			// change one binary operation to another
			for (auto& n_bin : allowed_nodes)
			{
				if (n_bin.get_arity() != 2 || n_bin.get_type() == old_node.get_type())
					continue;
				if (n_bin.is_allowed_left(*old_node.get_left()))
					candidates.push_back(create_node(n_bin.get_type(), old_node.get_left(), old_node.get_right()));
				if (!n_bin.is_symmetric() && n_bin.is_allowed_left(*old_node.get_right()))
					candidates.push_back(create_node(n_bin.get_type(), old_node.get_right(), old_node.get_left()));
			}
		}
	}

	vector<node> perturb_candidates(const node& old_node) {
		vector<node> candidates;
		candidates.reserve(1000);
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
		candidates.reserve(1000);
		add_const_finetune(old_node, candidates);
		add_change_to_subtree(old_node, candidates);
		//add_change_to_var_const(old_node, candidates);
		add_change_to_var_or_1(old_node, candidates);
		add_change_unary_applied(old_node, candidates);
		//add_change_variable_to_unary_applied(old_node, candidates);
		add_change_unary_to_another(old_node, candidates);
		add_change_binary_applied(old_node, candidates);
		//add_change_variable_constant_to_binary_applied(old_node, candidates);
		add_change_binary_to_another(old_node, candidates);
		return candidates;
	}

	vector<node> all_candidates(shared_ptr<node> passed_solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y, bool local_search) {
		shared_ptr < node> solution = node::node_copy(*passed_solution);
		vector<node> all_cand;
		all_cand.reserve(3000);
		unordered_set<string> all_cand_str;
		all_cand_str.reserve(3000);
		vector<node *> all_subtrees;
		all_subtrees.reserve(100);
		all_subtrees.push_back(solution.get());
		int i = 0;
		auto get_candidates = [this, local_search](const node& tree) {
			return local_search ? change_candidates(tree) : perturb_candidates(tree);
			};
		auto push_unique_candidate = [&all_cand, &all_cand_str](const node& cand) {
				const auto cand_str = cand.to_string();
				if (all_cand_str.find(cand_str) == all_cand_str.end())
				{
					all_cand.push_back(cand);
					all_cand_str.insert(cand_str);
				}
			};
		//cout << "Subtrees of " << passed_solution->to_string() << "\n--------------------------------" << endl;
		while (i < all_subtrees.size()) {
			//cout << all_subtrees[i]->to_string() << endl;
			auto& sub_tree = *all_subtrees[i];
			if (sub_tree.size() == solution->size())
			{
				// the whole tree is being changed
				const auto candidates = get_candidates(sub_tree);
				for (auto& cand : candidates)
				{
					push_unique_candidate(cand);
				}
			}
			if (sub_tree.get_arity() >= 1)
			{
				// the left subtree is being changed
				const auto candidates = get_candidates(*sub_tree.get_left());
				const auto old_left = *sub_tree.get_left();
				for (auto& cand : candidates) {
					sub_tree.set_left(cand);
					push_unique_candidate(node::deep_copy(*solution));
				}
				sub_tree.set_left(old_left);
				all_subtrees.push_back(sub_tree.get_left());
			}
			if (sub_tree.get_arity() >= 2)
			{
				// the right subtree is being changed
				const auto candidates = get_candidates(*sub_tree.get_right());
				const auto old_right = *sub_tree.get_right();
				for (auto& cand : candidates)
				{
					sub_tree.set_right(cand);
					push_unique_candidate(node::deep_copy(*solution));
				}
				sub_tree.set_right(old_right);
				all_subtrees.push_back(sub_tree.get_right());
			}
			i++;
		}
		for (auto& node : all_cand) {
			//cout << "Before: " << node.to_string() << endl;
			int it_max = 5;
			while (it_max > 0) {
				const auto size = node.size();
				node.expand();
				node.normalize_factor_constants(node_type::NONE, false);
				node.simplify();
				if (node.size() == size)
					break;
				it_max--;
			}
		}

		unordered_set<string> filtered_cand_strings;
		filtered_cand_strings.reserve(all_cand.size());
		vector<node> filtered_candidates;
		filtered_candidates.reserve(all_cand.size());
		for (auto& node : all_cand) {
			string node_str = node.to_string();
			if (filtered_cand_strings.find(node_str)!=filtered_cand_strings.end()) {
				//cout << node_str << " already exists." << endl;
				continue;
			}
			filtered_cand_strings.insert(node_str);
			filtered_candidates.push_back(node);
		}
		return filtered_candidates;
	}

	shared_ptr < node> tune_constants(shared_ptr<node> solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		shared_ptr < node> solution_copy = node::node_copy(*solution);
		// TODO: extract non constant factors followed by expression normalization and avoiding tuning already tuned expressions should be done earlier in the all_perturbations phase
		solution->expand();
		solution->simplify();
		vector<node*> all_factors;
		solution->extract_non_constant_factors(all_factors);
		vector<node *> factors;
		for (auto f : all_factors) {
			if (f->is<node_type::CONST>())
				continue;
			if (f->get_arity() == 2 && f->get_left()->is<node_type::CONST>() && f->get_right()->is<node_type::CONST>())
				continue; // this is also constant so ignore it
			if (f->is<node_type::MULTIPLY>() || f->is<node_type::PLUS>() || f->is<node_type::MINUS>()) { // exactly one of the terms is constant so just take another one into account because the constant will go to free term
				if (f->get_left()->is<node_type::CONST>()) {
					factors.push_back(f->get_right());
					continue;
				}
				else if (f->get_right()->is<node_type::CONST>()) {
					factors.push_back(f->get_left());
					continue;
				}
			}
			if (f->is<node_type::DIVIDE>() && f->get_right()->is<node_type::CONST>()) { // divider is constant so just ignore it
				factors.push_back(f->get_left());
				continue;
			}
			factors.push_back(f);
		}
		node free_term = node(1.0);
		factors.push_back(&free_term); // add free term

		Eigen::MatrixXd A(y.size(), factors.size());
		Eigen::VectorXd b = y;

		for (int i = 0; i < factors.size(); i++) {
			A.col(i) = factors[i]->evaluate_all(X);
		}

		auto coefs = A.colPivHouseholderQr().solve(b).eval();
		//for (auto coef : coefs)
		//	cout << coef << endl;

		shared_ptr < node> ols_solution = NULL;
		int i = 0;
		for (auto coef : coefs) {
			//cout << coefs[i] << "*"<< factors[i]->to_string()<<"+";
			if (value_zero(coef)) {
				i++;
				continue;
			}
			shared_ptr<node> new_fact{};
			if (factors[i]->is<node_type::CONST>())
				new_fact = make_shared<node>(coef * factors[i]->get_const_value());
			else {
				if (value_one(coef))
					new_fact = node::node_copy(*factors[i]);
				else {
					const node tmp(coef);
					new_fact = create_node_ptr(node_type::MULTIPLY, &tmp, factors[i]);
				}
			}
			if (!ols_solution)
				ols_solution = new_fact;
			else {
				ols_solution = create_node_ptr(node_type::PLUS, ols_solution.get(), new_fact.get());
			}
			i++;
		}
		if (!ols_solution)
			ols_solution = make_shared<node>(0.0);
		//ols_solution->simplify();
		return ols_solution;
	}

	tuple<double, double, int> fitness(shared_ptr < node> solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		fit_calls++;
		Eigen::ArrayXd yp = solution->evaluate_all(X);
		int size = solution->size();
		tuple<double, double, int> fit;

		if (classification) {
			double loss = 1 - utils::R2(y, yp);//  1 - utils::classification_accuracy(y, yp);
			double rmse =  utils::RMSE(y, yp);
			if (loss != loss || rmse!=rmse)// true only for NaN values
				return make_tuple<double, double, int>(1000, 1000, 1000);
			fit = tuple<double, double, int>{ loss, rmse,size };
		}
		else {
			const auto r2 = utils::R2(y, yp);
			const auto rmse = utils::RMSE(y, yp);
			if (r2 != r2 || rmse != rmse) // true only for NaN values
				return make_tuple<double, double, int>(1000, 1000, 1000);
			fit = tuple<double, double, int>{ 1 - r2, rmse, size };
		}
		return fit;
	}

	double fitness_value(tuple<double, double, int> fit) const noexcept {
		return (1 + get<0>(fit)) * (1 + get<1>(fit)) * (1 + get<2>(fit) * this->complexity_penalty);
	}

	int compare_fitness(tuple<double, double, int> fit1, tuple<double, double, int> fit2) const noexcept {
		// if one of the models is too large, do not accept it
		const auto size1 = get<2>(fit1);
		const auto size2 = get<2>(fit2);
		// if at least one of the complexities is to high and they are different, this is a clear criterion
		if ((size1 > max_complexity || size2 > max_complexity) && size1 != size2)
			return size1 - size2;
		const auto fit1_tot = fitness_value(fit1);
		const auto fit2_tot = fitness_value(fit2);
		if (fit1_tot < fit2_tot)
			return -1;
		if (fit1_tot > fit2_tot)
			return 1;
		return 0;
	}

	void print_state(const tuple<double, double, int>& curr_fitness) {
		std::cout << "it=" << main_it << "\tfit_calls=" << fit_calls << "\tls_calls=" << ls_calls;
		if (classification) {
			std::cout << "\tcurr_LOSS=" << get<0>(curr_fitness)  << "\tcurr_size=" << get<2>(curr_fitness);
			std::cout << "\tfinal_LOSS=" << get<0>(final_fitness) << "\tfinal_size=" << get<2>(final_fitness);
		}
		else {
			std::cout << "\tcurr_R2=" << (1 - get<0>(curr_fitness)) << "\tcurr_RMSE=" << get<1>(curr_fitness) << "\tcurr_size=" << get<2>(curr_fitness);
			std::cout << "\tfinal_R2=" << (1 - get<0>(final_fitness)) << "\tfinal_RMSE=" << get<1>(final_fitness) << "\tfinal_size=" << get<2>(final_fitness);
		}
		cout << "\tchecks_skip=" << skipped_perts << "/" << total_perts << "\tsol=";
		string sol_string = final_solution->to_string();
		if (sol_string.length() < 100)
			cout << sol_string << endl;
		else
			cout << sol_string.substr(0, 100) << "..." << endl;
	}

	bool dominates(const tuple<double, double, int>& p_fit, const tuple<double, double, int>& fit) const noexcept {
		return get<0>(p_fit) <= get<0>(fit) && get<1>(p_fit) <= get<1>(fit) && get<2>(p_fit) <= get<2>(fit);
	}

	bool is_dominated(const vector<tuple<double, double, int>>& pareto, const tuple<double, double, int>& fit) const noexcept {
		for (const auto &p_fit : pareto)
			if (dominates(p_fit, fit))
				return true;
		return false;
	}

	void add_to_pareto(vector<tuple<double, double, int>> &pareto, const tuple<double, double, int>& fit) {
		if (is_dominated(pareto, fit))
			return;
		for (int i = pareto.size() - 1; i >= 0; i--)
			if (dominates(fit, pareto[i]))
				pareto.erase(pareto.begin() + i);
		pareto.push_back(fit);
	}

	shared_ptr<node> local_search(shared_ptr<node> passed_solution, const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		vector<tuple<double, double, int>> pareto;
		ls_calls++;
		bool improved = true;
		shared_ptr<node> curr_solution = node::node_copy(*passed_solution);
		curr_solution = tune_constants(curr_solution, X, y);
		tuple<double, double, int> curr_fitness = fitness(curr_solution, X, y);
		while (improved && !finished()) {
			improved = false;
			vector<node> ls_perts = all_candidates(curr_solution, X, y, true);
			for (int j = 0; j < ls_perts.size(); j++) {
				shared_ptr<node> ls_pert = make_shared<node>(ls_perts[j]);
				if (finished())
					break;
				shared_ptr < node> ls_pert_tuned = tune_constants(ls_pert, X, y);
				tuple<double, double, int> ls_pert_tuned_fitness = fitness(ls_pert_tuned, X, y);
				if (verbose && fit_calls % 10000 == 0)
					print_state(curr_fitness);
				const auto is_dom = is_dominated(pareto, ls_pert_tuned_fitness);
				if (!is_dom && compare_fitness(ls_pert_tuned_fitness, curr_fitness) < 0) {
					improved = true;
					int it_max = 5;
					while (it_max > 0) {
						const auto size = ls_pert_tuned->size();
						ls_pert_tuned->expand();
						ls_pert_tuned->simplify();
						if (size == ls_pert_tuned->size())
							break;
						it_max--;
					}
					ls_pert_tuned_fitness = fitness(ls_pert_tuned, X, y);
					curr_solution = ls_pert_tuned;
					curr_fitness = ls_pert_tuned_fitness;
					add_to_pareto(pareto, ls_pert_tuned_fitness);
					//if(verbose)
					//	cout << "Pareto set size " << pareto.size() << endl;
					//cout << fit_calls << " New improvement in phase 2:\t" << get<0>(curr_fitness) << "\t"<<get<1>(curr_fitness)<<"\t"<<get<2>(curr_fitness) << "\t" << curr_solution->to_string() << endl;
				}
			}
		}
		return curr_solution;
	}

public:
	rils_rols(bool classification, int max_fit_calls, int max_seconds, double complexity_penalty, int max_complexity, double sample_size, bool verbose, int random_state) {
		this->classification = classification;
		this->max_fit_calls = max_fit_calls;
		this->max_seconds = max_seconds;
		this->complexity_penalty = complexity_penalty;
		this->max_complexity = max_complexity;
		this->sample_size = sample_size;
		this->verbose = verbose;
		this->random_state = random_state;
		cout << "JANO version without early exit" << endl;
		reset();
	}

	bool finished() const noexcept {
		return fit_calls >= max_fit_calls || duration_cast<seconds>(high_resolution_clock::now() - start_time).count() > max_seconds;// || (get<0>(final_fitness) < early_exit_eps && get<1>(final_fitness) < early_exit_eps);
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

#ifdef PYTHON_WRAPPER
	bool get_x(py::array_t<double> X, int data_cnt, int feat_cnt, vector<Eigen::ArrayXd>& Xe)
	{
		py::buffer_info buf_X = X.request();
		double* ptr_X = (double*)buf_X.ptr;
		if (buf_X.size != data_cnt * feat_cnt) {
			cout << "Size of X " << buf_X.size << " is not the same as the product of data count and feature count " << data_cnt * feat_cnt << endl;
			return false;
		}

		Xe.resize(feat_cnt);
		for (int i = 0; i < feat_cnt; i++)
		{
			Xe[i].resize(data_cnt);
		}

		for (int i = 0; i < data_cnt; i++)
		{
			for (int j = 0; j < feat_cnt; j++)
			{
				Xe[j][i] = ptr_X[i * feat_cnt + j];
			}
		}
		return true;
	}

	bool get_y(py::array_t<double> y, int data_cnt, Eigen::ArrayXd& ye)
	{
		py::buffer_info buf_y = y.request();
		double* ptr_y = (double*)buf_y.ptr;
		if (buf_y.size != data_cnt)
		{
			cout << "Size of y " << buf_y.size << " is not the same as the data count " << data_cnt << endl;
			return false;
		}
		ye.resize(data_cnt);
		for (int i = 0; i < data_cnt; i++)
		{
			ye[i] = ptr_y[i];
		}
		return true;
	}

	void fit(py::array_t<double> X, py::array_t<double> y, int data_cnt, int feat_cnt)
	{
		vector<Eigen::ArrayXd> Xe;
		Eigen::ArrayXd ye;

		if (!get_x(X, data_cnt, feat_cnt, Xe) || !get_y(y, data_cnt, ye))
		{
			exit(1);
		}

		fit_inner(Xe, ye);
	}

	py::array_t<double> predict(py::array_t<double> X, int data_cnt, int feat_cnt)
	{
		vector<Eigen::ArrayXd> Xe;
		if (!get_x(X, data_cnt, feat_cnt, Xe))
		{
			exit(1);
		}

		Eigen::ArrayXd res = final_solution->evaluate_all(Xe);
		py::array_t<double> res_np = py::array_t<double>(data_cnt);
		py::buffer_info buf_res_np = res_np.request();
		double* ptr_res_np = (double*)buf_res_np.ptr;
		for (int i = 0; i < data_cnt; i++)
		{
			if (classification)
				ptr_res_np[i] = res[i] >= 0.5 ? 1.0 : 0.0;
			else
				ptr_res_np[i] = res[i];
		}
		return res_np;
	}
#endif

	vector<int> relevant_features(const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& y) {
		const auto feat_cnt = X.size();
		vector<int> rel_feat;
		vector<tuple<double, int>> feat_by_r2;
		if (feat_cnt <= max_feat) {
			rel_feat.resize(feat_cnt);
			std::iota(rel_feat.begin(), rel_feat.end(), 0);
			return rel_feat;
		}
		for (int i = 0; i < feat_cnt; i++) {
			double r2 = utils::R2(X[i], y);
			feat_by_r2.push_back(tuple<double,int>{r2,i});
		}
		std::sort(feat_by_r2.begin(), feat_by_r2.end(), std::greater<>());
		for (int i = 0; i < max_feat; i++)
			rel_feat.push_back(get<1>(feat_by_r2[i]));
		return rel_feat;
	}

	void fit_inner(vector<Eigen::ArrayXd> X_all, Eigen::ArrayXd y_all) {
		reset();
		const auto sample_cnt = int(sample_size * y_all.size());
		vector<int> selected(y_all.size());

		std::iota(selected.begin(), selected.end(), 0);
		shuffle(selected.begin(), selected.end(), default_random_engine(random_state));

		// TODO: maybe done with eigen
		vector<Eigen::ArrayXd> X(X_all.size());
		for (size_t i = 0; i < X_all.size(); i++)
		{
			X[i].resize(sample_cnt);
		}
		Eigen::ArrayXd y(sample_cnt);

		for (int ix = 0; ix < sample_cnt; ix++) {
			const auto i = selected[ix];
			for (int j = 0; j < X.size(); j++)
			{
				X[j][ix] = X_all[j][i];
			}
			y[ix] = y_all[i];
		}
		// find at most max_feat relevant features and do not look the other ones
		vector<int> rel_feat = relevant_features(X, y);
		setup_nodes(rel_feat);
		final_solution = make_shared<node>(0.0);
		final_fitness = fitness(final_solution, X, y);
		// main loop
		bool improved = true;
		while (!finished()) {
			main_it += 1;
			shared_ptr < node> start_solution = final_solution;
			if (!improved) {
				// if there was no change in previous iteration, then the search is stuck in local optima so we make two consecutive random perturbations on the final_solution (best overall)
				vector<node> all_perts = all_candidates(final_solution, X, y, false);
				vector<node> all_2_perts = all_candidates(make_shared<node>(all_perts[rand() % all_perts.size()]), X, y, false);
				start_solution = node::node_copy(all_2_perts[rand() % all_2_perts.size()]);
				if(verbose)
					std::cout << "Randomized to " << start_solution->to_string() << endl;
			}
			improved = false;
			vector<node> all_perts = all_candidates(start_solution, X, y, false);
			if(verbose)
				std::cout << "Checking " << all_perts.size() << " perturbations of starting solution." << endl;
			vector<tuple<double, shared_ptr<node>>> r2_by_perts;
			for (int i = 0; i < all_perts.size(); i++) {
				if (finished())
					break;
				node pert = all_perts[i];
				string pert_str = pert.to_string();
				if (check_skip(pert_str))
					continue;
				shared_ptr < node> pert_tuned = node::node_copy(pert); // do nothing
				//shared_ptr < node> pert_tuned = tune_constants(make_shared<node>(pert), X, y);
				tuple<double, double, int> pert_tuned_fitness = fitness(pert_tuned, X, y);
				r2_by_perts.push_back(tuple<double, shared_ptr<node>>{get<0>(pert_tuned_fitness), pert_tuned});
			}
			std::sort(r2_by_perts.begin(), r2_by_perts.end(), TupleCompare<0>());
			// local search on each of these perturbations
			for (int i = 0; i < r2_by_perts.size(); i++) {
				if (finished())
					break;
				shared_ptr<node> ls_pert = get<1>(r2_by_perts[i]);
				const auto ls_pert_r2 = get<0>(r2_by_perts[i]);
				string pert_str = ls_pert->to_string();
				//cout << pert_str << endl;
				checked_perts.insert(pert_str);
				ls_pert = local_search(ls_pert, X, y);
				const auto ls_pert_fitness = fitness(ls_pert, X, y);
				const auto cmp = compare_fitness(ls_pert_fitness, final_fitness);
				if (cmp < 0) {
					improved = true;
					//call_and_verify_simplify(ls_pert, X, y);
					final_solution = node::node_copy(*ls_pert);
					final_fitness = ls_pert_fitness;
					if (verbose)
						print_state(final_fitness);
					const auto stop = high_resolution_clock::now();
					best_time = duration_cast<milliseconds>(stop - start_time).count() / 1000.0;
					//break;
				}
			}
		}
		const auto stop = high_resolution_clock::now();
		total_time = duration_cast<milliseconds>(stop - start_time).count() / 1000.0;
	}

	Eigen::ArrayXd predict_inner(const vector<Eigen::ArrayXd>& X) {
		return final_solution->evaluate_all(X);
	}


	string get_model_string() {
		return final_solution->to_string();
	}

	double get_best_time() const noexcept {
		return best_time;
	}

	double get_total_time() const noexcept {
		return total_time;
	}

	int get_fit_calls() const noexcept {
		return fit_calls;
	}
};

std::vector<double> get_row(const std::string& line)
{
	stringstream ss(line);
	vector<double> tokens;
	string tmp;
	while (getline(ss, tmp, '\t'))
		tokens.push_back(stod(tmp));

	return tokens;
}

int main()
{
	int random_state = 23654;
	int max_fit = 1000000;
	int max_time = 1200;
	double complexity_penalty = 0.001;
	int max_complexity = 200;
	double sample_size = 1;
	double train_share = 0.75;
	bool classification = false;
	string dir_path = "../paper_resources/random_12345_data";
	bool started = false;
	for (const auto& entry : fs::directory_iterator(dir_path)) {
		//if (entry.path().compare(".\\phoneme.csv")!=0) //".\\GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1.csv") != 0)
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
		const auto train_cnt = (size_t)(train_share * lines.size());
		const auto test_cnt = lines.size() - train_cnt;

		const auto firstRow = get_row(lines[0]);
		const auto X_columns = firstRow.size() - 1;

		vector<Eigen::ArrayXd> X_train(X_columns), X_test(X_columns);
		for (size_t i = 0; i < X_columns; i++)
		{
			X_train[i].resize(train_cnt);
			X_test[i].resize(test_cnt);
		}

		Eigen::ArrayXd y_train(train_cnt), y_test(test_cnt);
		bool sucess = true;
		for (int i = 0; i < lines.size(); i++) {
			string line = lines[i];

			const auto row = get_row(lines[i]);
			if (row.size() != firstRow.size())
			{
				std::cerr << "Invalid row size!" << std::endl;
				sucess = false;
				break;
			}

			if (i < train_cnt)
			{
				y_train[i] = row.back();
				for (size_t j = 0; j < X_columns; j++)
				{
					X_train[j][i] = row[j];
				}
			}
			else
			{
				y_test[i - train_cnt] = row.back();
				for (size_t j = 0; j < X_columns; j++)
				{
					X_test[j][i - train_cnt] = row[j];
				}
			}
		}
		if (!sucess)
		{
			std::cerr << "Problem with reading data" << std::endl;
			continue;
		}
		rils_rols rr(classification, max_fit, max_time, complexity_penalty, max_complexity, sample_size, true, random_state);
		rr.fit_inner(X_train, y_train);
		const auto yp_train = rr.predict_inner(X_train);
		const auto rmse_train = utils::RMSE(y_train, yp_train);
		const auto yp_test = rr.predict_inner(X_test);
		const auto rmse = utils::RMSE(y_test, yp_test);
		ofstream out_file;
		stringstream ss;
		if (classification) {
			const auto acc_train = utils::classification_accuracy(y_train, yp_train);
			const auto acc = utils::classification_accuracy(y_test, yp_test);
			ss << setprecision(PRECISION) << entry << "\tACC=" << acc << "\tACC_tr="<<acc_train;
		}
		else {
			const auto r2_train = utils::R2(y_train, yp_train);
			const auto r2 = utils::R2(y_test, yp_test);
			ss << setprecision(PRECISION) << entry << "\tR2=" << r2 << "\tR2_tr=" << r2_train;
		}
		ss<< "\tRMSE=" << rmse << "\tRMSE_tr=" << rmse_train << "\ttotal_time=" << rr.get_total_time() << "\tbest_time=" << rr.get_best_time() << "\tfit_calls=" << rr.get_fit_calls() << "\tmodel = " << rr.get_model_string() << endl;
		std::cout << ss.str();
		out_file.open("out.txt", ios_base::app);
		out_file << ss.str();
		out_file.close();
	}
}

#ifdef PYTHON_WRAPPER

PYBIND11_MODULE(rils_rols_cpp, m) {
	py::class_<rils_rols>(m, "rils_rols")
		.def(py::init<bool, int, int, double, int, double, bool, int>())
		.def("fit", &rils_rols::fit)
		.def("predict", &rils_rols::predict)
		.def("get_model_string", &rils_rols::get_model_string)
		.def("get_best_time", &rils_rols::get_best_time)
		.def("get_fit_calls", &rils_rols::get_fit_calls)
		.def("get_total_time", &rils_rols::get_total_time);
}

#endif