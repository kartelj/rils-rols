#pragma once
#include <iostream>
#include <vector>
#include <ostream>
#include <string>
#include <memory>
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

constexpr int get_arity(node_type type) noexcept
{
	switch (type)
	{
	case node_type::CONST:
	case node_type::VAR:
		return 0;
	case node_type::SIN:
	case node_type::COS:
	case node_type::LN:
	case node_type::EXP:
	case node_type::SQRT:
	case node_type::SQR:
		return 1;
	}
	return 2;
}

constexpr bool get_symmetric(node_type type) noexcept
{
	switch (type)
	{
	case node_type::MINUS:
	case node_type::DIVIDE:
	case node_type::POW: // pow is set to be unary because the unary changes will take care of base, and the specific method is needed to take care for exponents -- add_pow_finetune
		return false;
	}
	return true;
}

class node : public std::enable_shared_from_this<node>
{
	shared_ptr<node> left{ nullptr };
	shared_ptr<node> right{ nullptr };
	int arity{ 0 };
	bool symmetric{ false };
	node_type type{ node_type::NONE };
	int var_index{ -1 };
	double const_value{ 0.0 };

public:
	node() noexcept = default;

	explicit node(node_type type) noexcept
		: left(nullptr)
		, right(nullptr)
		, arity(::get_arity(type))
		, symmetric(::get_symmetric(type))
		, type(type)
		, var_index(-1)
		, const_value(0.0)
	{
	}

	// !shallow copy left and right, to deep copy use create_node/create_node_ptr
	explicit node(node_type type, shared_ptr<node> left, shared_ptr<node> right) noexcept
		: left(left)
		, right(right)
		, arity(::get_arity(type))
		, symmetric(::get_symmetric(type))
		, type(type)
		, var_index(-1)
		, const_value(0.0)
	{
	}

	// constant
	explicit node(double const_value) noexcept
		: left(nullptr)
		, right(nullptr)
		, arity(::get_arity(node_type::CONST))
		, symmetric(::get_symmetric(node_type::CONST))
		, type(node_type::CONST)
		, var_index(-1)
		, const_value(const_value)
	{
	}

	// variable
	explicit node(int var_index) noexcept
		: left(nullptr)
		, right(nullptr)
		, arity(::get_arity(node_type::VAR))
		, symmetric(::get_symmetric(node_type::VAR))
		, type(node_type::VAR)
		, var_index(var_index)
		, const_value(0.0)
	{
	}

	void update_with(shared_ptr<node> src) noexcept
	{
		left = src->left;
		right = src->right;
		arity = src->arity;
		const_value = src->const_value;
		var_index = src->var_index;
		symmetric = src->symmetric;
		type = src->type;
	}

	static shared_ptr<node> node_copy(const node &n) {
		shared_ptr<node> nc = make_shared<node>(n);
		if(n.left)
			nc->left = node_copy(*n.left);
		if(n.right)
			nc->right = node_copy(*n.right);
		return nc;
	}

	static node deep_copy(const node& n) {
		node nc = n;
		if (n.left)
			nc.left = node_copy(*n.left);
		if (n.right)
			nc.right = node_copy(*n.right);
		return nc;
	}

	inline auto get_left() const noexcept
	{
		return left.get();
	}

	inline auto get_right() const noexcept
	{
		return right.get();
	}

	// !shallow copy
	inline void set_left(shared_ptr<node> new_left) noexcept
	{
		left = new_left;
	}

	// !shallow copy
	inline void set_left(const node& new_left)
	{
		left = make_shared<node>(new_left);
	}

	// !shallow copy
	inline void set_right(shared_ptr<node> new_right) noexcept
	{
		right = new_right;
	}

	// !shallow copy
	inline void set_right(const node& new_right)
	{
		right = make_shared<node>(new_right);
	}

	inline auto get_type() const noexcept
	{
		return type;
	}

	inline void set_type(node_type t) noexcept
	{
		type = t;
		arity = ::get_arity(t);
		symmetric = ::get_symmetric(t);
	}

	inline auto get_arity() const noexcept
	{
		return arity;
	}

	inline auto get_var_index() const noexcept
	{
		assert(is<node_type::VAR>());
		return var_index;
	}

	inline auto get_const_value() const noexcept
	{
		assert(is<node_type::CONST>());
		return const_value;
	}

	inline void set_const_value(double val) noexcept
	{
		set_type(node_type::CONST);
		const_value = val;
		left = nullptr;
		right = nullptr;
	}

	inline void add_const(double delta) noexcept
	{
		assert(is<node_type::CONST>());
		const_value += delta;
	}

	inline auto is_symmetric() const noexcept
	{
		return symmetric;
	}

	template<node_type T>
	inline bool is() const noexcept
	{
		return type == T;
	}

	template<node_type T>
	inline bool is_not() const noexcept
	{
		return type != T;
	}

	Eigen::ArrayXd evaluate_inner(const vector<Eigen::ArrayXd>& X, const Eigen::ArrayXd& a, const Eigen::ArrayXd& b) noexcept(false);

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
			return  "("+left->to_string() + "*" + right->to_string()+")";
		case node_type::DIVIDE:
			return  "("+left->to_string() + "/" + right->to_string()+")";
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
			return "("+left->to_string() + "*" + left->to_string()+")";
		case node_type::POW:
			return "pow("+left->to_string() + "," + right->to_string()+")";
		default:
			return "*****UNKNOWN*****";
		}
	};

	bool is_allowed_left(const node& node) const noexcept;

	Eigen::ArrayXd evaluate_all(const vector<Eigen::ArrayXd>& X);

	vector< shared_ptr<node>> extract_constants_references();

	void extract_non_constant_factors(vector<node*>& all_factors);

	int size() const noexcept;

	void simplify();

	void normalize_constants(node_type parent_type);

	void normalize_factor_constants(node_type parent_type, bool inside_factor);

	void expand();
};

inline bool value_zero(double val) noexcept {
	return abs(val) < EPS;
}

inline bool value_one(double val) noexcept {
	return abs(val - 1) < EPS;
}

inline auto create_node_ptr(node_type type, const node* left, const node* right = nullptr)
{
	return make_shared<node>(type, node::node_copy(*left), right ? node::node_copy(*right) : nullptr);
}

inline auto create_node(node_type type, const node* left, const node* right = nullptr)
{
	return node(type, node::node_copy(*left), right ? node::node_copy(*right) : nullptr);
}

template<bool MAKE_COPY>
void get_all_subtrees(const node& root, vector<node>& sub_trees)
{
	auto pos = sub_trees.size();
	[[maybe_unused]] const auto prev_size = pos;
	sub_trees.push_back(root);
	while (pos < sub_trees.size()) {
		// adding children of current element
		const auto& curr = sub_trees[pos];
		if (curr.get_left())
		{
			if constexpr (MAKE_COPY)
				sub_trees.push_back(node::deep_copy(*curr.get_left()));
			else
				sub_trees.push_back(*curr.get_left());
		}
		if (curr.get_right())
		{
			if constexpr (MAKE_COPY)
				sub_trees.push_back(node::deep_copy(*curr.get_right()));
			else
				sub_trees.push_back(*curr.get_right());
		}
		pos++;
	}
	assert(sub_trees.size() - prev_size == root.size());
}
