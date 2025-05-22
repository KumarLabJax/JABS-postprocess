from __future__ import annotations

from typing import List, Union

import numpy as np

from jabs_postprocess.utils.features import JABSFeature


class Expression:
	"""Float vector of an expression."""
	def __init__(self, data: np.ndarray, description: str):
		"""Initializes an expression vector with data and a description."""
		self._data = np.asarray(data, dtype=np.float64)
		self._description = str(description)

	@property
	def data(self):
		"""Stored float vector data."""
		return self._data

	@property
	def description(self):
		"""String description of the expression."""
		return self._description

	def add(self, other: Expression):
		"""Addition operator to another expression."""
		return Expression(np.add(self.data, other.data), f'({self.description} + {other.description})')

	def subtract(self, other: Expression):
		"""Subtraction operator to another expression."""
		return Expression(np.subtract(self.data, other.data), f'({self.description} - {other.description})')

	def multiply(self, other: Expression):
		"""Multiplication operator to another expression."""
		return Expression(np.multiply(self.data, other.data), f'({self.description} * {other.description})')

	def divide(self, other: Expression):
		"""Division operator to another expression."""
		return Expression(np.divide(self.data, other.data), f'({self.description} / {other.description})')

	def abs(self):
		"""Applies the absolute value to an expression."""
		return Expression(np.abs(self.data), f'abs({self.description})')

	@classmethod
	def from_config(cls, feature: JABSFeature, config: Union[dict, str]):
		"""Construct an expression vector based on configuration.

		Args:
			feature: feature file where vector data will be read
			config: configuration dict of an expression or a string of feature vector data to read
		"""
		if isinstance(config, dict):
			key, values = config.copy().popitem()
			ops = [cls.from_config(feature, x) for x in values]
			if key == 'add':
				assert len(ops) == 2
				return ops[0].add(ops[1])
			elif key == 'subtract':
				assert len(ops) == 2
				return ops[0].subtract(ops[1])
			elif key == 'multiply':
				assert len(ops) == 2
				return ops[0].multiply(ops[1])
			elif key == 'divide':
				assert len(ops) == 2
				return ops[0].divide(ops[1])
			elif key == 'abs':
				assert len(ops) == 1
				return ops[0].abs()
			raise ValueError(f'Operation {key} not recognized.')
		else:
			# Check if the current string is actually a constant scalar
			try:
				scalar = float(config)
				return cls(scalar, config)
			# String is a key to be read from the feature file
			except ValueError:
				return cls(feature.get_key_data(config), config)

	def __str__(self):
		return self.description

	def __repr__(self):
		return self.__str__()


class Relation:
	"""Boolean vector of an inequality or logical relation."""
	def __init__(self, operands: List[Union[Relation, Expression]], relation: str):
		"""Constructor from 2 expressions and a relation.
		
		Args:
			operands: Relation or Expression objects to combine
			relation: how to combine the operands

		"""
		assert len(operands) > 0
		# Basic inequalities
		if relation in self.get_inequality_options():
			# Inequalities have to operate on expressions, not boolean relations
			assert len(operands) == 2
			if any([isinstance(x, Relation) for x in operands]):
				raise ValueError(f'Inequality operations can only summarize expression vectors, but at least 1 boolean vector was included. Included expression: {", ".join([x.description for x in operands])}')

			if relation in self.get_less_than_options():
				self._data = np.less(operands[0].data, operands[1].data)
				self._description = f'{operands[0].description} < {operands[1].description}'
			elif relation in self.get_less_than_equal_options():
				self._data = np.less_equal(operands[0].data, operands[1].data)
				self._description = f'{operands[0].description} <= {operands[1].description}'
			elif relation in self.get_geater_than_options():
				self._data = np.greater(operands[0].data, operands[1].data)
				self._description = f'{operands[0].description} > {operands[1].description}'
			elif relation in self.get_geater_than_equal_options():
				self._data = np.greater_equal(operands[0].data, operands[1].data)
				self._description = f'{operands[0].description} >= {operands[1].description}'
			# NANs always evaluate to false, so change to 3-state
			# -1: NAN in comparison
			# 0: False
			# 1: True
			self._data = self._data.astype(np.int8)
			self._data[np.isnan(operands[0].data)] = -1
			self._data[np.isnan(operands[1].data)] = -1

		# Logical operations
		elif relation in self.get_logical_options():
			# Logical operations cannot operate on expressions
			# Some logical operations require 1 operand to be a scalar
			scalar_ops = [x for x in operands if x.data.ndim == 0]
			expression_ops = [x for x in operands if not x.data.ndim == 0]
			call_matrix = np.stack([x.data for x in expression_ops])
			if len(scalar_ops) > 0:
				scalars = np.stack([x.data for x in scalar_ops])
			else:
				scalars = []
			op_str = ", ".join([x.description for x in expression_ops])
			if any([isinstance(x, Expression) for x in expression_ops]):
				raise ValueError(f'Logical operations can only summarize boolean vectors, but at least 1 expression was included. Included comparisons: {op_str}')

			if relation in self.get_any_options():
				self._data = np.any(call_matrix > 0, axis=0)
				self._description = f'Any of: ({op_str})'
			if relation in self.get_all_options():
				self._data = np.all(call_matrix > 0, axis=0)
				self._description = f'All of: ({op_str})'
			if relation in self.get_minimum_options():
				assert len(scalars) == 1
				self._data = np.sum(call_matrix > 0, axis=0) >= scalars[0]
				self._description = f'Minimum of {scalars[0]}: ({op_str})'
			if relation in self.get_maximum_options():
				assert len(scalars) == 1
				self._data = np.sum(call_matrix > 0, axis=0) <= scalars[0]
				self._description = f'Maximum of {scalars[0]}: ({op_str})'

			self._data = self._data.astype(np.int8)
			# Adjust the calls to properly handle NANs
			# If any of the sub-calls were unable to make a call, don't make a call
			missing_calls = np.any(call_matrix == -1, axis=0)
			self._data[missing_calls] = -1
		else:
			raise ValueError(f'Relation "{relation}" not recognized.')

	@property
	def data(self):
		"""Stored boolean vector data."""
		return self._data

	@property
	def description(self):
		"""Stored description."""
		return self._description

	@classmethod
	def from_config(cls, feature: JABSFeature, config: dict):
		"""Constructs a relation based on configuration.

		Args:
			feature: feature file where the expression data is read
			config: configuration dict of a relation of expressions

		Notes:
			This function is designed to recursively walk the tree in the config.
		"""
		key, operand_list = config.copy().popitem()
		evaluated_list = []
		for cur_operand in operand_list:
			# If the next operand is a string or does not have a relation key, it's an expression
			if isinstance(cur_operand, (str, int, float)) or not list(cur_operand.keys())[0] in cls.get_available_options():
				evaluated_list.append(Expression.from_config(feature, cur_operand))
			else:
				evaluated_list.append(Relation.from_config(feature, cur_operand))
		return cls(evaluated_list, key)

	@staticmethod
	def get_less_than_options():
		"""List of strings that represent less than."""
		return ['<', 'lt', 'less than']

	@staticmethod
	def get_less_than_equal_options():
		"""List of strings that represent less than or equal to."""
		return ['<=', '=<', 'lte', 'less than equal', 'less than or equal']

	@staticmethod
	def get_geater_than_options():
		"""List of strings that represent greater than."""
		return ['>', 'gt', 'greater than']

	@staticmethod
	def get_geater_than_equal_options():
		"""List of strings that represent greater than or equal to."""
		return ['>=', '=>', 'gte', 'greater than equal', 'greater than or equal']

	@staticmethod
	def get_any_options():
		"""List of strings that represent a logical "or" operation."""
		return ['any', 'or', '|']

	@staticmethod
	def get_all_options():
		"""List of strings that represent a logical "and" operation."""
		return ['all', 'and', '&']

	@staticmethod
	def get_minimum_options():
		"""List of strings that represent an "at least" operation."""
		return ['minimum', 'at least']

	@staticmethod
	def get_maximum_options():
		"""List of strings that represent "at most" operation."""
		return ['maximum', 'at most']

	@staticmethod
	def get_inequality_options():
		"""List of all inequality options."""
		return Relation.get_less_than_options() + Relation.get_less_than_equal_options() + Relation.get_geater_than_options() + Relation.get_geater_than_equal_options()

	@staticmethod
	def get_logical_options():
		"""List of all logical operations."""
		return Relation.get_any_options() + Relation.get_all_options() + Relation.get_minimum_options() + Relation.get_maximum_options()

	@staticmethod
	def get_available_options():
		"""All options availabile in this module."""
		return Relation.get_inequality_options() + Relation.get_logical_options()

	def __str__(self):
		return self.description

	def __repr__(self):
		return self.__str__()
