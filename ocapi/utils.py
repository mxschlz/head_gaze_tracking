import yaml


def load_yaml_file(file_path):
	"""
	Loads data from a YAML file.

	Args:
		file_path: The path to the YAML file.

	Returns:
		The loaded YAML data as a Python dictionary or list, or None if an error occurs.
	"""

	try:
		with open(file_path, 'r') as file:
			data = yaml.safe_load(file)
		return data
	except FileNotFoundError:
		print(f"Error: File not found at '{file_path}'")
		return None
	except yaml.YAMLError as e:
		print(f"Error parsing YAML file: {e}")
		return None




if __name__ == "__main__":
	file = load_yaml_file(file_path="/config.yml")
