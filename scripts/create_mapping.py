import re
import json
import string
import argparse

# add keyword arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--string', '-s', help='Create a mapping of letters from a string', default=re.sub('[JZ]', '', string.ascii_uppercase))
parser.add_argument('--output', '-o', help='Path to output json file', default='../data/mapping.json')
args = parser.parse_args()

if __name__ == '__main__':
	# store mapping of letters
	s = sorted(set(args.string))
	mapping = {c: i for i, c in enumerate(s)}
	with open(args.output, 'w') as f:
		json.dump(mapping, f, indent=4)