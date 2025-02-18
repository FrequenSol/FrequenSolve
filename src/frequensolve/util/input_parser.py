import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["InputBlock", "InputParser", "str_to_array"]


@dataclass
class InputBlock:
    """Data structure for organizing input blocks (hierarchical sections).

    Attributes:
       name (str): Block name (e.g., "Problem", "Mesh").
       args (dict): Key-value pairs specifying named arguments.
       sub_blocks (List[InputBlock]): Nested blocks of type InputBlock.
    """

    name: str
    args: dict = field(default_factory=dict)
    sub_blocks: List["InputBlock"] = field(default_factory=list)

    def find_block(self, name: str) -> Optional["InputBlock"]:
        """Recursively search for a sub-block matching the given name.

        Attributes:
           name (str): The block name to find.
           A reference to the matching InputBlock or None if not found.
        """
        if self.name == name:
            return self
        for block in self.sub_blocks:
            result = block.find_block(name)
            if result is not None:
                return result
        return None


# ----------------------------------------------------------------------
# Block Parsing Helpers
# ----------------------------------------------------------------------
def is_begin_block(line: str) -> bool:
    """Determine if a line marks the start of a block.

    Attributes:
       line (str): A line of text to parse.
       True if the line starts a block, False otherwise.
    """
    return bool(re.match(r"^\s*\[\s*[a-zA-Z][a-zA-Z0-9_]*\s*\]", line))


def is_end_block(line: str) -> bool:
    """Determine if a line marks the end of a block (i.e. "[]").

    Attributes:
       line (str): A line of text to parse.
       True if the line is "[]", False otherwise.
    """
    return bool(re.match(r"^\s*\[\s*\]", line))


def is_arg_line(line: str) -> bool:
    """Check if a line contains a key-value pair (with '=').

    Attributes:
       line (str): A line of text to parse.
       True if '=' appears in the line, False otherwise.
    """
    return "=" in line


def read_block(lines: List[str], istart: int) -> Tuple[int, "InputBlock"]:
    """Recursively read and parse a block from the list of lines.

    Attributes:
       lines (List[str]): List of lines from the input file.
       istart (int): Index from which to begin parsing the block.
       (offset, block):
          offset (int): The total number of lines consumed by this block.
          block (InputBlock): The parsed InputBlock.
       ValueError: If block start syntax is invalid.
    """
    i = istart
    name_match = re.match(r"^\s*\[\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\]", lines[i])
    if not name_match:
        raise ValueError(f"Invalid block start at line {i}: {lines[i]}")

    block_name = name_match[1]
    block = InputBlock(name=block_name)
    i += 1

    for iter in range(len(lines)):
        line = lines[i]
        if is_begin_block(line):
            n, sub_block = read_block(lines, i)
            block.sub_blocks.append(sub_block)
            i += n
        elif is_arg_line(line):
            match = re.search(r"([a-zA-Z][a-zA-Z0-9_]*)\s*\=\s*(.*)", line)
            key = match[1].strip()
            val = match[2].strip()
            block.args[key] = val
            i += 1
        elif is_end_block(line):
            i += 1
            break

    return i - istart, block


@dataclass
class InputParser:
    """Parser that reads an input file and organizes data into blocks.

    Attributes:
       file (str): Path to the input file.
       blocks (List[InputBlock]): Top-level blocks parsed from the file.

    """

    file: str
    blocks: List[InputBlock] = field(default_factory=list)

    @classmethod
    def read(cls, input_file: str) -> "InputParser":
        """Class method to parse an input file and return an InputParser instance.

        Attributes:
           input_file (str): Path to the input file.
           An initialized InputParser instance with parsed blocks.
        """
        with open(input_file, "r") as fin:
            txt = fin.read()

        # Remove comments
        txt = re.sub(r"#.*", "", txt)

        # Expand global variables
        i = txt.find("[")
        glob = txt[: i - 1]
        for match in re.findall(r"\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\=\s*(.*)", glob):
            name = match[0].strip()
            val = match[1].strip()
            txt = txt.replace("{" + name + "}", val)

        # Expand sub-files
        for match in re.findall(r"\s*(>>\s*.*)", txt):
            subfile = match.strip().lstrip(">>").strip()
            with open(subfile, "r") as fin:
                subtxt = fin.read()
                subtxt = re.sub(r"#.*", "", subtxt)
            txt = txt.replace(match, subtxt)

        txt = re.sub(r"(\s*\n)+", "\n", txt, flags=re.MULTILINE)
        lines = txt.strip("\n").split("\n")

        iblocks = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if is_begin_block(line):
                n, block = read_block(lines, i)
                iblocks.append(block)
            else:
                n = 1
            i += n

        return cls(file=input_file, blocks=iblocks)

    def get_block(self, block_name: str) -> InputBlock:
        """Retrieve a block by name, searching sub-blocks recursively.

        Attributes:
           block_name (str): The name of the block to find.
           The matching InputBlock.
           ValueError: If no block with the given name is found.
        """
        for block in self.blocks:
            result = block.find_block(block_name)
            if result is not None:
                return result
        raise ValueError(f"Block with name '{block_name}' not found.")

    def get_block_args(self, block_name: str) -> dict:
        """Convenience method to retrieve a dictionary of arguments from a named block.

        Attributes:
           block_name (str): The name of the block.
           A dictionary of arguments for that block.
           ValueError: If the block is not found.
        """
        block = self.get_block(block_name)
        return block.args

    @property
    def sweep_params(self) -> Tuple[float, float, float]:
        """Parse the parameter sweep block for frequency sweep parameters.

        Attributes:
           (f_min, f_max, df):
              f_min (float): Minimum frequency.
              f_max (float): Maximum frequency.
              df (float): Frequency step size.
           ValueError: If 'freq' has an invalid format.
        """
        block = self.get_block("ParameterSweep")
        freq_range = re.search(r"{([\d:.]*)}", block.args["freq"])
        if not freq_range:
            raise ValueError(
                "Invalid frequency range format in 'ParameterSweep' block."
            )

        f_min, f_max, df = map(float, freq_range[1].split(":"))
        return f_min, f_max, df


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def str_to_array(str_val: str) -> Optional[np.ndarray]:
    """
    @brief  Convert a whitespace-delimited string to a NumPy array of floats.
    @param  str_val  The string to parse.
    @return A NumPy array of floats, or None if the string is empty.
    """
    if str_val:
        return np.array(list(map(float, re.split(r"\s+", str_val.strip()))))
    else:
        return None


def get_njob(file: str) -> int:
    """
    @brief  Compute the number of jobs from the frequency sweep parameters in the input file.
    @param  file  Path to the input file.
    @return n_jobs = number of discrete steps in the frequency range.
    """
    input = InputParser.read(file)
    f_min, f_max, df = input.sweep_params
    n_jobs = int((f_max - f_min) / df) + 1
    return n_jobs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 input_parser.py <file_path>")
    else:
        print(get_njob(sys.argv[1]))
