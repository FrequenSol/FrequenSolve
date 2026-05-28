"""Parser for legacy block-structured fast-solver input files."""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["InputBlock", "InputParser", "str_to_array"]


@dataclass
class InputBlock:
    """One named block from a legacy solver input file.

    Attributes:
        name: Block name, such as ``"Problem"`` or ``"Mesh"``.
        args: Key-value pairs parsed from assignment lines in the block.
        sub_blocks: Nested child blocks.
    """

    name: str
    args: dict = field(default_factory=dict)
    sub_blocks: List["InputBlock"] = field(default_factory=list)

    def find_block(self, name: str) -> Optional["InputBlock"]:
        """Recursively search for a sub-block matching the given name.

        Args:
            name: Block name to find.

        Returns:
            The matching block, or ``None`` if this block tree does not contain
            it.
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

    Args:
        line: Line of text to parse.

    Returns:
        ``True`` when the line starts a named block.
    """
    return bool(re.match(r"^\s*\[\s*[a-zA-Z][a-zA-Z0-9_]*\s*\]", line))


def is_end_block(line: str) -> bool:
    """Determine if a line marks the end of a block (i.e. "[]").

    Args:
        line: Line of text to parse.

    Returns:
        ``True`` when the line closes the current block.
    """
    return bool(re.match(r"^\s*\[\s*\]", line))


def is_arg_line(line: str) -> bool:
    """Check if a line contains a key-value pair (with '=').

    Args:
        line: Line of text to parse.

    Returns:
        ``True`` when the line contains an assignment.
    """
    return "=" in line


def read_block(lines: List[str], istart: int) -> Tuple[int, "InputBlock"]:
    """Recursively read and parse a block from the list of lines.

    Args:
        lines: Lines from the input file.
        istart: Index where the block begins.

    Returns:
        A tuple of ``(offset, block)`` where ``offset`` is the number of lines
        consumed and ``block`` is the parsed input block.

    Raises:
        ValueError: If the block start syntax is invalid.
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
    """Parsed representation of a legacy block-structured input file.

    Attributes:
        file: Path to the input file.
        blocks: Top-level blocks parsed from the file.
    """

    file: str
    blocks: List[InputBlock] = field(default_factory=list)

    @classmethod
    def read(cls, input_file: str) -> "InputParser":
        """Parse an input file and return an ``InputParser`` instance.

        Args:
            input_file: Path to the input file.

        Returns:
            An initialized parser with top-level blocks populated.
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

        Args:
            block_name: Name of the block to find.

        Returns:
            The matching input block.

        Raises:
            ValueError: If no block with the given name is found.
        """
        for block in self.blocks:
            result = block.find_block(block_name)
            if result is not None:
                return result
        raise ValueError(f"Block with name '{block_name}' not found.")

    def get_block_args(self, block_name: str) -> dict:
        """Convenience method to retrieve a dictionary of arguments from a named block.

        Args:
            block_name: Name of the block.

        Returns:
            The block's parsed key-value arguments.

        Raises:
            ValueError: If the block is not found.
        """
        block = self.get_block(block_name)
        return block.args

    @property
    def sweep_params(self) -> Tuple[float, float, float]:
        """Parse the parameter sweep block for frequency sweep parameters.

        Returns:
            ``(f_min, f_max, df)`` frequency sweep values.

        Raises:
            ValueError: If ``freq`` has an invalid format.
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
