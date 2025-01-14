import sys, os, re
import numpy as np

from typing import List, Tuple, Optional
from dataclasses import dataclass, field

__all__ = ['InputBlock', 'InputParser', 'str_to_array']

@dataclass
class InputBlock:
   """
   @class  InputBlock
   @brief  Data structure for organizing input blocks (hierarchical sections).
   @details Each block can contain:
            - A name
            - A dictionary of key-value arguments
            - A list of sub-blocks (nested blocks)

   @param name        Block name (e.g., "Problem", "Mesh").
   @param args        Key-value pairs specifying named arguments.
   @param sub_blocks  Nested blocks of type InputBlock.
   """
   name: str
   args: dict = field(default_factory=dict)
   sub_blocks: List['InputBlock'] = field(default_factory=list)

   def find_block(self, name: str) -> Optional['InputBlock']:
      """
      @brief  Recursively search for a sub-block matching the given name.
      @param  name  The block name to find.
      @return A reference to the matching InputBlock or None if not found.
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
   """
   @brief  Determine if a line marks the start of a block.
   @param  line  A line of text to parse.
   @return True if the line starts a block, False otherwise.
   """
   return bool(re.match(r'^\s*\[\s*[a-zA-Z][a-zA-Z0-9_]*\s*\]', line))

def is_end_block(line: str) -> bool:
   """
   @brief  Determine if a line marks the end of a block (i.e. "[]").
   @param  line  A line of text to parse.
   @return True if the line is "[]", False otherwise.
   """
   return bool(re.match(r'^\s*\[\s*\]', line))

def is_arg_line(line: str) -> bool:
   """
   @brief  Check if a line contains a key-value pair (with '=').
   @param  line  A line of text to parse.
   @return True if '=' appears in the line, False otherwise.
   """
   return "=" in line

def read_block(lines: List[str], istart: int) -> Tuple[int, 'InputBlock']:
   """
   @brief  Recursively read and parse a block from the list of lines.
   @param  lines   List of lines from the input file.
   @param  istart  Index from which to begin parsing the block.
   @return (offset, block)
           offset: The total number of lines consumed by this block.
           block:  The parsed InputBlock.
   @throws ValueError if block start syntax is invalid.
   """
   i = istart
   name_match = re.match(r'^\s*\[\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\]', lines[i])
   if not name_match:
      raise ValueError(f"Invalid block start at line {i}: {lines[i]}")

   block_name = name_match[1]
   block = InputBlock(name=block_name)
   i += 1

   for iter in range(len(lines)):
      line = lines[i]
      if (is_begin_block(line)):
         n, sub_block = read_block(lines, i)
         block.sub_blocks.append(sub_block)
         i += n
      elif is_arg_line(line):
         match = re.search(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\=\s*(.*)', line)
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
   """
   @class  InputParser
   @brief  Parser that reads an input file and organizes data into blocks.
   @details The parser loads the file, processes comments, expands variables/sub-files,
            and provides query methods to retrieve blocks and arguments.

   @param file    Path to the input file.
   @param blocks  Top-level blocks parsed from the file (list of InputBlock).
   """
   file: str
   blocks: List[InputBlock] = field(default_factory=list)
   
   @classmethod
   def read(cls, input_file: str) -> 'InputParser':
      """
      @brief   Class method to parse an input file and return an InputParser instance.
      @details - Removes comments (#...)
               - Substitutes global variables (e.g. {NAME})
               - Includes sub-files (lines with '>> <filename>')
               - Collects blocks ([Name], [SubName], etc.)
      @param   input_file  Path to the input file.
      @return  An initialized InputParser instance with parsed blocks.
      """
      with open(input_file, 'r') as fin:
         txt = fin.read()
   
      # Remove comments
      txt = re.sub(r'#.*', "", txt)
         
      # Expand global variables
      i = txt.find("[")
      glob = txt[:i-1]
      for match in re.findall(r'\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\=\s*(.*)', glob):
         name = match[0].strip()
         val  = match[1].strip()
         txt = txt.replace("{" + name + "}", val)
         
      # Expand sub-files
      for match in re.findall(r'\s*(>>\s*.*)', txt):
         subfile = match.strip().lstrip('>>').strip()
         with open(subfile, 'r') as fin:
            subtxt = fin.read()
            subtxt = re.sub(r'#.*', "", subtxt)
         txt = txt.replace(match, subtxt)
         
      txt = re.sub(r'(\s*\n)+', '\n', txt, flags=re.MULTILINE)
      lines = txt.strip("\n").split("\n")
      
      iblocks = []
      
      i = 0
      while i < len(lines):
         line = lines[i]
         if (is_begin_block(line)):
            n, block = read_block(lines, i)
            iblocks.append(block)
         else:
            n = 1
         i += n
      
      return cls(file = input_file, blocks = iblocks)

   def get_block(self, block_name: str) -> InputBlock:
      """
      @brief  Retrieve a block by name, searching sub-blocks recursively.
      @param  block_name  The name of the block to find.
      @return The matching InputBlock.
      @throws ValueError if no block with the given name is found.
      """
      for block in self.blocks:
         result = block.find_block(block_name)
         if result is not None:
            return result
      raise ValueError(f"Block with name '{block_name}' not found.")

   def get_block_args(self, block_name: str) -> dict:
      """
      @brief  Convenience method to retrieve a dictionary of arguments from a named block.
      @param  block_name  The name of the block.
      @return A dictionary of arguments for that block.
      @throws ValueError if the block is not found.
      """
      block = self.get_block(block_name)
      return block.args

   @property
   def sweep_params(self) -> Tuple[float, float, float]:
      """
      @brief Parse the parameter sweep block for frequency sweep parameters.
      @return (f_min, f_max, df)
      @throws ValueError if 'freq' has an invalid format.
      """
      block = self.get_block("ParameterSweep")
      freq_range = re.search(r'{([\d:.]*)}', block.args["freq"])
      if not freq_range:
         raise ValueError("Invalid frequency range format in 'ParameterSweep' block.")

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
      return np.array(list(map(float, re.split(r'\s+', str_val.strip()))))
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
