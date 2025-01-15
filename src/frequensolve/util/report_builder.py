from jinja2 import Environment, FileSystemLoader

import os, sys, shutil
import subprocess
from pathlib import Path

from dataclasses import dataclass, field
from typing import List


@dataclass
class Figure:
   """A figure in a report."""
   title:   str = ""
   caption: str = ""
   image:   str = ""


@dataclass
class Section:
   """A section of a report."""
   title:   str = ""
   figures: List[Figure] = field(default_factory=list)

   def add_figure(self, figure: Figure) -> Figure:
      """Adds a Figure object to the section."""
      self.figures.append(figure)
      return self.figures[-1]

   def new_figure(self, title="", caption="", image="") -> Figure:
      """Adds an empty figure to the section or initializes with given values."""
      fig = Figure(title=title, caption=caption, image=image)
      self.figures.append(fig)
      return fig


@dataclass
class Report:
   title:    str = "Simulation Report"
   subtitle: str = "An Automated Simulation Report"
   author:   str = ""
   sections: List[Section] = field(default_factory=list)

   def add_section(self, section: Section) -> Section:
      """Adds a section to report

      Attributes:
         section (Section): The section to add
      """
      self.sections.append(section)
      return self.sections[-1]

   def new_section(self, title="") -> Section:
      """Adds an empty section to report

      Attributes:
         title (str): Title of the section
      """
      sec = Section(title=title)
      self.sections.append(sec)
      return sec
      
   def generate(self,path, timeout=15):
      """Compiles PDF from LaTeX

      Attributes:
         path (str): Where PDF will be saved
         timeout (int): Timeout for compilation command (stalls on error)
      """
      current_dir = os.getcwd()
      
      # Set up temporary directory
      work_dir = "/tmp/fstmpdisk/report"
      setup_work_directory(work_dir)
   
      # Load the LaTeX template
      env = Environment(loader = FileSystemLoader(work_dir))
      template = env.get_template("main.tex")

      # Render the LaTeX document
      latex = template.render(report = self)
      with open(os.path.join(work_dir,"report.tex"), "w") as f:
          f.write(latex)
          
      # Compile the LaTeX document to PDF
      try:
         logfile = os.path.join(work_dir, "tex.log")
         with open(logfile, "w") as f:
            # First pass to generate intermediate files
            print("Running first pass of pdflatex...")
            output = subprocess.check_output(["pdflatex", "report.tex"],
                                             cwd     = work_dir,
                                             stderr  = subprocess.STDOUT,
                                             timeout = timeout,
                                             text    = True)
            f.write("\n\n\nFirst Pass:\n\n\n")
            f.write(output)
            
            # Second pass for TOC, references, etc.
            print("Running second pass of pdflatex...")
            output = subprocess.check_output(["pdflatex", "report.tex"],
                                             cwd     = work_dir,
                                             stderr  = subprocess.STDOUT,
                                             timeout = timeout,
                                             text    = True)
            f.write("\n\n\nSecond Pass:\n\n\n")
            f.write(output)

         file = os.path.join(path, "report.pdf") if not os.path.isfile(path) else path
         file_work = os.path.join(work_dir,"report.pdf")
         if not file_work == file:
            if os.path.exists(file):
               os.remove(file)
            shutil.move(file_work,file)
         print(f"Report generated successfully: {file}")
         print(f"    Compilation log: {logfile}")
         
      except subprocess.CalledProcessError as e:
         print("Report compilation failed.")
         print(f"Error: {e.output}".encode().decode('unicode_escape'))

      except subprocess.TimeoutExpired as e:
         print("Report generation timed out.")
         print(f"Output before timeout:\n{e.output}".encode().decode('unicode_escape'))
         
      except Exception as e:
         print(f"An unexpected error occurred: {e}")

# Utility functions
def setup_work_directory(work_dir):
   """Sets up the work directory for the report."""
   fs_dir = os.environ["FREQUENSOL_DIR"]

   # Get template directory
   template_dir = os.path.join(fs_dir,"trunk/files/templates/report/")
   Path(work_dir).mkdir(parents=True, exist_ok=True)

   if os.path.exists(work_dir):
      shutil.rmtree(work_dir)
   shutil.copytree(template_dir, work_dir)


# TODO:
# - Setup template, summarizing run parameters
#    - Also summarize resources
# - Verification template (e.g. with convergence studies)
# - Optionally add "input" appendix
# - Add way to insert text
# - Later, add FWI info
# - Performance information
