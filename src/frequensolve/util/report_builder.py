import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from jinja2 import Environment, FileSystemLoader

__all__ = ["Report", "Figure", "Section"]


@dataclass(init=False)
class Figure:
    """A figure included in a generated report.

    Args:
        title: Figure title.
        caption: Figure caption.
        image: Image path.
        width: Fractional LaTeX text width.
    """

    title: str = ""
    caption: str = ""
    image: Path = field(default_factory=Path)
    width: float = 1.0

    def __init__(
        self,
        title: str = "",
        caption: str = "",
        image: Union[str, Path] = "",
        width: float = 1.0,
    ):
        if isinstance(image, str):
            image = Path(image)
        self.image = image.resolve()
        self.title = title
        self.caption = caption
        self.width = width


@dataclass
class Section:
    """A report section containing figures.

    Args:
        title: Section title.
        figures: Initial figures.
    """

    title: str = ""
    figures: List[Figure] = field(default_factory=list)

    def add_figure(
        self,
        figure: Figure,
    ) -> Figure:
        """Add a figure object to the section.

        Args:
            figure: Figure to add.

        Returns:
            Stored figure.
        """
        self.figures.append(figure)
        return self.figures[-1]

    def new_figure(self, title="", caption="", image="", width=1.0) -> Figure:
        """Create, add, and return a new figure.

        Args:
            title: Figure title.
            caption: Figure caption.
            image: Image path.
            width: Fractional LaTeX text width.
        """
        fig = Figure(title=title, caption=caption, image=image, width=width)
        self.figures.append(fig)
        return fig


@dataclass
class Report:
    """LaTeX-backed PDF report description.

    Args:
        title: Report title.
        subtitle: Report subtitle.
        author: Report author.
        sections: Initial report sections.
    """

    title: str = "Simulation Report"
    subtitle: str = "An Automated Simulation Report"
    author: str = ""
    sections: List[Section] = field(default_factory=list)

    def add_section(self, section: Section) -> Section:
        """Add a section to the report.

        Args:
            section: Section to add.

        Returns:
            Stored section.
        """
        self.sections.append(section)
        return self.sections[-1]

    def new_section(self, title: str = "") -> Section:
        """Create, add, and return a new section.

        Args:
            title: Section title.

        Returns:
            Newly added section.
        """
        sec = Section(title=title)
        self.sections.append(sec)
        return sec

    def generate(self, path: Union[str, Path], name="report", timeout: int = 30):
        """Compile the report to PDF using the configured LaTeX template.

        Args:
            path: Output PDF file or directory.
            name: Output report stem.
            timeout: Timeout in seconds for each ``pdflatex`` pass.
        """
        current_dir = os.getcwd()
        path = Path(path)
        # Set up temporary directory
        work_dir = Path(f"/tmp/fstmpdisk/{name}")
        setup_work_directory(work_dir)

        # Load the LaTeX template
        env = Environment(loader=FileSystemLoader(work_dir))
        template = env.get_template("main.tex")

        # Render the LaTeX document
        latex = template.render(report=self)
        with open(work_dir / f"{name}.tex", "w") as f:
            f.write(latex)

        # Compile the LaTeX document to PDF
        try:
            logfile = work_dir / "tex.log"
            with open(logfile, "w") as f:
                # First pass to generate intermediate files
                print("Running first pass of pdflatex...")
                output = subprocess.check_output(
                    ["pdflatex", f"{name}.tex"],
                    cwd=str(work_dir),
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    text=True,
                )
                f.write("\n\n\nFirst Pass:\n\n\n")
                f.write(output)

                # Second pass for TOC, references, etc.
                print("Running second pass of pdflatex...")
                output = subprocess.check_output(
                    ["pdflatex", "report.tex"],
                    cwd=str(work_dir),
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    text=True,
                )
                f.write("\n\n\nSecond Pass:\n\n\n")
                f.write(output)

            file = (
                (path / f"{name}").with_suffix(".pdf") if not path.is_file() else path
            )
            file_work = work_dir / f"{name}.pdf"
            if not file_work == file:
                if file.exists():
                    file.unlink()
                shutil.move(file_work, file)
            print(f"Report generated successfully: {file}")
            print(f"    Compilation log: {logfile}")

        except subprocess.CalledProcessError as e:
            print("Report compilation failed.")
            print(f"Error: {e.output}".encode().decode("unicode_escape"))

        except subprocess.TimeoutExpired as e:
            print("Report generation timed out.")
            print(
                f"Output before timeout:\n{e.output}".encode().decode("unicode_escape")
            )

        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# Utility functions
def setup_work_directory(work_dir: Path):
    """Set up the temporary LaTeX work directory for report generation.

    Args:
        work_dir: Directory that will receive the report template files.
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        from frequensolve._optional import optional_dependency_error

        raise optional_dependency_error(
            "Report work-directory setup",
            extra="hpc",
            dependencies=("python-dotenv",),
            error=exc,
        ) from exc

    load_dotenv()
    fs_dir = os.getenv("FS_SOLVER_PATH")

    # Get template directory
    template_dir = Path(fs_dir) / "trunk/files/templates/report/"
    work_dir.mkdir(parents=True, exist_ok=True)

    if work_dir.exists():
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
