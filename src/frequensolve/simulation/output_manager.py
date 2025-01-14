
# ----------------------------------------------------------------------
# 9. Output
# ----------------------------------------------------------------------
class ParaviewOutput(BaseModel):
    """
    @class ParaviewOutput
    @brief Represents the Paraview subsection in the Output section.
    """
    directory: str = "../output/paraview/"
    components: List[str] = Field(default_factory=lambda: ["pressure"])
    prefix: str = "paraview"
    upscale: int = 1

    def __str__(self) -> str:
        comp_str = " ".join(self.components)
        return (
            "   [Paraview]\n"
            f"      directory  = {self.directory}\n"
            f"      components = {comp_str}\n"
            f"      prefix     = {self.prefix}\n"
            f"      upscale    = {self.upscale}\n"
            "   []\n"
        )


class Output(BaseModel):
    """
    @class Output
    @brief Represents the Output section, which may include multiple sub-outputs (e.g. Paraview).
    """
    directory: str = "../output/"
    paraview_output: Optional[ParaviewOutput] = None

    def set_paraview_output(self, pv_out: ParaviewOutput) -> None:
        self.paraview_output = pv_out

    def __str__(self) -> str:
        out = "[Output]\n"
        out += f"   directory     = {self.directory}\n"
        if self.paraview_output is not None:
            out += str(self.paraview_output)
        out += "[]\n"
        return out
