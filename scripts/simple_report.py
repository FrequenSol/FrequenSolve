import os, sys

try:
   def add_model_section(section):
      def gen_figures():

         images = {}
         
         # Set up paraview manager
         pv = ParaviewManager(input_file = input_file,
                              fontsize   = 32)

         for prop in ["Vp", "Vs"]:
            pv.show_field(field     = prop,
                          colorbar  = True,
                          title     = f"{prop} (km/s)",
                          colormap  = "YGB_r",
                          show_pml  = False)
            pv.show_axes()
         
            # Plot sources and receivers
            pv.show_sources()
            pv.show_receivers()
         
            # Save screenshot
            file = os.path.join(report_dir,f"{prop}.pdf")
            pv.screenshot(file)
            images[prop] = file
         
         return images
         
      images = gen_figures()
      
      # Add "Model" section
      section.add_figure(
         Figure(
            title = r"Compressive Velocity \texorpdfstring{$(V_p)$}{(Vp)}",
            image = images["Vp"],
         )
      )
      section.add_figure(
         Figure(
            title = r"Shear Velocity \texorpdfstring{$(V_s)$}{(Vs)}",
            image = images["Vs"],
         )
      )
except BaseException as e:
   print(f"{e}")


def add_gather_section(section):

   acq = Acquisition.from_file(input_file, upscale = 2)
   
   ifig = 0
   for field in acq.list_fields():
      for source in acq.list_sources():
         ifig += 1
         file = os.path.join(report_dir,f"gather_{ifig}.pdf")
         if "field" in field:
            pass
         else:
            shot = acq.read_shot_TD(field,source)
            plot_gather(shot,
                        A     = 0.2,
                        cmap  = "gray",
                        Tf    = 3.2,
                        units = "km",
                        save  = file)

            section.add_figure(
               Figure(
                  title = f"Shot {source} -- {field}".replace("_",""),
                  image = file
               )
            )


if __name__ == '__main__':
   input_file = sys.argv[1]
   
   # Try handle virtual env if provided
   if '--virtual-env' in sys.argv:
      virtualEnvPath = sys.argv[sys.argv.index('--virtual-env') + 1]
      virtualEnv = virtualEnvPath + '/bin/activate_this.py'
      if sys.version_info.major < 3:
         execfile(virtualEnv, dict(__file__=virtualEnv))
      else:
         exec(open(virtualEnv).read(), {'__file__': virtualEnv})
         
   from frequensolve         import *
   from frequensolve.seismic import *
   from frequensolve.util    import *
   
   # Get output directory
   input = InputParser.read(input_file)
   report_dir = os.path.abspath(input.get_block("Problem").args["directory"])
   
   # Generate report object
   report = Report(title="Simulation Report", subtitle="Desert Model")
   
   try:
      # Add model section
      section = report.new_section("Model")
      add_model_section(section)
   except BaseException as e:
      print(f"{e}")
   
   # Add gather section
   section = report.new_section("Gathers")
   add_gather_section(section)
   
   report.generate(report_dir)
