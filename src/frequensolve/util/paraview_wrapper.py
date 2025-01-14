#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

import os, sys
import numpy as np
from paraview.simple import *

from dataclasses import dataclass, field
from typing import List, Optional

from .input_parser import *

paraview.simple._DisableFirstRenderCameraReset()

__all__ = ['ParaviewManager']

cmaps = {
   'YGB': 'Yellow - Gray - Blue',
   'earth': 'Blue - Green - Orange',
   'RdGy': 'Gray and Red',
   'viridis': 'Viridis (matplotlib)',
   'turbo': 'Turbo',
   'BBW': 'Black, Blue and White',
   'fast': 'Fast',
   'cool2warm': 'Cool to Warm',
   'cool2warm_ext': 'Cool to Warm (Extended)',
   'inferno': 'Inferno (matplotlib)'
}

def read_xmf(file):
   """
   @brief   Read XMF files using the XDMFReader.
   @param   file  Path to the XMF file.
   @return  An XDMFReader object linked to the file.
   """
   return XDMFReader(registrationName=f"{file}*", FileNames=[file])

def get_labels(x0, x1, dx):
   """
   @brief   Generate axis labels within the range [x0, x1] with step dx.
   @param   x0  Lower bound of the range.
   @param   x1  Upper bound of the range.
   @param   dx  Step size for labels.
   @return  A list of float labels spanning [x0, x1].
   """
   X0 = np.floor(x0 / dx) * dx
   X1 = np.ceil(x1 / dx) * dx + dx / 2.0
   return [x for x in np.arange(X0, X1, dx) if x0 <= x <= x1]


@dataclass
class ColorMapManager:
   """
   @class   ColorMapManager
   @brief   Manages color mapping for ParaView fields.
   @details Allows applying presets, inverting, opacity control, and LUT rescaling.
   """
   def apply_colormap(self, fieldLUT, colormap, invert=False):
      """
      @brief   Apply a color map preset to a field LUT.
      @param   fieldLUT   The LookUpTable for the field.
      @param   colormap   A string key to look up in cmaps.
      @param   invert     If True, the map is inverted after applying.
      """
      if invert:
         colormap = colormap[:-2]
      fieldLUT.ApplyPreset(cmaps[colormap], True)
      if invert:
         fieldLUT.InvertTransferFunction()

   def configure_opacity(self, fieldLUT, enable_opacity):
      """
      @brief   Enable or disable opacity mapping.
      @param   fieldLUT       The LookUpTable for the field.
      @param   enable_opacity True to enable opacity, False otherwise.
      """
      fieldLUT.EnableOpacityMapping = 1 if enable_opacity else 0

   def rescale_lut(self, fieldLUT, limits):
      """
      @brief   Rescale LUT to provided limits or data range.
      @param   fieldLUT  The LookUpTable for the field.
      @param   limits    Either None or (min, max) for explicit LUT scaling.
      """
      if limits:
         fieldLUT.RescaleTransferFunction(limits[0], limits[1])
      else:
         fieldLUT.RescaleTransferFunctionToDataRange(True)


class DataManager:
   """
   @class   DataManager
   @brief   Parses the input file and reads in simulation data for Paraview.
   @details Loads field, source, and receiver data (if available) as XMF.
   """
   def __init__(self, input_file):
      """
      @brief Constructor for DataManager.
      @param input_file Path to the input configuration file.
      """
      self.input_file = input_file
      self._initialize_input_data()

   def _initialize_input_data(self):
      """
      @brief   Parse input file and set relevant data properties (dimension, bounds, etc.).
      @details Also reads the main field, sources, and receivers if available.
      """
      input = InputParser.read(self.input_file)
      self.dim = int(input.get_block("Problem").args["dimension"])
      
      solve_on = input.get_block("Solver").args["solve_on"]
      ngrids = input.get_block("Solver").args["n_grids"]
      self.ngrids = 1 if solve_on == "final" else ngrids
      
      mesh_block = input.get_block("Mesh")
      self.x0 = str_to_array(mesh_block.args["l_bound"])
      self.x1 = str_to_array(mesh_block.args["u_bound"])
      
      # Get paraview output directory
      dir    = input.get_block("Problem").args["directory"]
      dir    = input.get_block("Paraview").args.get("directory", dir)
      prefix = input.get_block("Paraview").args["prefix"]
      comps  = input.get_block("Paraview").args["components"]
      
      subdir      = os.path.join(dir, f"{prefix}")
      field_file  = os.path.join(dir, f"{prefix}_{ngrids.rjust(5,'0')}.xmf")
      src_file    = os.path.join(subdir, "sources.xmf")
      recv_file   = os.path.join(subdir, "receivers.xmf")
      
      # Read field
      self.field_XMF = read_xmf(field_file)
      
      # Read Sources (if exists)
      if os.path.exists(src_file):
         self.source_XMF = read_xmf(src_file)
      else:
         self.source_XMF = None
         
      # Read Receivers (if exists)
      if os.path.exists(recv_file):
         self.receiver_XMF = read_xmf(recv_file)
      else:
         self.receiver_XMF = None
      

@dataclass
class ColorbarSetter:
   """
   @class  ColorbarSetter
   @brief  Manages and applies colorbar properties (title, size, font, etc.).
   """
   font:       str   = 'Arial'
   fontsize:   int   = 30
   bold:       int   = 0
   thickness:  int   = 35
   length:     float = 0.4
   location_x: float = 0.9
   location_y: float = 0.3
   format:     Optional[str]       = None
   labels:     Optional[List[float]] = None
   title:      Optional[str]       = None
         
   def apply(self, field, title, renderView):
      """
      @brief   Apply colorbar settings to the given field in the specified renderView.
      @param   field       The name of the field to which the colorbar applies.
      @param   title       The colorbar title (if any).
      @param   renderView  The ParaView render view object.
      """
      fieldLUT = GetColorTransferFunction(field)
      colorBar = GetScalarBar(fieldLUT, renderView)
      
      self.set_title(colorBar, title)
      self.set_font(colorBar)
      self.set_size(colorBar)
      self.set_location(colorBar)
      self.set_labels(field, colorBar)

   def set_title(self, colorBar, title):
      """
      @brief   Set the colorbar title and remove component title.
      @param   colorBar Colorbar proxy object.
      @param   title    Title text to display.
      """
      if title:
         colorBar.Title = title
      colorBar.ComponentTitle = ""
         
   def set_font(self, colorBar,
                font: Optional[str] = None,
                size: Optional[int] = None,
                bold: Optional[int] = None):
      """
      @brief Set colorbar font properties.
      @param colorBar Colorbar proxy object.
      @param font     Font name or path.
      @param size     Font size.
      @param bold     1 if bold, 0 otherwise.
      """
      if font is None:
         font = self.font
      if size is None:
         size = self.fontsize
      if bold is None:
         bold = 0
         
      if font in ['Arial']:
         colorBar.TitleFontFamily = font
         colorBar.LabelFontFamily = font
         colorBar.TitleBold = bold
         colorBar.LabelBold = bold
      else:
         colorBar.TitleFontFamily = 'File'
         colorBar.LabelFontFamily = 'File'
         colorBar.TitleFontFile = font
         colorBar.LabelFontFile = font
      
      colorBar.TitleFontSize = size
      colorBar.LabelFontSize = size

   def set_size(self, colorBar,
                thickness: Optional[int] = None,
                length: Optional[float] = None):
      """
      @brief Set colorbar thickness and length.
      @param colorBar   Colorbar proxy object.
      @param thickness  Bar thickness in pixels.
      @param length     Bar length in normalized coordinates [0,1].
      """
      if thickness is None:
         thickness = self.thickness
      if length is None:
         length = self.length
         
      colorBar.ScalarBarThickness = thickness
      colorBar.ScalarBarLength = length
      
   def set_location(self, colorBar,
                    location: Optional[List[float]] = None):
      """
      @brief   Set colorbar location in the render view.
      @param   colorBar  Colorbar proxy object.
      @param   location  [x, y] positions in normalized display coordinates.
      """
      if location is None:
         location = [self.location_x, self.location_y]
         
      colorBar.WindowLocation = 'Any Location'
      colorBar.Position       = location
      
   def set_labels(self, field, colorBar, labels: Optional[List[float]] = None):
      """
      @brief   Define custom labels or generate them automatically.
      @param   field    Field name to label.
      @param   colorBar The colorbar proxy object.
      @param   labels   Custom label list or None for auto-generation.
      """
      if labels is None:
         # Create custom labels
         fieldLUT = GetColorTransferFunction(field)
         fieldLUT.RescaleTransferFunctionToDataRange(True)
         
         cmin, cmax = [fieldLUT.RGBPoints[0], fieldLUT.RGBPoints[-4]]
         
         rng = np.max(cmax - cmin)   # Colorbar range
         nL = int(np.log2(rng))      # Range as a power of 2
         dL = 2**(nL-2)              # Between 4 and 8 labels
         labels = get_labels(cmin, cmax, dL)
      
      # Define labels
      colorBar.AddRangeLabels  = 0
      colorBar.UseCustomLabels = 1
      colorBar.CustomLabels    = labels

      if self.format == "auto":
         colorBar.AutomaticLabelFormat = 1
      else:
         colorBar.AutomaticLabelFormat = 0
         if self.format is None:
            d  = labels[1] - labels[0]
            ds = f"{dL:.8f}".rstrip("0")
            precision = np.maximum(1, len(ds.split(".")[1]))
            colorBar.LabelFormat = f"%-#6.{precision}f"
         else:
            colorBar.LabelFormat = self.format


@dataclass
class AxesManager:
   """
   @class  AxesManager
   @brief  Controls the appearance and labeling of axes in the render view.
   @details Includes custom font, bold, color, and labeling for X, Y, Z axes.
   """
   dim:        int
   font:       str   = 'Arial'
   fontsize:   int   = 30
   bold:       int   = 0
   units:      str   = "km"
   precision:  int   = 2
   titles:     Optional[List[str]] = None
   labels:     Optional[List[List[float]]] = None
   color = [0.0, 0.0, 0.0]
   axes_to_label: int   = 7
   
   def __post_init__(self):
      """
      @brief   Called automatically after dataclass __init__ to set default titles and labels.
      """
      if self.titles is None:
         if self.dim == 2:
            self.titles = [ f"X ({self.units})\n",
                            f"Depth ({self.units})     ",
                            f"" ]
         else:
            self.titles = [ f"X ({self.units})\n",
                            f"Y ({self.units})\n",
                            f"Depth ({self.units})     " ]
                       
      if self.labels is None:
         self.labels = [None, None, None]

   def apply(self, renderView):
      """
      @brief Apply configured axes properties (titles, labels, font, color, etc.) to a render view.
      @param renderView The ParaView render view object.
      """
      self.set_titles(renderView)
      self.set_labels(renderView)
      self.set_font(renderView)
      self.set_bold(renderView)
      self.set_color(renderView)
      self.set_axes_to_label(renderView)
      
   def set_color(self, renderView, color: Optional[List[float]] = None):
      """
      @brief Set the axes grid color in the render view.
      @param renderView  The ParaView render view object.
      @param color       [R, G, B] color list or None for default.
      """
      if color is None:
         color = self.color
      renderView.AxesGrid.GridColor = color
      
   def set_axes_to_label(self, renderView, axes_to_label: Optional[int] = None):
      """
      @brief Configure which axes to label (bitmask).
      @param renderView     The ParaView render view object.
      @param axes_to_label  e.g., 7 means show X, Y, Z labels (1+2+4).
      """
      if axes_to_label is None:
         axes_to_label = self.axes_to_label
      renderView.AxesGrid.AxesToLabel = axes_to_label

   def set_titles(self, renderView, titles: Optional[List[str]] = None):
      """
      @brief Set custom or default titles for X, Y, Z axes.
      @param renderView The ParaView render view object.
      @param titles     A list of three titles [XTitle, YTitle, ZTitle].
      """
      if titles is None:
         titles = self.titles
                       
      renderView.AxesGrid.XTitle = titles[0]
      renderView.AxesGrid.YTitle = titles[1]
      renderView.AxesGrid.ZTitle = titles[2]
      
   def set_labels(self, renderView, labels: Optional[List] = None, precision: Optional[int] = None):
      """
      @brief Set custom axis labels for X, Y, Z.
      @param renderView  The ParaView render view object.
      @param labels      e.g. [[x_labels], [y_labels], [z_labels]] or None if auto.
      @param precision   Decimal precision to apply if custom labels are used.
      """
      if labels is None:
         labels = self.labels
      if precision is None:
         precision = self.precision
         
      # X
      if labels[0] is None:
         renderView.AxesGrid.XAxisUseCustomLabels = 0
      else:
         renderView.AxesGrid.XAxisUseCustomLabels = 1
         renderView.AxesGrid.XAxisPrecision       = precision
         renderView.AxesGrid.XAxisLabels          = labels[0]
      
      # Y
      if labels[1] is None:
         renderView.AxesGrid.YAxisUseCustomLabels = 0
      else:
         renderView.AxesGrid.YAxisUseCustomLabels = 1
         renderView.AxesGrid.YAxisPrecision       = precision
         renderView.AxesGrid.YAxisLabels          = labels[1]
      
      # Z
      if labels[2] is None:
         renderView.AxesGrid.ZAxisUseCustomLabels = 0
      else:
         renderView.AxesGrid.ZAxisUseCustomLabels = 1
         renderView.AxesGrid.ZAxisPrecision       = precision
         renderView.AxesGrid.ZAxisLabels          = labels[2]
      
   def set_font(self, renderView, font: Optional[str] = None, size: Optional[int] = None):
      """
      @brief Set font family and size for X, Y, Z axis titles and labels.
      @param renderView The ParaView render view object.
      @param font       Font name or path.
      @param size       Font size.
      """
      if font is None:
         font = self.font
      if size is None:
         size = self.fontsize
   
      if font in ['Arial']:
         renderView.AxesGrid.XTitleFontFamily = font
         renderView.AxesGrid.YTitleFontFamily = font
         renderView.AxesGrid.ZTitleFontFamily = font
         
         renderView.AxesGrid.XLabelFontFamily = font
         renderView.AxesGrid.YLabelFontFamily = font
         renderView.AxesGrid.ZLabelFontFamily = font
      else:
         renderView.AxesGrid.XTitleFontFamily = 'File'
         renderView.AxesGrid.YTitleFontFamily = 'File'
         renderView.AxesGrid.ZTitleFontFamily = 'File'
         
         renderView.AxesGrid.XLabelFontFamily = 'File'
         renderView.AxesGrid.YLabelFontFamily = 'File'
         renderView.AxesGrid.ZLabelFontFamily = 'File'
         
         renderView.AxesGrid.XTitleFontFile = font
         renderView.AxesGrid.YTitleFontFile = font
         renderView.AxesGrid.ZTitleFontFile = font
         
         renderView.AxesGrid.XLabelFontFile = font
         renderView.AxesGrid.YLabelFontFile = font
         renderView.AxesGrid.ZLabelFontFile = font
         
      renderView.AxesGrid.XTitleFontSize = size
      renderView.AxesGrid.YTitleFontSize = size
      renderView.AxesGrid.ZTitleFontSize = size
      
      renderView.AxesGrid.XLabelFontSize = size
      renderView.AxesGrid.YLabelFontSize = size
      renderView.AxesGrid.ZLabelFontSize = size
      
   def set_bold(self, renderView, bold_flag: Optional[int] = None):
      """
      @brief Make axis titles and labels bold if bold_flag = 1.
      @param renderView  The ParaView render view object.
      @param bold_flag   1 for bold, 0 otherwise.
      """
      if bold_flag is None:
         bold_flag = self.bold
         
      renderView.AxesGrid.XTitleBold = bold_flag
      renderView.AxesGrid.YTitleBold = bold_flag
      renderView.AxesGrid.ZTitleBold = bold_flag

      renderView.AxesGrid.XLabelBold = bold_flag
      renderView.AxesGrid.YLabelBold = bold_flag
      renderView.AxesGrid.ZLabelBold = bold_flag
      

@dataclass
class RenderManager:
   """
   @class   RenderManager
   @brief   Creates and configures ParaView render views, including 2D/3D setups and axes.
   @details Uses the AxesManager internally to label axes and set fonts/colors.
   """
   dim:      int
   x0:       np.ndarray
   x1:       np.ndarray
   units:    str        = "km"
   rx:       int        = 1920
   ry:       int        = 1080
   font:     str        = "Arial"
   fontsize: int        = 30
   bold:     int        = 0
   axes:     Optional[AxesManager] = None
   
   def create_render_view(self):
      """
      @brief Create and configure a ParaView render view (2D or 3D).
      @return The newly created render view.
      """
      renderView = GetActiveViewOrCreate('RenderView')
      renderView.ViewSize = [self.rx, self.ry]
      
      if self.dim == 2:
         self._configure_2d_view(renderView)
      else:
         renderView.InteractionMode = '3D'
         
      self.axes = AxesManager(dim      = self.dim,
                              font     = self.font,
                              fontsize = self.fontsize,
                              units    = self.units)
                         
      # Compute axes labels
      L  = np.max(self.x1 - self.x0)   # Axis range
      nL = int(np.log2(L))             # Axis range as power of 2
      dL = 2**(nL-3)                   # Between 4 and 8 labels

      xlabels = get_labels(self.x0[0], self.x1[0], dL)
      ylabels = get_labels(self.x0[1], self.x1[1], dL)
      zlabels = None
      if self.dim == 3:
         zlabels = get_labels(self.x0[2], self.x1[2], dL)
      self.axes.labels = [xlabels, ylabels, zlabels]
      
      # Apply settings to axes
      self.axes.apply(renderView)

      return renderView

   def _configure_2d_view(self, renderView):
      """
      @brief Configure settings for a 2D view.
      @param renderView The ParaView render view object.
      """
      renderView.InteractionMode = '2D'
      renderView.OrientationAxesZVisibility = 0
      renderView.OrientationAxesYLabelText = 'Z'
      renderView.OrientationAxesVisibility = 0
      renderView.ResetActiveCameraToPositiveZ()
      renderView.AdjustRoll(-180.0)
      
      xc = (self.x1 + self.x0) / 2
      L = self.x1 - self.x0
      l = np.max(L)
      
      renderView.CameraPosition   = [xc[0], xc[1], -2 * l]
      renderView.CameraFocalPoint = [xc[0], xc[1], 0.0]
      renderView.CameraParallelScale = 0.4 * l
      
      # Ensure there is at least one light
      if len(renderView.AdditionalLights) == 0:
         light = AddLight(view = renderView)
         light.Intensity = 2.0


class ParaviewManager:
   """
   @class   ParaviewManager
   @brief   High-level manager for ParaView workflows:
            data reading, rendering, colorbars, sources/receivers, and screenshots.
   """
   def __init__(self, input_file, **kwargs):
      """
      @brief Constructor for ParaviewManager.
      @param input_file  Path to input configuration file.
      @param kwargs      Additional settings (font, fontsize, bold, etc.).
      """
      fs_dir = os.environ["FREQUENSOL_DIR"]
      default_font = os.path.join(fs_dir, "trunk/files/misc/fonts/roboto/Roboto-Condensed.ttf")
      self.font     = kwargs.get("font", default_font)
      self.fontsize = kwargs.get("fontsize", 30)
      self.bold     = kwargs.get("bold", False)
      self.units    = kwargs.get("units", "km")
      bold = 1 if self.bold else 0
      
      # Create colormap manager
      self.colormap_manager = ColorMapManager()
      
      # Create data manager
      self.data_manager = DataManager(input_file)
      
      # Create render manager
      self.render_manager = RenderManager(
                                 dim      = self.data_manager.dim,
                                 x0       = self.data_manager.x0,
                                 x1       = self.data_manager.x1,
                                 rx       = kwargs.get("resolution", [1920, 1080])[0],
                                 ry       = kwargs.get("resolution", [1920, 1080])[1],
                                 units    = self.units,
                                 font     = self.font,
                                 bold     = bold,
                                 fontsize = self.fontsize
                              )
      self.renderView = self.render_manager.create_render_view()
      
      # Create colorbar setter
      self.colorbar_setter = ColorbarSetter(
                                 font     = self.font,
                                 fontsize = self.fontsize,
                                 bold     = bold,
                              )
      
      # Initialize active_field flag
      self.active_field = None

   def show_field(self, field, comp=None, colorbar=False, **kwargs):
      """
      @brief  Display a specific field in the render view, optionally with a colorbar and vector component.
      @param  field     Name of the field to show.
      @param  comp      Vector component ('X','Y','Z','Magnitude') if applicable.
      @param  colorbar  Whether to show a colorbar.
      @param  kwargs    Additional optional arguments:
                        - 'show_pml' (bool)
                        - 'colormap', 'opacity', 'limits', 'title'
      @return display   A handle to the displayed object.
      """
      Hide(self.active_field)
      
      if kwargs.get("show_pml", True):
         self.active_field = self.data_manager.field_XMF
         display = Show(self.data_manager.field_XMF, self.renderView, 'UnstructuredGridRepresentation')
      else:
         display = self._create_clipped_display()
      
      # Set which field to color
      display.ColorArrayName = ['POINTS', field]

      # If vector, configure component
      if comp:
         if comp not in ["X", "Y", "Z", "Magnitude"]:
            raise ValueError(f"Invalid vector component: {comp}")
         ColorBy(display, ('POINTS', field, comp))

      fieldLUT = self._configure_colormap(field, kwargs)
      fieldLUT.AutomaticRescaleRangeMode = 'Never'
      fieldLUT.RescaleOnVisibilityChange = 1
      
      display.LookupTable = fieldLUT
      
      title = kwargs.get("title", field)
      self.colorbar_setter.apply(field      = field,
                                 title      = title,
                                 renderView = self.renderView)
      
      if kwargs.get("type") == "mesh":
         display.Representation = 'Surface With Edges'
         display.LineWidth = 2.0
         display.EdgeColor = [0.0, 0.0, 0.0]
      else:
         display.Representation = 'Surface'
      
      if colorbar:
         display.SetScalarBarVisibility(self.renderView, True)
      else:
         display.SetScalarBarVisibility(self.renderView, False)
         
      display.RescaleTransferFunctionToDataRange(True)
      
      self._configure_colormap(field, kwargs)
      
      display.DisableLighting = 1
      
      return display

   def _create_clipped_display(self):
      """
      @brief Create a clipped display for the field (for hiding PML, etc.).
      @return The Clip object display.
      """
      clip1 = Clip(registrationName='Clip1', Input=self.data_manager.field_XMF)
      clip1.ClipType = 'Box'
      clip1.ClipType.UseReferenceBounds = 1

      if self.data_manager.dim == 2:
         clip1.ClipType.Bounds = [self.data_manager.x0[0], self.data_manager.x1[0],
                                  -0.5, self.data_manager.x1[1],
                                   0.0, 1.0 ]
         clip1.ClipType.Position = [0.0, 0.0, -0.5]
         clip1.ClipType.Length   = [1.0, 1.0,  1.0]
      else:
         clip1.ClipType.Bounds = [self.data_manager.x0[0], self.data_manager.x1[0],
                                  self.data_manager.x0[1], self.data_manager.x1[1],
                                  self.data_manager.x0[2], self.data_manager.x1[2]]
         clip1.ClipType.Position = [0.0, 0.0, 0.0]
         clip1.ClipType.Length   = [1.0, 1.0, 1.0]

      display = Show(clip1, self.renderView, 'UnstructuredGridRepresentation')
      self.active_field = clip1

      return display

   def _configure_colormap(self, field, kwargs):
      """
      @brief Configure color map for the specified field using ColorMapManager.
      @param field   Field name.
      @param kwargs  Dictionary that may contain keys like 'colormap', 'opacity', 'limits'.
      @return fieldLUT The resulting LookUpTable after applying color settings.
      """
      fieldLUT = GetColorTransferFunction(field)
      colormap = kwargs.get("colormap", "YGB_r")
      invert   = colormap.endswith("_r")

      self.colormap_manager.apply_colormap(fieldLUT, colormap, invert)
      
      enable_opacity = kwargs.get("opacity", False)
      self.colormap_manager.configure_opacity(fieldLUT, enable_opacity)
      
      self.colormap_manager.rescale_lut(fieldLUT, kwargs.get("limits"))
      return fieldLUT

   def show_axes(self):
      """
      @brief Make the axes grid visible in the current render view.
      """
      self.renderView.AxesGrid.Visibility = 1

   def hide_axes(self):
      """
      @brief Hide the axes grid in the current render view.
      """
      self.renderView.AxesGrid.Visibility = 0

   def show_receivers(self, **kwargs):
      """
      @brief Display receiver markers in the scene, if available.
      @param kwargs Optional settings such as 'size', 'color', 'opacity'.
      @return display The display object if the receivers exist, else None.
      """
      if self.data_manager.receiver_XMF:
         display = Show(self.data_manager.receiver_XMF, self.renderView, 'UnstructuredGridRepresentation')
         
         display.Representation = 'Surface'
         display.RenderPointsAsSpheres = 1
         
         # Set receiver point size
         size = kwargs.get("size", 10.0)
         display.PointSize = size
         
         if "color" in kwargs:
            display.AmbientColor = kwargs.get("color")
            display.DiffuseColor = kwargs.get("color")
         else:
            display.AmbientColor = [0.1, 0.1, 0.9]
            display.DiffuseColor = [0.1, 0.1, 0.9]
            
         opacity = kwargs.get("opacity", 1.0)
         display.Opacity = opacity
         
         return display
      return None
      
   def hide_receivers(self):
      """
      @brief Hide receiver markers if present.
      """
      if self.data_manager.receiver_XMF:
         Hide(self.data_manager.receiver_XMF, self.renderView)
      
   def show_sources(self, **kwargs):
      """
      @brief Display source markers in the scene, if available.
      @param kwargs Optional settings such as 'size', 'color', 'opacity'.
      @return display The display object if the sources exist, else None.
      """
      if self.data_manager.source_XMF:
         display = Show(self.data_manager.source_XMF, self.renderView, 'UnstructuredGridRepresentation')
         display.Representation = 'Surface'
         display.RenderPointsAsSpheres = 1
         
         # Set source point size
         size = kwargs.get("size", 15.0)
         display.PointSize = size
         
         if "color" in kwargs:
            display.AmbientColor = kwargs.get("color")
            display.DiffuseColor = kwargs.get("color")
         else:
            display.AmbientColor = [0.9, 0.1, 0.1]
            display.DiffuseColor = [0.9, 0.1, 0.1]
            
         opacity = kwargs.get("opacity", 1.0)
         display.Opacity = opacity
         
         return display
      return None
      
   def hide_sources(self):
      """
      @brief Hide source markers if present.
      """
      if self.data_manager.source_XMF:
         Hide(self.data_manager.source_XMF, self.renderView)
      
   def screenshot(self, file):
      """
      @brief  Take a screenshot or export the scene to a file.
      @param  file  Path to the output image or vector file (.png, .jpg, .tiff, .svg, .pdf).
      """
      self.renderView.Update()
      if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tiff"):
         SaveScreenshot(file, GetActiveView(), ImageResolution=[self.render_manager.rx, self.render_manager.ry])
      if file.endswith(".svg") or file.endswith(".pdf"):
         ExportView(file, view=self.renderView, Rasterize3Dgeometry=0)


if __name__ == '__main__':

   input_file = sys.argv[1]
   
   # Get output directory
   input = InputParser.read(input_file)
   report_dir = os.path.abspath(input.get_block("Problem").args["directory"])
   
   # Initialize paraview manager
   pv = ParaviewManager(input_file = input_file,
                        fontsize   = 32,
                        bold       = True)
   # Plot wavespeeds
   for prop in ["Vp", "Vs"]:
      pv.show_field(field     = prop,
                    colorbar  = True,
                    title     = f"{prop} (km/s)",
                    colormap  = "YGB_r",
                    show_pml  = False)
                    
      # Plot sources and receivers
      pv.show_sources()
      pv.show_receivers()
      
      # Show axes
      pv.show_axes()
   
   # Hide sources & recievers
   pv.hide_sources()
   pv.hide_receivers()
      
   # Hide axes
   pv.hide_axes()
   
   # Plot Displacement Components
   for comp in ["X", "Y"]:
      pv.show_field(field    = "disp_1_im",
                    comp     = comp,
                    colorbar = False,
                    colormap = "RdGy",
                    limits   = [-5,5])
      pv.screenshot(f"./disp_{comp}.pdf")
      
   # Plot Displacement Amplitude
   pv.show_field(field    = "disp_1_abs",
                 comp     = "Magnitude",
                 colorbar = False,
                 colormap = "RdGy",
                 limits   = [0,5])
   pv.screenshot(f"./disp.pdf")
