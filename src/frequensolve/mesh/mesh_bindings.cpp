#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mesh.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_mesh, m) {
   // Bind the vertex class
   py::class_<vertex>(m, "Vertex")
      .def(py::init<const std::vector<double>&>())
      .def("get_coords", &vertex::getCoords)
      .def("set_coords", &vertex::setCoords);

   // Bind the base element class
   py::class_<elem>(m, "Element")
      .def("is_active", &elem::isActive)
      .def("set_active", &elem::setActive)
      .def("domain", &elem::domain)
      .def("set_domain", &elem::setDomain)
      .def("indices", &elem::indices)
      .def("kind", &elem::kind)
      .def("is_inverted", &elem::isInverted);

   // Bind the mesh class
   py::class_<mesh>(m, "Mesh")
      .def(py::init<int>())
      .def("add_vertex", [](mesh& m, py::array_t<double> coords) {
         auto r = coords.unchecked<1>();
         std::vector<double> c(r.data(0), r.data(0) + r.shape(0));
         m.add_vertex(c);
      })
      .def("add_element", [](mesh& m, const std::string& type, 
                            const std::vector<int>& vertices, int domain) {
         if (type == "edge")
            m.add_element<edge>(vertices, domain);
         else if (type == "triangle") 
            m.add_element<triangle>(vertices, domain);
         else if (type == "quad")
            m.add_element<quad>(vertices, domain);
         // Add other element types as needed
      })
      .def("remove_inverted_elements", &mesh::remove_inverted_elements)
      .def("remove_unused_vertices", &mesh::remove_unused_vertices)
      .def("write", &mesh::write)
      .def("get_vertices", [](const mesh& m) {
         return m.getVertices();
      })
      .def("get_elements", [](const mesh& m) {
         return m.getElements();
      });
} 