/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Constructor.ccp
 * @author Frank Dellaert
 * @author Andrew Melim
 * @author Richard Roberts
 **/

#include <iostream>
#include <fstream>
#include <algorithm>

#include <boost/lexical_cast.hpp>

#include "utilities.h"
#include "Constructor.h"
#include "Class.h"

using namespace std;
using namespace wrap;

/* ************************************************************************* */
string Constructor::matlab_wrapper_name(Str className) const {
  string str = "new_" + className;
  return str;
}

/* ************************************************************************* */
void Constructor::proxy_fragment(FileWriter& file,
                                 const std::string& wrapperName, bool hasParent,
                                 const int id, const ArgumentList args) const {
  size_t nrArgs = args.size();
  // check for number of arguments...
  file.oss << "      elseif nargin == " << nrArgs;
  if (nrArgs > 0) file.oss << " && ";
  // ...and their types
  bool first = true;
  for (size_t i = 0; i < nrArgs; i++) {
    if (!first) file.oss << " && ";
    file.oss << "isa(varargin{" << i + 1 << "},'" << args[i].matlabClass(".")
             << "')";
    first = false;
  }
  // emit code for calling constructor
  if (hasParent)
    file.oss << "\n        [ my_ptr, base_ptr ] = ";
  else
    file.oss << "\n        my_ptr = ";
  file.oss << wrapperName << "(" << id;
  // emit constructor arguments
  for (size_t i = 0; i < nrArgs; i++) {
    file.oss << ", ";
    file.oss << "varargin{" << i + 1 << "}";
  }
  file.oss << ");\n";
}

/* ************************************************************************* */
string Constructor::wrapper_fragment(FileWriter& file, Str cppClassName,
                                     Str matlabUniqueName,
                                     boost::optional<string> cppBaseClassName,
                                     int id, const ArgumentList& al) const {
  const string wrapFunctionName =
      matlabUniqueName + "_constructor_" + boost::lexical_cast<string>(id);

  file.oss << "void " << wrapFunctionName
           << "(int nargout, mxArray *out[], int nargin, const mxArray *in[])"
           << endl;
  file.oss << "{\n";
  file.oss << "  mexAtExit(&_deleteAllObjects);\n";
  // Typedef boost::shared_ptr
  file.oss << "  typedef boost::shared_ptr<" << cppClassName << "> Shared;\n";
  file.oss << "\n";

  // Check to see if there will be any arguments and remove {} for consiseness
  if (al.size() > 0) al.matlab_unwrap(file);  // unwrap arguments
  file.oss << "  Shared *self = new Shared(new " << cppClassName << "("
           << al.names() << "));" << endl;
  file.oss << "  collector_" << matlabUniqueName << ".insert(self);\n";

  if (verbose_)
    file.oss << "  std::cout << \"constructed \" << self << std::endl;" << endl;
  file.oss
      << "  out[0] = mxCreateNumericMatrix(1, 1, mxUINT32OR64_CLASS, mxREAL);"
      << endl;
  file.oss << "  *reinterpret_cast<Shared**> (mxGetData(out[0])) = self;"
           << endl;

  // If we have a base class, return the base class pointer (MATLAB will call
  // the base class collectorInsertAndMakeBase to add this to the collector and
  // recurse the heirarchy)
  if (cppBaseClassName) {
    file.oss << "\n";
    file.oss << "  typedef boost::shared_ptr<" << *cppBaseClassName
             << "> SharedBase;\n";
    file.oss << "  out[1] = mxCreateNumericMatrix(1, 1, mxUINT32OR64_CLASS, "
                "mxREAL);\n";
    file.oss << "  *reinterpret_cast<SharedBase**>(mxGetData(out[1])) = new "
                "SharedBase(*self);\n";
  }

  file.oss << "}" << endl;

  return wrapFunctionName;
}

/* ************************************************************************* */
void Constructor::python_wrapper(FileWriter& wrapperFile, Str className) const {
  wrapperFile.oss << "  .def(\"" << name_ << "\", &" << className
                  << "::" << name_ << ");\n";
}

/* ************************************************************************* */
bool Constructor::hasDefaultConstructor() const {
  for (size_t i = 0; i < nrOverloads(); i++) {
    if (argumentList(i).size() == 0) return true;
  }
  return false;
}

/* ************************************************************************* */
void Constructor::emit_cython_pxd(FileWriter& pxdFile, const Class& cls) const {
  for (size_t i = 0; i < nrOverloads(); i++) {
    ArgumentList args = argumentList(i);

    // generate the constructor
    pxdFile.oss << "        " << cls.pxdClassName() << "(";
    args.emit_cython_pxd(pxdFile, cls.pxdClassName(), cls.templateArgs);
    pxdFile.oss << ") " << "except +\n";
  }
}

/* ************************************************************************* */
void Constructor::emit_cython_pyx(FileWriter& pyxFile, const Class& cls) const {
  for (size_t i = 0; i < nrOverloads(); i++) {
    ArgumentList args = argumentList(i);
    pyxFile.oss << "        try:\n";
    pyxFile.oss << pyx_resolveOverloadParams(args, true, 3);
    pyxFile.oss
        << argumentList(i).pyx_convertEigenTypeAndStorageOrder("            ");

    pyxFile.oss << "            self." << cls.shared_pxd_obj_in_pyx() << " = "
        << cls.shared_pxd_class_in_pyx() << "(new " << cls.pxd_class_in_pyx()
        << "(" << args.pyx_asParams() << "))\n";
    pyxFile.oss << "        except (AssertionError, ValueError):\n";
    pyxFile.oss << "            pass\n";
  }
}

/* ************************************************************************* */
