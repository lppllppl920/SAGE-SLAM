/**
 * @file ReturnValue.h
 *
 * @brief Encapsulates a return value from a method
 * @date Dec 1, 2011
 * @author Alex Cunningham
 * @author Richard Roberts
 */

#include "ReturnType.h"
#include "TemplateSubstitution.h"
#include "FileWriter.h"
#include "TypeAttributesTable.h"
#include "utilities.h"

#pragma once

namespace wrap {

/**
 * Encapsulates return type of a method or function, possibly a pair
 */
struct ReturnValue {

  bool isPair;
  ReturnType type1, type2;

  friend struct ReturnValueGrammar;

  /// Default constructor
  ReturnValue() :
      isPair(false) {
  }

  /// Construct from type
  ReturnValue(const ReturnType& type) :
      isPair(false), type1(type) {
  }

  /// Construct from pair type arguments
  ReturnValue(const ReturnType& t1, const ReturnType& t2) :
      isPair(true), type1(t1), type2(t2) {
  }

  /// Destructor
  virtual ~ReturnValue() {}

  virtual void clear() {
    type1.clear();
    type2.clear();
    isPair = false;
  }

  bool isVoid() const {
    return !isPair && !type1.isPtr && (type1.name() == "void");
  }

  bool operator==(const ReturnValue& other) const {
    return isPair == other.isPair && type1 == other.type1
        && type2 == other.type2;
  }

  /// Substitute template argument
  ReturnValue expandTemplate(const TemplateSubstitution& ts) const;

  std::string returnType() const;

  std::string matlab_returnType() const;

  void wrap_result(const std::string& result, FileWriter& wrapperFile,
      const TypeAttributesTable& typeAttributes) const;

  void emit_matlab(FileWriter& proxyFile) const;

  /// @param className the actual class name to use when "This" is specified
  void emit_cython_pxd(FileWriter& file, const std::string& className,
                       const std::vector<std::string>& templateArgs) const;
  std::string pyx_returnType() const;
  std::string pyx_casting(const std::string& var) const;

  friend std::ostream& operator<<(std::ostream& os, const ReturnValue& r) {
    if (!r.isPair && r.type1.category == ReturnType::VOID)
      os << "void";
    else
      os << r.returnType();
    return os;
  }

};

//******************************************************************************
// http://boost-spirit.com/distrib/spirit_1_8_2/libs/spirit/doc/grammar.html
struct ReturnValueGrammar: public classic::grammar<ReturnValueGrammar> {

  wrap::ReturnValue& result_; ///< successful parse will be placed in here
  ReturnTypeGrammar returnType1_g, returnType2_g; ///< Type parsers

  /// Construct type grammar and specify where result is placed
  ReturnValueGrammar(wrap::ReturnValue& result) :
      result_(result), returnType1_g(result.type1), returnType2_g(result.type2) {
  }

  /// Definition of type grammar
  template<typename ScannerT>
  struct definition {

    classic::rule<ScannerT> pair_p, returnValue_p;

    definition(ReturnValueGrammar const& self) {
      using namespace classic;

      pair_p = (str_p("pair") >> '<' >> self.returnType1_g >> ','
          >> self.returnType2_g >> '>')[assign_a(self.result_.isPair, T)];

      returnValue_p = pair_p | self.returnType1_g;
    }

    classic::rule<ScannerT> const& start() const {
      return returnValue_p;
    }

  };
};
// ReturnValueGrammar

}// \namespace wrap
