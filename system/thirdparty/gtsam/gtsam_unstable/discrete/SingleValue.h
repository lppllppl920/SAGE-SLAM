/*
 * SingleValue.h
 * @brief domain constraint
 * @date Feb 6, 2012
 * @author Frank Dellaert
 */

#pragma once

#include <gtsam_unstable/discrete/Constraint.h>
#include <gtsam/discrete/DiscreteKey.h>

namespace gtsam {

  /**
   * SingleValue constraint
   */
  class GTSAM_UNSTABLE_EXPORT SingleValue: public Constraint {

    /// Number of values
    size_t cardinality_;

    /// allowed value
    size_t value_;

    DiscreteKey discreteKey() const {
      return DiscreteKey(keys_[0],cardinality_);
    }

  public:

    typedef boost::shared_ptr<SingleValue> shared_ptr;

    /// Constructor
    SingleValue(Key key, size_t n, size_t value) :
      Constraint(key), cardinality_(n), value_(value) {
    }

    /// Constructor
    SingleValue(const DiscreteKey& dkey, size_t value) :
      Constraint(dkey.first), cardinality_(dkey.second), value_(value) {
    }

    // print
    virtual void print(const std::string& s = "",
        const KeyFormatter& formatter = DefaultKeyFormatter) const;

    /// equals
    bool equals(const DiscreteFactor& other, double tol) const {
      if(!dynamic_cast<const SingleValue*>(&other))
        return false;
      else {
        const SingleValue& f(static_cast<const SingleValue&>(other));
        return (cardinality_==f.cardinality_) && (value_==f.value_);
      }
    }

    /// Calculate value
    virtual double operator()(const Values& values) const;

    /// Convert into a decisiontree
    virtual DecisionTreeFactor toDecisionTreeFactor() const;

    /// Multiply into a decisiontree
    virtual DecisionTreeFactor operator*(const DecisionTreeFactor& f) const;

    /*
     * Ensure Arc-consistency
     * @param j domain to be checked
     * @param domains all other domains
     */
    bool ensureArcConsistency(size_t j, std::vector<Domain>& domains) const;

    /// Partially apply known values
    virtual Constraint::shared_ptr partiallyApply(
        const Values& values) const;

    /// Partially apply known values, domain version
    virtual Constraint::shared_ptr partiallyApply(
        const std::vector<Domain>& domains) const;
  };

} // namespace gtsam
