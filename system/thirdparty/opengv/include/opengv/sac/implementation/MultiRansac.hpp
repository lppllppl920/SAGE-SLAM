/******************************************************************************
 * Authors:  Laurent Kneip & Paul Furgale                                     *
 * Contact:  kneip.laurent@gmail.com                                          *
 * License:  Copyright (c) 2013 Laurent Kneip, ANU. All rights reserved.      *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 * * Redistributions of source code must retain the above copyright           *
 *   notice, this list of conditions and the following disclaimer.            *
 * * Redistributions in binary form must reproduce the above copyright        *
 *   notice, this list of conditions and the following disclaimer in the      *
 *   documentation and/or other materials provided with the distribution.     *
 * * Neither the name of ANU nor the names of its contributors may be         *
 *   used to endorse or promote products derived from this software without   *
 *   specific prior written permission.                                       *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"*
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE  *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE *
 * ARE DISCLAIMED. IN NO EVENT SHALL ANU OR THE CONTRIBUTORS BE LIABLE        *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL *
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT         *
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY  *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF     *
 * SUCH DAMAGE.                                                               *
 ******************************************************************************/

//Note: has been derived from ROS

template<typename P>
opengv::sac::MultiRansac<P>::MultiRansac(
    int maxIterations, double threshold, double probability) :
    MultiSampleConsensus<P>(maxIterations, threshold, probability)
{}

template<typename P>
opengv::sac::MultiRansac<P>::~MultiRansac()
{}

template<typename PROBLEM_T>
bool
opengv::sac::MultiRansac<PROBLEM_T>::computeModel(
    int debug_verbosity_level )
{
  typedef PROBLEM_T problem_t;
  typedef typename problem_t::model_t model_t;

  iterations_ = 0;
  int n_best_inliers_count = -INT_MAX;
  double k = 1.0;

  std::vector<std::vector<int> > selection;
  model_t model_coefficients;

  int n_inliers_count = 0;
  unsigned skipped_count = 0;
  // supress infinite loops by just allowing 10 x maximum allowed iterations for
  // invalid model parameters!
  const unsigned max_skip = max_iterations_ * 10;

  // Iterate
  while( iterations_ < k && skipped_count < max_skip )
  {
    // Get X samples which satisfy the model criteria
    sac_model_->getSamples( iterations_, selection );

    if( selection.empty() )
    {
      fprintf( stderr,
          "[sm::RandomSampleConsensus::computeModel] No samples could be selected!\n" );
      break;
    }

    // Search for inliers in the point cloud for the current plane model M
    if(!sac_model_->computeModelCoefficients( selection, model_coefficients ))
    {
      //++iterations_;
      ++ skipped_count;
      continue;
    }

    // Select the inliers that are within threshold_ from the model
    //sac_model_->selectWithinDistance(model_coefficients, threshold_, inliers);
    //if(inliers.empty() && k > 1.0)
    //  continue;

    n_inliers_count = sac_model_->countWithinDistance( model_coefficients, threshold_ );

    // Better match ?
    if( n_inliers_count > n_best_inliers_count )
    {
      n_best_inliers_count = n_inliers_count;

      // Save the current model/inlier/coefficients selection as being the best so far
      model_              = selection;
      model_coefficients_ = model_coefficients;

      //MultiRansac preparation for probability computation
      std::shared_ptr<std::vector< std::vector<int> > > multiIndices =
          sac_model_->getIndices();
      size_t multiIndicesNumber = 0;
      for( size_t multiIter = 0; multiIter < multiIndices->size(); multiIter++ )
        multiIndicesNumber += multiIndices->at(multiIter).size();

      size_t multiSelectionSize = 0;
      for( size_t multiIter = 0; multiIter < selection.size(); multiIter++ )
        multiSelectionSize += selection[multiIter].size();

      // Compute the k parameter (k=log(z)/log(1-w^n))
      double w = static_cast<double> (n_best_inliers_count) /
          static_cast<double> (multiIndicesNumber);
      double p_no_outliers = 1.0 - pow(w, static_cast<double> (multiSelectionSize));
      p_no_outliers =
          (std::max) (std::numeric_limits<double>::epsilon(), p_no_outliers);
          // Avoid division by -Inf
      p_no_outliers =
          (std::min) (1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers);
          // Avoid division by 0.
      k = log(1.0 - probability_) / log(p_no_outliers);
    }

    ++iterations_;
    if(debug_verbosity_level > 1)
      fprintf(stdout,
          "[sm::RandomSampleConsensus::computeModel] Trial %d out of %f: %d inliers (best is: %d so far).\n",
          iterations_, k, n_inliers_count, n_best_inliers_count );
    if(iterations_ > max_iterations_)
    {
      if(debug_verbosity_level > 0)
        fprintf(stdout,
            "[sm::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
      break;
    }
  }

  if(debug_verbosity_level > 0)
  {
    size_t multiModelSize = 0;
    for( size_t modelIter = 0; modelIter < model_.size(); modelIter++ )
      multiModelSize += model_[modelIter].size();
    fprintf(stdout,
        "[sm::RandomSampleConsensus::computeModel] Model: %zu size, %d inliers.\n",
        multiModelSize, n_best_inliers_count );
  }

  if(model_.empty())
  {
    inliers_.clear();
    return (false);
  }

  // Get the set of inliers that correspond to the best model found so far
  sac_model_->selectWithinDistance( model_coefficients_, threshold_, inliers_ );

  return (true);
}
