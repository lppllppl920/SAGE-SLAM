/**
 * ****************************************************************************
 * Copyright (c) 2015, Robert Lukierski.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * ****************************************************************************
 * Generic templated RANSAC.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_RANSAC_HPP
#define VISIONCORE_MATH_RANSAC_HPP

#include <algorithm>
#include <vector>
#include <limits>
#include <functional>
#include <limits>

namespace vc
{
    
namespace math
{

template<typename t_model, typename t_data>
class RANSAC
{
public:
    typedef t_model ModelT;
    typedef t_data DataContainerT;
    typedef typename DataContainerT::value_type DataValueT;
    
    typedef std::function<double (const ModelT&, std::size_t, const DataContainerT&)> CostFunctionT;
    typedef std::function<ModelT (const std::vector<std::size_t>&, const DataContainerT&)> ModelFunctionT;
    
    RANSAC(ModelFunctionT amf, CostFunctionT acf, const DataContainerT& adata, int mss)
        : mf(amf), cf(acf), data(adata), minimum_set_size(mss)
    {
    }

    ModelT estimate(std::vector<std::size_t>& inliers, unsigned int iterations, double max_datum_fit_error, std::size_t min_consensus_size)
    {
        if(data.size() < minimum_set_size)
        {
            return ModelT();
        }

        ModelT best_model;
        std::vector<std::size_t> best_consensus_set;
        double best_error = std::numeric_limits<double>::max();

        for(unsigned int k = 0; k < iterations; ++k)
        {
            std::vector<std::size_t> maybe_inliers;
            selectRandomCandidates(maybe_inliers, data.size());
            ModelT maybe_model = mf(maybe_inliers, data);
            std::vector<std::size_t> consensus_set(maybe_inliers);

            for(std::size_t i = 0; i < data.size() ; ++i)
            {
                if(find(maybe_inliers.begin(), maybe_inliers.end(), i) == maybe_inliers.end())
                {
                    const double err = cf(maybe_model, i, data);
                    if(err < max_datum_fit_error)
                    {
                        consensus_set.push_back(i);
                    }
                }
            }

            if (consensus_set.size() >= min_consensus_size)
            {
                ModelT better_model = mf(consensus_set, data);
                double this_error = 0.0;
                for(std::size_t i = 0; i < consensus_set.size() ; ++i)
                {
                    double err = cf(better_model, consensus_set[i], data);
                    this_error += err*err;
                }
                
                this_error /= consensus_set.size();
                
                if(this_error < best_error)
                {
                    best_model = better_model;
                    best_consensus_set = consensus_set;
                    best_error = this_error;
                }
            }
        }

        inliers = best_consensus_set;
        return best_model;
    }

protected:
    void selectRandomCandidates(std::vector<std::size_t>& set, std::size_t num_elements)
    {
        while(set.size() < minimum_set_size)
        {
            const std::size_t i = rand() % num_elements;
            if (find(set.begin(), set.end(), i) == set.end())
            {
                set.push_back(i);
            }
        }
    }

    ModelFunctionT mf;
    CostFunctionT cf;
    const DataContainerT& data;
    std::size_t minimum_set_size;
};

}

}

#endif // VISIONCORE_RANSAC_HPP
