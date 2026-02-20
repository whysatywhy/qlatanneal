#pragma once

#include <cstddef>

#include "qanneal/sqa_state.hpp"

namespace qanneal {

class SQAObserver {
public:
    virtual ~SQAObserver() = default;
    virtual void record(std::size_t step,
                        double beta,
                        double gamma,
                        double avg_energy,
                        const SQAState &state) = 0;
};

}
