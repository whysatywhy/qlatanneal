#pragma once

#include <vector>

#include "qanneal/metrics.hpp"
#include "qanneal/observer.hpp"
#include "qanneal/sqa_observer.hpp"

namespace qanneal {

class MetricsObserver : public Observer {
public:
    std::vector<double> energy_trace;
    std::vector<double> magnetization_trace;

    void record(std::size_t step,
                double beta,
                double energy,
                const State &state) override {
        (void)step;
        (void)beta;
        energy_trace.push_back(energy);
        magnetization_trace.push_back(magnetization(state));
    }

    void clear() {
        energy_trace.clear();
        magnetization_trace.clear();
    }
};

class SQAMetricsObserver : public SQAObserver {
public:
    std::vector<double> energy_trace;
    std::vector<double> magnetization_trace;

    void record(std::size_t step,
                double beta,
                double gamma,
                double avg_energy,
                const SQAState &state) override {
        (void)step;
        (void)beta;
        (void)gamma;
        energy_trace.push_back(avg_energy);

        const std::size_t replicas = state.replicas();
        const std::size_t slices = state.slices();
        const std::size_t spins = state.spins();
        double sum = 0.0;
        for (std::size_t r = 0; r < replicas; ++r) {
            for (std::size_t t = 0; t < slices; ++t) {
                const int8_t *ptr = state.slice_ptr(r, t);
                for (std::size_t i = 0; i < spins; ++i) {
                    sum += static_cast<double>(ptr[i]);
                }
            }
        }
        const double denom = static_cast<double>(replicas * slices * spins);
        magnetization_trace.push_back(denom > 0.0 ? sum / denom : 0.0);
    }

    void clear() {
        energy_trace.clear();
        magnetization_trace.clear();
    }
};

}
