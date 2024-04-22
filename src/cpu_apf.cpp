#include <pybind11/pybind11.h>
#include <queue>
#include <map>

namespace {

    inline long get_2d(long i, long j, long size_dim1) {
        return j + i * size_dim1;
    }

    struct Point2d {
        long x;
        long y;
    };

    inline float distance_2d(Point2d p1, Point2d p2) {
        return (float) sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }

    struct Candidate {
        Point2d p{};
        Point2d ori{};
        float distance{};

        Candidate(long x, long y, float distance) {
            this->p = {x, y};
            this->ori = {x, y};
            this->distance = distance;
        }

        Candidate(Point2d p, Point2d ori, float distance) {
            this->p = p;
            this->ori = ori;
            this->distance = distance;
        }
    };

    const int directions[4][2] = {
            {0, 1},
            {0, -1},
            {1, 0},
            {-1, 0},
    };

    void cpu_apf(void *out, const void **in) {
        // Parse the inputs
        const std::int64_t size_dim0 = *reinterpret_cast<const std::int64_t *>(in[0]);
        const std::int64_t size_dim1 = *reinterpret_cast<const std::int64_t *>(in[1]);
        const auto *map_weed = reinterpret_cast<const bool *>(in[2]);

        // Parse the outputs
        auto *out_buf = reinterpret_cast<float *>(out);

        std::queue<Candidate> candidates;
        bool visited[size_dim0 * size_dim1];

        for (std::int64_t i = 0; i < size_dim0; ++i) {
            for (std::int64_t j = 0; j < size_dim1; ++j) {
                long aim = get_2d(i, j, size_dim1);
                out_buf[aim] = 0.;
                if (map_weed[aim]) {
                    candidates.emplace(i, j, 0.);
                    visited[aim] = true;
                } else {
                    visited[aim] = false;
                }
//                out_buf[aim] = (float) aim;
            }
        }

        while (not candidates.empty()) {
            auto candidate = candidates.front();
            candidates.pop();
            for (auto &direction: directions) {
                Point2d new_position = {candidate.p.x + direction[0], candidate.p.y + direction[1]};
                long aim = get_2d(new_position.x, new_position.y, size_dim1);
                if (0 <= aim and aim < size_dim0 * size_dim1 and not visited[aim]) {
                    float new_distance = distance_2d(candidate.ori, new_position);
                    candidates.emplace(new_position, candidate.ori, new_distance);
                    visited[aim] = true;
                    out_buf[aim] = new_distance;
                }
            }
        }
    }

    // https://en.cppreference.com/w/cpp/numeric/bit_cast
    template<class To, class From>
    typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value, To>::type
    bit_cast(const From &src) noexcept {
        static_assert(
                std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to be trivially constructible");

        To dst;
        memcpy(&dst, &src, sizeof(To));
        return dst;
    }

    template<typename T>
    pybind11::capsule EncapsulateFunction(T *fn) {
        return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
    }

    pybind11::dict Registrations() {
        pybind11::dict dict;
        dict["cpu_apf_bool"] = EncapsulateFunction(cpu_apf);
        return dict;
    }

    PYBIND11_MODULE(cpu_apf, m
    ) {
    m.def("registrations", &Registrations);
}

}  // namespace