/*
    Trying to understand 8-bit matrix swizzling better

    Observations:
    - As always, swizzling is only done intra-row, never inter-row
    - TMA load & TMA store do the swizzling identically, and in a way that
        store followed by a load will restore the original matrix
    - Swizzling pattern repeats every 16 rows
    - For 16x128, swizzling pattern is as follows:

  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
 16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
 32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
 48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
 64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
 80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
 96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15

*/

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <cstdint>

using namespace kittens;

// Kernel globals
struct globals {
    using tile = st_fp8e8m0<128, 128>;

    gl<fp8e8m0, 1, 1, 128, 128, tile> inputs;
    gl<fp8e8m0, 1, 1, 128, 128, tile> loaded;
    gl<fp8e8m0, 1, 1, 128, 128, tile> stored;

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(1); } // use single thread
    __host__ inline int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 1024; }
};

// Kernel implementation
__global__ void kernel(const __grid_constant__ globals G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::tile &tile = allocator.allocate<globals::tile>();

    // Set up mbarrier
    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    // Load inputs
    tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
    tma::load_async(tile, G.inputs, {0, 0}, inputs_arrived);

    // Wait for inputs to be loaded
    wait(inputs_arrived, 0);

    // Store back to global memory
    for (int i = 0; i < 128 * 128; i++) {
        G.loaded.raw_ptr[i] = tile.data[i];
    }

    // Replace with continous data
    for (int i = 0; i < 128 * 128; i++) {
        reinterpret_cast<uint8_t &>(tile.data[i]) = static_cast<uint8_t>(i % 128);
    }

    // Store back to global memory
    tma::store_async(G.stored, tile, {0, 0});
    tma::store_async_wait();
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::inputs,
        &globals::loaded,
        &globals::stored
    );
}
