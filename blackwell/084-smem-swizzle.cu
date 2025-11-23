/*
    Observations:

    - If inner dim bytes is less than swizzle bytes, TMA will zero pad the swizzle atom
    - BF16 swizzle atom patterns:
        - Swizzle 32B (8x2 atom; 8x16 for BF16):
              0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15 
             16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31 
             32    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47 
             48    49    50    51    52    53    54    55    56    57    58    59    60    61    62    63 
             72    73    74    75    76    77    78    79    64    65    66    67    68    69    70    71 
             88    89    90    91    92    93    94    95    80    81    82    83    84    85    86    87 
            104   105   106   107   108   109   110   111    96    97    98    99   100   101   102   103 
            120   121   122   123   124   125   126   127   112   113   114   115   116   117   118   119
        - Swizzle 64B (8x4 atom; 8x32 for BF16):
              0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31 
             32    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47    48    49    50    51    52    53    54    55    56    57    58    59    60    61    62    63 
             72    73    74    75    76    77    78    79    64    65    66    67    68    69    70    71    88    89    90    91    92    93    94    95    80    81    82    83    84    85    86    87 
            104   105   106   107   108   109   110   111    96    97    98    99   100   101   102   103   120   121   122   123   124   125   126   127   112   113   114   115   116   117   118   119 
            144   145   146   147   148   149   150   151   152   153   154   155   156   157   158   159   128   129   130   131   132   133   134   135   136   137   138   139   140   141   142   143 
            176   177   178   179   180   181   182   183   184   185   186   187   188   189   190   191   160   161   162   163   164   165   166   167   168   169   170   171   172   173   174   175 
            216   217   218   219   220   221   222   223   208   209   210   211   212   213   214   215   200   201   202   203   204   205   206   207   192   193   194   195   196   197   198   199 
            248   249   250   251   252   253   254   255   240   241   242   243   244   245   246   247   232   233   234   235   236   237   238   239   224   225   226   227   228   229   230   231 
        - Swizzle 128B (8x8 atom; 8x64 for BF16):
              0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47    48    49    50    51    52    53    54    55    56    57    58    59    60    61    62    63 
             72    73    74    75    76    77    78    79    64    65    66    67    68    69    70    71    88    89    90    91    92    93    94    95    80    81    82    83    84    85    86    87   104   105   106   107   108   109   110   111    96    97    98    99   100   101   102   103   120   121   122   123   124   125   126   127   112   113   114   115   116   117   118   119 
            144   145   146   147   148   149   150   151   152   153   154   155   156   157   158   159   128   129   130   131   132   133   134   135   136   137   138   139   140   141   142   143   176   177   178   179   180   181   182   183   184   185   186   187   188   189   190   191   160   161   162   163   164   165   166   167   168   169   170   171   172   173   174   175 
            216   217   218   219   220   221   222   223   208   209   210   211   212   213   214   215   200   201   202   203   204   205   206   207   192   193   194   195   196   197   198   199   248   249   250   251   252   253   254   255   240   241   242   243   244   245   246   247   232   233   234   235   236   237   238   239   224   225   226   227   228   229   230   231 
            288   289   290   291   292   293   294   295   296   297   298   299   300   301   302   303   304   305   306   307   308   309   310   311   312   313   314   315   316   317   318   319   256   257   258   259   260   261   262   263   264   265   266   267   268   269   270   271   272   273   274   275   276   277   278   279   280   281   282   283   284   285   286   287 
            360   361   362   363   364   365   366   367   352   353   354   355   356   357   358   359   376   377   378   379   380   381   382   383   368   369   370   371   372   373   374   375   328   329   330   331   332   333   334   335   320   321   322   323   324   325   326   327   344   345   346   347   348   349   350   351   336   337   338   339   340   341   342   343 
            432   433   434   435   436   437   438   439   440   441   442   443   444   445   446   447   416   417   418   419   420   421   422   423   424   425   426   427   428   429   430   431   400   401   402   403   404   405   406   407   408   409   410   411   412   413   414   415   384   385   386   387   388   389   390   391   392   393   394   395   396   397   398   399 
            504   505   506   507   508   509   510   511   496   497   498   499   500   501   502   503   488   489   490   491   492   493   494   495   480   481   482   483   484   485   486   487   472   473   474   475   476   477   478   479   464   465   466   467   468   469   470   471   456   457   458   459   460   461   462   463   448   449   450   451   452   453   454   455 
*/

#include "kittens.cuh"

using namespace kittens;

using DTYPE = bf16;
static constexpr int M = 8;
static constexpr int N = 64;
static constexpr int TILE_M = 8;
static constexpr int TILE_N = 64;

__global__ void kernel(const __grid_constant__ CUtensorMap tmap) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    DTYPE *smem = reinterpret_cast<DTYPE*>(((__shm_base + 1023) / 1024) * 1024);

    // Initialize mbarriers
    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    // Load
    const int row = TILE_M * 0;
    const int col = TILE_N * 0;
    tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(DTYPE));
    asm volatile("{cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];}"
        :: "l"(__cvta_generic_to_shared(smem)), "l"(&tmap), "r"(row), "r"(col), "l"(__cvta_generic_to_shared(&inputs_arrived))
        : "memory");
    wait(inputs_arrived, 0);

    // Inspect
    #pragma unroll
    for (int i = 0; i < TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            printf("%5d ", std::bit_cast<uint16_t>(smem[i * TILE_N + j]));
        }
        printf("\n");
    }
}

int main(void) {
    // Allocate host memory
    DTYPE *data_host = new DTYPE[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) 
        data_host[i] = std::bit_cast<DTYPE>(uint16_t(i));
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    DTYPE *data_device;
    CUDACHECK(cudaMalloc(&data_device, M * N * sizeof(DTYPE)));
    std::cout << "Allocated device memory" << std::endl;

    // Copy to device memory
    CUDACHECK(cudaMemcpy(data_device, data_host, M * N * sizeof(DTYPE), cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Generate tensor descriptor
    CUtensorMap tmap;
    static constexpr int rank = 2;
    uint64_t gmem_shape [2] = {N, M}; // inner-dim first!
    uint64_t gmem_stride[1] = {N * sizeof(bf16)};
    uint32_t smem_shape [2] = {TILE_N, TILE_M};
    uint32_t smem_stride[2] = {1, 1};
    CUCHECK(cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)data_device,
        &gmem_shape[0],
        &gmem_stride[0],
        &smem_shape[0],
        &smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,

        // --------------------------------------
        // SWIZZLE_32B requires the SMEM inner dimension to be <= 32 bytes
        // SWIZZLE_64B requires the SMEM inner dimension to be <= 64 bytes
        // SWIZZLE_128B* require the SMEM inner dimension to be <= 128 bytes
        // --------------------------------------
        // CU_TENSOR_MAP_SWIZZLE_NONE,
        // CU_TENSOR_MAP_SWIZZLE_32B,
        // CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_SWIZZLE_128B,
        // --------------------------------------

        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    // Set dynamic SMEM
    constexpr size_t DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY));

    // Launch kernel
    kernel<<<1, 1, DYNAMIC_SHARED_MEMORY, 0>>>(tmap);
    CUDACHECK(cudaDeviceSynchronize());

    return 0;
}
