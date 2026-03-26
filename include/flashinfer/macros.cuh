#ifndef FLASHINFER_MARCOS_CUH_
#define FLASHINFER_MARCOS_CUH_

// global features
#if defined(USE_SAIL)
    #define SAIL_ENABLE_FULL_FEATS          1
    #define SAIL_ENABLE_AIU                 1
#else
    #define SAIL_ENABLE_FULL_FEATS          0
    #define SAIL_ENABLE_AIU                 0
#endif

// arch: sm80;sm89
#if SAIL_ENABLE_FULL_FEATS && (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800))
    #define SAIL_ARCH_SM80_FEAT             1
#else
    #define SAIL_ARCH_SM80_FEAT             0
#endif

#if SAIL_ENABLE_AIU && SAIL_ARCH_SM80_FEAT
    #define SAIL_ARCH_SM80_AIU              1
#else
    #define SAIL_ARCH_SM80_AIU              0
#endif

#if SAIL_ENABLE_FULL_FEATS && (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 890))
    #define SAIL_ARCH_SM89_FEAT             1
#else
    #define SAIL_ARCH_SM89_FEAT             0
#endif

// mma compatible
#if SAIL_ENABLE_FULL_FEATS == 0 || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    #define SAIL_MMA_COMPATIBLE             1
#else
    #define SAIL_MMA_COMPATIBLE             0
#endif

#endif // FLASHINFER_MARCOS_CUH_