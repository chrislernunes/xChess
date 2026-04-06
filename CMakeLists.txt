cmake_minimum_required(VERSION 3.20)
project(LightningChess VERSION 0.1.0 LANGUAGES CXX)

# ─── Options ──────────────────────────────────────────────────────────────────
option(USE_CUDA   "Enable CUDA GPU acceleration"          ON)
option(USE_TORCH  "Enable LibTorch neural-net evaluation" ON)
option(BUILD_TESTS "Build perft / unit tests"             ON)

# ─── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# ─── CUDA ─────────────────────────────────────────────────────────────────────
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86) # Pascal → Ampere
    add_compile_definitions(USE_CUDA)
    message(STATUS "CUDA enabled — architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA disabled — CPU fallback active")
endif()

# ─── LibTorch ─────────────────────────────────────────────────────────────────
if(USE_TORCH)
    # Set TORCH_DIR via cmake -DTORCH_DIR=/path/to/libtorch
    if(DEFINED TORCH_DIR)
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
        add_compile_definitions(USE_TORCH)
        message(STATUS "LibTorch found: ${TORCH_VERSION}")
    else()
        message(WARNING "LibTorch NOT found — NN evaluation disabled (HCE only)")
        set(USE_TORCH OFF)
    endif()
endif()

# ─── Engine sources ───────────────────────────────────────────────────────────
set(ENGINE_SOURCES
    src/main.cpp
    src/engine.cpp
    src/evaluator.cpp
    src/chess_board.cpp
)

if(USE_CUDA)
    list(APPEND ENGINE_SOURCES src/gpu_nn.cu)
endif()

add_executable(LightningChess ${ENGINE_SOURCES})

target_include_directories(LightningChess PRIVATE src)

# Link LibTorch
if(USE_TORCH AND Torch_FOUND)
    target_link_libraries(LightningChess PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(LightningChess PRIVATE ${TORCH_INCLUDE_DIRS})
    # Required for MSVC + LibTorch
    if(MSVC)
        set_property(TARGET LightningChess PROPERTY
            VS_DEBUGGER_ENVIRONMENT "PATH=${TORCH_INSTALL_PREFIX}/lib;%PATH%")
    endif()
endif()

# Link CUDA runtime
if(USE_CUDA)
    target_link_libraries(LightningChess PRIVATE CUDA::cudart CUDA::cublas)
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_perft tests/test_perft.cpp src/engine.cpp src/evaluator.cpp src/chess_board.cpp)
    target_include_directories(test_perft PRIVATE src)
    if(USE_TORCH AND Torch_FOUND)
        target_link_libraries(test_perft PRIVATE ${TORCH_LIBRARIES})
    endif()
    add_test(NAME perft_startpos COMMAND test_perft)
endif()

# ─── Install ──────────────────────────────────────────────────────────────────
install(TARGETS LightningChess RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "─── LightningChess build configuration ───")
message(STATUS "  CUDA    : ${USE_CUDA}")
message(STATUS "  Torch   : ${USE_TORCH}")
message(STATUS "  Tests   : ${BUILD_TESTS}")
message(STATUS "──────────────────────────────────────────")
