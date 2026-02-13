#!/bin/bash

# Modern color palette
CYAN='\033[0;36m'
BRIGHT_BLUE='\033[1;34m'
BRIGHT_GREEN='\033[1;32m'
BRIGHT_YELLOW='\033[1;33m'
BRIGHT_RED='\033[1;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Format Rust files
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}Formatting Rust files...${NC}"
cargo fmt --all

# Format excluded crates (cuda_kernels, metal_kernels)
if [ -d "crates/hodu_cuda_kernels" ]; then
    echo -e "  ${CYAN}→${NC} ${DIM}crates/hodu_cuda_kernels${NC}"
    (cd crates/hodu_cuda_kernels && cargo fmt)
fi

if [ -d "crates/hodu_metal_kernels" ]; then
    echo -e "  ${CYAN}→${NC} ${DIM}crates/hodu_metal_kernels${NC}"
    (cd crates/hodu_metal_kernels && cargo fmt)
fi

# Format benchmark crates
if [ -d "benchmarks" ]; then
    for cargo_toml in benchmarks/*/Cargo.toml; do
        if [ -f "$cargo_toml" ]; then
            bench_dir=$(dirname "$cargo_toml")
            echo -e "  ${CYAN}→${NC} ${DIM}${bench_dir}${NC}"
            (cd "$bench_dir" && cargo fmt --all)
        fi
    done
fi

echo -e "${BRIGHT_GREEN}✓${NC} Rust formatting complete\n"

# Format C/C++/CUDA/Metal files
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}Formatting C/C++/CUDA/Metal files...${NC}"

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} clang-format is not installed"
    echo -e "${DIM}   Install with: ${MAGENTA}brew install clang-format${NC}\n"
else
    # Counter for formatted files
    count=0

    # Find and format all C/C++/CUDA/Metal files recursively
    while IFS= read -r -d '' file
    do
        echo -e "  ${CYAN}→${NC} ${DIM}$file${NC}"
        clang-format -i "$file"
        ((count++))
    done < <(find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.metal" \) -not -path "*/target/*" -not -path "*/.*" -not -path "*/libs/*" -print0)

    if [ $count -eq 0 ]; then
        echo -e "${DIM}  No C/C++/CUDA/Metal files found${NC}"
    else
        echo -e "${BRIGHT_GREEN}✓${NC} Formatted ${BOLD}${count}${NC} C/C++/CUDA/Metal file(s)"
    fi
fi

echo ""

# Format Python files
echo -e "\n${BRIGHT_BLUE}▶${NC} ${BOLD}Formatting Python files...${NC}"

# Check if ruff is installed
if ! command -v ruff &> /dev/null
then
    echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} ruff is not installed"
    echo -e "${DIM}   Install with: ${MAGENTA}pip install ruff${NC}\n"
else
    # Counter for formatted files
    py_count=0

    # Find all .py files recursively from current directory
    while IFS= read -r -d '' file
    do
        echo -e "  ${CYAN}→${NC} ${DIM}$file${NC}"
        ruff format "$file"
        ((py_count++))
    done < <(find . -type f -name "*.py" -not -path "*/.*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -not -path "*/venvs/*" -not -path "*/target/*" -not -path "*/libs/*" -print0)

    if [ $py_count -eq 0 ]; then
        echo -e "${DIM}  No .py files found${NC}"
    else
        echo -e "${BRIGHT_GREEN}✓${NC} Formatted ${BOLD}${py_count}${NC} Python file(s)"
    fi
fi

echo -e "\n${BOLD}${BRIGHT_GREEN}All formatting complete!${NC}"
