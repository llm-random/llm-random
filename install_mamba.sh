# Check if nvcc command (CUDA compiler) is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA is not installed"
    exit 1
fi

# Get CUDA version using nvcc
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}')

# Check if CUDA version is at least 11.6
if [[ $(echo "$cuda_version 11.6" | awk '{if ($1 >= $2) print "true"; else print "false"}') == "true" ]]; then
    echo "CUDA is installed and version is at least 11.6 (Found version: $cuda_version)"
else
    echo "CUDA is installed but version is below 11.6 (Found version: $cuda_version)"
fi
