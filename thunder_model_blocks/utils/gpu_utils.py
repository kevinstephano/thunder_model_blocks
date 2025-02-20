import subprocess
import sys

def install_pynvml():
    """
    Check if pynvml is installed, and if not, install it using pip.
    Returns True if pandas is successfully installed or already present,
    False if installation fails.
    """
    try:
        import pynvml as nvml
        return nvml
    except ImportError:
        print("pynvml is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-ml-py"])
            print("pynvml has been successfully installed!")
            import pynvml as nvml
            return nvml 
        except subprocess.CalledProcessError:
            print("Failed to install pynvml. Please try installing manually using:")
            print("pip install pynvml")
            return None

nvml = install_pynvml()
assert nvml is not None, "Pynvml is not installed!" 

def lock_max_gpu_clocks():
    """Locks the maximum SM clock of an NVIDIA GPU using pynvml.
    """
    try:
        nvml.nvmlInit()  # Initialize NVML
        num_devices = nvml.nvmlDeviceGetCount()
        
        for gpu_id in range(num_devices):
            handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)  # Get the GPU handle

            is_supported = nvml.nvmlDeviceGetAPIRestriction(handle, nvml.NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS)
            if not is_supported:
                print(f"Warning: GPU {gpu_id} does not support application clock management")
                continue
            max_clock = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)
            nvml.nvmlDeviceSetClock(handle, nvml.NVML_CLOCK_SM, max_clock)
         
            print(f"Successfully locked GPU {gpu_id} SM clock to {max_clock} MHz.")

    except nvml.NVMLError as error:
        #print(f"Error locking GPU clock: {error}")
        raise nvml.NVMLError(f"Error locking GPU clock: {error}")
    finally:
        nvml.nvmlShutdown()  # Shutdown NVML


def reset_gpu_clocks():
    """
    Reset the clock settings to default for specified NVIDIA GPU(s).
    
    Raises:
        NVMLError: If there's an error initializing NVML or accessing GPU information
    """
    try:
        nvml.nvmlInit()
        
        num_devices = nvml.nvmlDeviceGetCount()
        
        for gpu_id in range(num_devices):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Check if applications clocks are supported
                is_supported = nvml.nvmlDeviceGetAPIRestriction(handle, nvml.NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS)
                print()
                if not is_supported:
                    print(f"Warning: GPU {gpu_id} does not support application clock management")
                    continue
                
                # Reset the clocks
                nvml.nvmlDeviceResetApplicationsClocks(handle)
                print(f"Successfully reset clocks for GPU {gpu_id}")
                
            except nvml.NVMLError as e:
                print(f"Error resetting GPU {gpu_id}: {e}")
                continue
    finally:
        # Always clean up
        nvml.nvmlShutdown()
