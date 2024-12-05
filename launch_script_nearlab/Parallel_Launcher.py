import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import signal
import atexit

class ParallelLauncher:
    def __init__(self):
        self.processes = []
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Register cleanup handler
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def launch_processes(self, script_path: str, num_processes: int, args_per_process=None):
        """
        Launch multiple instances of a Python script in parallel
        
        Args:
            script_path: Path to the Python script to run
            num_processes: Number of processes to launch
            args_per_process: List of lists containing arguments for each process
        """
        print(f"Launching {num_processes} instances of {script_path}")
        
        for i in range(num_processes):
            # Create unique log file for each process
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"process_{i}_{timestamp}.log"
            
            # Get process-specific arguments if provided
            process_args = []
            if args_per_process and i < len(args_per_process):
                process_args = args_per_process[i]
            
            # Construct command with any process-specific arguments
            cmd = [sys.executable, script_path] + process_args
            
            try:
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    self.processes.append(process)
                    print(f"Started process {i} with PID: {process.pid}")
            except Exception as e:
                print(f"Error launching process {i}: {e}")

    def wait_for_completion(self):
        """Wait for all processes to complete and show their status"""
        while self.processes:
            for process in self.processes[:]:  # Create a copy of the list to modify it
                if process.poll() is not None:  # Process has finished
                    print(f"Process (PID: {process.pid}) completed with return code: {process.returncode}")
                    self.processes.remove(process)
            time.sleep(1)
        print("All processes completed")

    def cleanup(self):
        """Terminate all running processes"""
        for process in self.processes:
            if process.poll() is None:  # If process is still running
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print(f"Terminated process {process.pid}")

    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        print("\nReceived termination signal. Cleaning up...")
        self.cleanup()
        sys.exit(0)

def main():
    num_processes=sys.argv([1]),
    # Example of different arguments for each process
    args_list = [
        [f"--id={i}", f"--data=dataset_{i}.txt"] for i in range(num_processes)
    ]
    
    launcher = ParallelLauncher()
    
    # Launch 100 processes with their respective arguments
    launcher.launch_processes(
        script_path="/home/csartor1/code/NODS/grid_search_Aplus_Aminus/run_simulation.py",  # Replace with your script name
        num_processes=num_processes,
        args_per_process=args_list
    )
    
    # Wait for all processes to complete
    launcher.wait_for_completion()

if __name__ == "__main__":
    main()