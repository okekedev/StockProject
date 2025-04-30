"""
Utility for running external Python scripts.
"""
import subprocess
import os

def run_script(script_path, args=None):
    """
    Run an external Python script with optional arguments.
    
    Args:
        script_path (str): Path to the Python script.
        args (list, optional): List of command-line arguments.
        
    Returns:
        tuple: (success, output_or_error)
            - success (bool): True if script executed successfully, False otherwise.
            - output_or_error (str): Either stdout output or error message.
    """
    if not os.path.exists(script_path):
        return False, f"Error: Script not found - {script_path}"
    
    # Prepare command
    command = ['python', script_path]
    if args:
        command.extend(args)
    
    try:
        # Run script with captured output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Script error: {e.stderr}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"