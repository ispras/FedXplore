import os
import sys
import hydra
import subprocess


def redirect_stdout_to_log():
    # Read output file (created by >output/file.txt)
    redirect_file = subprocess.run(
        ["readlink", "-f", f"/proc/{os.getpid()}/fd/1"], capture_output=True, text=True
    ).stdout
    redirect_file = redirect_file[: len(redirect_file) - 1]  # delete \n
    os.remove(redirect_file)

    # Get file to log learning (in dir created by hydra)
    main_log_file = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/output.txt"
    )

    # Swap stdout to log file
    f = open(main_log_file, "w")
    sys.stdout = f
    sys.stderr = f

    # Create link to log file (output/file.txt link to log file)
    os.symlink(main_log_file, redirect_file)

    print("Information about files:")
    print(f"File to logging: {main_log_file}")
    print(f"Link file: {redirect_file}")
    print()
