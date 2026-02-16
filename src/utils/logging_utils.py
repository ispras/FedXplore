import os
import sys
import hydra
import subprocess


def redirect_stdout_to_log():
    # Read output file (created by >output/file.txt)
    redirect_file = subprocess.run(
        ["readlink", "-f", f"/proc/{os.getpid()}/fd/1"], capture_output=True, text=True
    ).stdout.strip()

    # hydra log file
    main_log_file = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/output.txt"
    )
    f = open(main_log_file, "w")

    # If stdout is a terminal -> keep console + also write to file
    if redirect_file.startswith("/dev/pts/"):

        class Tee:
            def __init__(self, a, b):
                self.a, self.b = a, b

            def write(self, s):
                self.a.write(s)
                self.b.write(s)

            def flush(self):
                self.a.flush()
                self.b.flush()

        sys.stdout = Tee(sys.__stdout__, f)
        sys.stderr = Tee(sys.__stderr__, f)

        print("Information about files:")
        print(f"File to logging: {main_log_file}")
        print()
        return

    # Otherwise stdout was redirected to a file
    os.remove(redirect_file)

    sys.stdout = f
    sys.stderr = f

    os.symlink(main_log_file, redirect_file)

    print("Information about files:")
    print(f"File to logging: {main_log_file}")
    print(f"Link file: {redirect_file}")
    print()
