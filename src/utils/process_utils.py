import psutil
import os
import signal
import time
import sys
import traceback
import warnings
from hydra.core.hydra_config import HydraConfig


def handle_signal(sign, frame):
    # We wanna process the signal once
    signal.signal(sign, signal.SIG_IGN)
    raise RuntimeError(sign)  # send main process to except block


def errors_parent_handler(func):
    def wrapper(*args, **kwargs):
        # Code was written trying to support Windows users
        # but it hasn't been tested.
        if os.name == "nt":  # Windows
            warnings.warn(
                "The code hasn't been tested on Windows. There may be errors."
            )

        # psutil is to save a list of process for except block
        current_process = psutil.Process()

        signal.signal(
            signal.SIGTERM,
            handle_signal,
        )

        try:
            # call train
            func(*args, **kwargs)

        except SystemExit:  # Federated learning end
            pass

        except BaseException as e:
            signal.signal(
                signal.SIGTERM,
                signal.SIG_IGN,
            )  # In case of errors in father process we don't need to receive any signals

            if not (isinstance(e, RuntimeError) and e.args[0] == signal.SIGTERM):
                print(traceback.format_exc())

            children = current_process.children()
            for indx, child in enumerate(children):
                try:
                    os.kill(child.pid, signal.SIGTERM)
                except:  # already dead
                    pass
                print(f"Child {indx} was killed!")

        remove_trust_map_file()
        sys.exit(0)

    return wrapper


def errors_child_handler(func):
    def wrapper(*args, **kwargs):
        # here psutil is to know ppid (since windows doesn't support ppid)
        child_process = psutil.Process()

        signal.signal(
            signal.SIGTERM,
            handle_signal,
        )

        try:
            # call multiprocess_client
            func(*args, **kwargs)

        except SystemExit:  # The process completed training
            sys.exit(0)

        except BaseException as e:
            time.sleep(2)
            if not (isinstance(e, RuntimeError) and e.args[0] == signal.SIGTERM):
                print(traceback.format_exc())
            """
            Sleep is necessary.
            If there is a mistake in the beginning of multiprocess_client,
            Father will kill those who have already born.
            So other children will survive.
            """
            try:
                os.kill(child_process.parent().pid, signal.SIGTERM)
            except:
                pass
            sys.exit(1)

    return wrapper


def remove_trust_map_file():
    try:
        save_dir = HydraConfig.get().runtime.output_dir
        os.remove(os.path.join(save_dir, "trust_map_file.csv"))
        print(f"Remove local trust_map_file.csv")
    except:
        pass
