import subprocess
import time
import os
import sys
import argparse # Added for command-line arguments

# --- Configuration ---
# !!! IMPORTANT: Replace this with the correct path to your Stockfish executable if needed !!!
# The default can be overridden by the --stockfish-path command-line argument.
DEFAULT_STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
# --- End Configuration ---

def run_stockfish_test(engine_path, mode, value):
    """
    Launches Stockfish and runs analysis based on the selected mode.

    Args:
        engine_path (str): Path to the Stockfish executable.
        mode (str): 'time' or 'depth'.
        value (int): Time limit in ms (for 'time' mode) or target depth (for 'depth' mode).

    Returns:
        dict: A dictionary containing the result, or None on failure.
              For 'time' mode: {'max_depth': int}
              For 'depth' mode: {'time_ms': int|None, 'max_depth_observed': int}
    """
    if not os.path.exists(engine_path):
        print(f"Error: Stockfish executable not found at '{engine_path}'")
        print("Please update the DEFAULT_STOCKFISH_PATH or use the --stockfish-path argument.")
        return None

    # Basic check for executable (might not be perfect on all OS)
    if not os.path.isfile(engine_path) or (sys.platform != "win32" and not os.access(engine_path, os.X_OK)):
        print(f"Error: Path '{engine_path}' is not recognized as an executable file.")
        # return None # Let it try anyway, Popen will fail if it's truly not executable

    print(f"Starting Stockfish: {engine_path}")

    try:
        engine = subprocess.Popen(
            [engine_path],
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            encoding='utf-8'
        )
    except OSError as e:
        print(f"Error launching Stockfish: {e}")
        print("Ensure the path is correct and you have permissions.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during launch: {e}")
        return None

    def send_command(cmd):
        # print(f"Sending UCI: {cmd}") # Uncomment for debugging
        try:
            if engine.poll() is None: # Check if process is still running
                 engine.stdin.write(cmd + '\n')
                 engine.stdin.flush()
            else:
                 raise BrokenPipeError("Stockfish process terminated before command could be sent.")
        except (BrokenPipeError, OSError) as e:
             print(f"Error: Could not send command '{cmd}'. Stockfish might have crashed or closed stdin.")
             print(f"  Error details: {e}")
             # Check stderr for clues
             try:
                 stderr_output = engine.stderr.read()
                 if stderr_output:
                     print("--- Stockfish Stderr Output ---")
                     print(stderr_output)
                     print("-------------------------------")
             except Exception as stderr_e:
                 print(f"Could not read stderr: {stderr_e}")
             raise # Re-raise to be caught by the main try...except

    output_lines = [] # Store output for debugging if needed
    result = None # Initialize result

    try:
        # Initialize UCI
        send_command("uci")
        while True:
            line = engine.stdout.readline().strip()
            output_lines.append(line)
            if line == "uciok": break
            if not line and engine.poll() is not None:
                raise RuntimeError("Stockfish process exited unexpectedly during UCI init.")

        # Check readiness
        send_command("isready")
        while True:
            line = engine.stdout.readline().strip()
            output_lines.append(line)
            if line == "readyok": break
            if not line and engine.poll() is not None:
                 raise RuntimeError("Stockfish process exited unexpectedly waiting for readyok.")

        # Set position
        send_command("position startpos")

        # --- Start Search based on mode ---
        start_time = time.monotonic() # Start timer *before* sending 'go'
        go_command = ""
        if mode == 'time':
            time_ms = int(value)
            print(f"\nStarting search: Mode=time, Limit={time_ms}ms")
            go_command = f"go movetime {time_ms}"
        elif mode == 'depth':
            target_depth = int(value)
            print(f"\nStarting search: Mode=depth, Target Depth={target_depth}")
            go_command = f"go depth {target_depth}"
        else:
            raise ValueError("Invalid mode specified") # Should be caught by argparse

        send_command(go_command)

        # --- Process Output ---
        max_depth_reached = 0
        time_taken_ms = None
        target_depth_first_hit_time = None # Use monotonic time for first hit

        # Set a very generous overall timeout, especially for depth mode
        # This prevents infinite hangs if Stockfish behaves unexpectedly
        overall_timeout_seconds = 3600 # 1 hour - adjust if needed for very deep searches

        while True:
            # Check overall timeout first
            current_elapsed_total = time.monotonic() - start_time
            if current_elapsed_total > overall_timeout_seconds:
                print(f"\nError: Overall timeout ({overall_timeout_seconds}s) exceeded. Stopping read loop.")
                # Attempt to force stop Stockfish if it's still running
                if engine.poll() is None:
                    try: send_command("stop")
                    except: pass # Ignore errors trying to stop forcefully
                time.sleep(0.1) # Give it a moment
                # Treat as failure or incomplete? Depends on what we have so far.
                # For depth mode, if target wasn't hit, it's failure.
                # For time mode, report what we got.
                if mode == 'depth' and target_depth_first_hit_time is None:
                    print("Target depth was not reached before timeout.")
                    result = {"time_ms": None, "max_depth_observed": max_depth_reached}
                elif mode == 'time':
                     print("Reporting max depth reached before timeout.")
                     result = {"max_depth": max_depth_reached}
                else: # Depth mode, target was hit but timeout occurred before bestmove? Unlikely but possible
                     time_taken_ms = int((target_depth_first_hit_time - start_time) * 1000)
                     result = {"time_ms": time_taken_ms, "max_depth_observed": max_depth_reached}
                break # Exit the read loop

            # Read line (readline should block, no need for complex timeout here)
            line = engine.stdout.readline().strip()
            if not line:
                # Check if process exited cleanly or unexpectedly
                if engine.poll() is not None:
                     print("\nStockfish process ended.")
                     # If we haven't seen 'bestmove' yet, it might be an issue
                     # Check if we got a result before it ended
                     if result is None: # If no result determined yet
                          if mode == 'time':
                              result = {"max_depth": max_depth_reached}
                          elif mode == 'depth':
                              if target_depth_first_hit_time:
                                   time_taken_ms = int((target_depth_first_hit_time - start_time) * 1000)
                                   result = {"time_ms": time_taken_ms, "max_depth_observed": max_depth_reached}
                              else:
                                   print("Stockfish ended before target depth was reported.")
                                   result = {"time_ms": None, "max_depth_observed": max_depth_reached}
                     break # Exit loop if process ended
                else:
                    # Empty line but process still running? Might happen. Continue reading.
                    # print("Debug: Received empty line, process still running.")
                    continue


            output_lines.append(line) # Store all output
            # print(f"Received: {line}") # Uncomment for detailed debugging

            if line.startswith("info"):
                parts = line.split()
                current_depth = 0
                try:
                    if "depth" in parts:
                        depth_index = parts.index("depth")
                        current_depth = int(parts[depth_index + 1])
                        max_depth_reached = max(max_depth_reached, current_depth)

                        # --- Mode-specific logic within info parsing ---
                        if mode == 'depth' and target_depth_first_hit_time is None:
                            if current_depth >= target_depth:
                                target_depth_first_hit_time = time.monotonic() # Record time of first hit
                                elapsed_at_hit = target_depth_first_hit_time - start_time
                                print(f"  Target depth {target_depth} reached at {elapsed_at_hit:.3f}s (Reported depth: {current_depth})")

                except (ValueError, IndexError):
                    pass # Ignore malformed info lines

            elif line.startswith("bestmove"):
                print(f"  'bestmove' received.")
                # --- Finalize results based on mode ---
                if mode == 'time':
                    result = {"max_depth": max_depth_reached}
                elif mode == 'depth':
                    if target_depth_first_hit_time:
                        time_taken_ms = int((target_depth_first_hit_time - start_time) * 1000)
                        result = {"time_ms": time_taken_ms, "max_depth_observed": max_depth_reached}
                    else:
                        # Bestmove received but target depth was never seen in info lines
                        print(f"Warning: 'bestmove' received, but target depth {target_depth} was never reported.")
                        result = {"time_ms": None, "max_depth_observed": max_depth_reached}
                break # Search is finished, exit the read loop

    except (BrokenPipeError, RuntimeError, Exception) as e:
         print(f"\nAn error occurred during Stockfish communication: {e}")
         # Print recent output if an error happened, helps debugging
         print("\n--- Last ~15 output lines from Stockfish ---")
         for l in output_lines[-15:]: # Print last 15 lines
              print(l)
         print("--------------------------------------------")
         result = None # Indicate failure

    finally:
        # --- Cleanup ---
        print("Cleaning up Stockfish process...")
        if engine.poll() is None: # Check if process still exists
            try:
                # Send quit command if still running
                print("Sending 'quit' command...")
                send_command("quit")
                engine.wait(timeout=2) # Wait briefly for graceful exit
                print("Stockfish exited gracefully.")
            except subprocess.TimeoutExpired:
                print("Stockfish did not quit via command, terminating forcefully.")
                engine.terminate()
                try:
                     engine.wait(timeout=1)
                     print("Stockfish terminated.")
                except subprocess.TimeoutExpired:
                     print("Stockfish did not terminate cleanly, killing.")
                     engine.kill() # Last resort
            except (BrokenPipeError, OSError, Exception) as e:
                print(f"Error during cleanup (sending quit/waiting): {e}")
                # If sending quit failed, try terminating directly
                if engine.poll() is None:
                    print("Terminating Stockfish process...")
                    engine.terminate()
                    time.sleep(0.5) # Give OS time
                    if engine.poll() is None: engine.kill()

        # Ensure streams are closed (redundant if using 'with' but safe)
        try:
            if engine.stdin: engine.stdin.close()
            if engine.stdout: engine.stdout.close()
            if engine.stderr: engine.stderr.close()
        except Exception:
            pass # Ignore errors closing streams that might already be closed

    return result

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Run Stockfish analysis in one of two modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    parser.add_argument("--stockfish-path",
                        default=DEFAULT_STOCKFISH_PATH,
                        help="Path to the Stockfish executable.")

    parser.add_argument("--mode", required=True, choices=['time', 'depth'],
                        help="Operating mode: 'time' (find max depth for a given time) "
                             "or 'depth' (find time to reach a given depth).")

    parser.add_argument("--time", type=int,
                        help="Time limit in milliseconds (required for --mode=time). Example: 1000 for 1 second.")

    parser.add_argument("--depth", type=int,
                        help="Target search depth (required for --mode=depth). Example: 20")

    args = parser.parse_args()

    # --- Validate arguments based on mode ---
    value = None
    if args.mode == 'time':
        if args.time is None:
            parser.error("--time <milliseconds> is required when --mode=time")
        if args.time <= 0:
             parser.error("--time must be a positive integer.")
        value = args.time
    elif args.mode == 'depth':
        if args.depth is None:
            parser.error("--depth <depth_level> is required when --mode=depth")
        if args.depth <= 0:
             parser.error("--depth must be a positive integer.")
        value = args.depth

    # --- Execute Test ---
    final_result = run_stockfish_test(args.stockfish_path, args.mode, value)

    # --- Print Results ---
    print("-" * 30)
    if final_result is None:
        print(f"❌ Test failed or was interrupted.")
    elif args.mode == 'time':
        max_depth = final_result.get("max_depth", 0)
        if max_depth > 0:
            print(f"✅ Mode 'time': Max depth reached in {args.time}ms was {max_depth}")
        else:
            # Check if the process actually ran but just didn't report depth
            # This check requires more sophisticated state tracking in run_stockfish_test
            # For now, assume if 0, it didn't report depth in time.
             print(f"❓ Mode 'time': Stockfish ran, but reported depth 0 or no depth info was found within {args.time}ms.")
    elif args.mode == 'depth':
        time_ms = final_result.get("time_ms")
        max_depth_obs = final_result.get("max_depth_observed", 0)
        if time_ms is not None:
            print(f"✅ Mode 'depth': Time to reach target depth {args.depth} was {time_ms}ms ({time_ms/1000.0:.3f}s)")
            if max_depth_obs > args.depth:
                 print(f"   (Note: Max depth observed during the full search until 'bestmove' was {max_depth_obs})")
            elif max_depth_obs < args.depth:
                 # This case shouldn't happen if time_ms is not None, but good to note
                 print(f"   (Warning: Max depth observed overall ({max_depth_obs}) was less than target {args.depth}, but target was reported hit earlier?)")
        else:
            print(f"❌ Mode 'depth': Target depth {args.depth} was NOT reached before the search ended.")
            print(f"   (Max depth observed during search was {max_depth_obs})")

    print("-" * 30)

if __name__ == "__main__":
    # Set default path - replace if needed, or rely on the command-line arg
    # Ensure the default path uses forward slashes or double backslashes
    DEFAULT_STOCKFISH_PATH = DEFAULT_STOCKFISH_PATH.replace("\\", "/")

    # Example of how you might set it specifically for Windows if not using the argument:
    # if sys.platform == "win32":
    #     DEFAULT_STOCKFISH_PATH = "C:/path/to/your/stockfish.exe" # Use forward slashes

    main()