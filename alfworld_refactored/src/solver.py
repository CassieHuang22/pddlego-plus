import tempfile, shutil, signal
from pathlib import Path
import subprocess
import os

TIME_LIMIT_SECONDS = 20  

def _kill_proc_tree(pid: int, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass

def _ensure_planutils_or_raise() -> None:
    if shutil.which("planutils") is None:
        raise RuntimeError(
            "planutils not found on PATH. Install and set up:\n"
            "  pipx install planutils && planutils setup && "
            "  planutils install dual-bfws-ffparser && planutils install val"
        )

def run_solver(domain_file: str,
               problem_file: str,
               solver: str = "dual-bfws-ffparser",
               max_retries: int = 1,
               validate_with_val: bool = True) -> dict:
    _ensure_planutils_or_raise()


    last_error = None
    for attempt in range(max_retries):
        with tempfile.TemporaryDirectory(prefix="pddl_run_") as tmpdir:
            tmp = Path(tmpdir)
            dom_path = tmp / "domain.pddl"
            prob_path = tmp / "problem.pddl"
            plan_path = tmp / "plan"         
            val_plan_path = tmp / "plan_tmp.txt"

            dom_path.write_text(domain_file, encoding="utf-8")
            prob_path.write_text(problem_file, encoding="utf-8")
            if plan_path.exists():
                plan_path.unlink()

            cmd = ["planutils", "run", solver, str(dom_path), str(prob_path)]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(tmp),
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=TIME_LIMIT_SECONDS)
            except subprocess.TimeoutExpired:
                _kill_proc_tree(proc.pid, signal.SIGTERM)
                try:
                    stdout, stderr = proc.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    _kill_proc_tree(proc.pid, signal.SIGKILL)
                    stdout, stderr = proc.communicate()

            solver_log = (stderr or "") + (stdout or "")

            plan_text = ""
            if plan_path.exists() and plan_path.stat().st_size:
                plan_text = plan_path.read_text(encoding="utf-8")

            result = {
                "output": {"plan": plan_text},
                "stderr": solver_log.strip()
            }
            return result

   
        last_error = RuntimeError("Unknown planner error.")

    raise RuntimeError(
        f"Max retries exceeded. Failed to run local solver.\n"
        f"Last error: {last_error}\n"
        f"--- Failing Domain File ---\n{domain_file}\n"
        f"--- Failing Problem File ---\n{problem_file}\n"
        f"--------------------------"
    )
