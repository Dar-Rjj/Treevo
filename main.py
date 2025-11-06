import hydra
import logging 
import os
import uuid
from pathlib import Path
import subprocess
from utils.utils import init_client, print_hyperlink


ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")
    logging.info(f"Max Function Evaluation: {cfg.max_fe}")

    client = init_client(cfg)
    # optional clients for operators (ReEvo)
    long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    
    # add treevo(tree)
    if cfg.algorithm == "reevo":
        from reevo import ReEvo as LHH
    elif cfg.algorithm == "ael":
        from baselines.ael.ga import AEL as LHH
    elif cfg.algorithm == "eoh":
        from baselines.eoh import EoH as LHH
    elif cfg.algorithm == "TReEvo":
        from TReEvo import TReEvo as LHH
    elif cfg.algorithm == "TreEvo":
        from TreEvo import TreEvo as LHH
    else:
        raise NotImplementedError

    # Main algorithm
    # add treevo(tree)
    if cfg.algorithm != "reevo" and cfg.algorithm != "treevo" and cfg.algorithm != "treevoo":
        lhh = LHH(cfg, ROOT_DIR, client)
    else:
        lhh = LHH(cfg, ROOT_DIR, client, long_reflector_llm=long_ref_llm, short_reflector_llm=short_ref_llm, 
                  crossover_llm=crossover_llm, mutation_llm=mutation_llm)
        
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
    logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")
    
    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    unique_filename = f"gpt_temp_best_result_{uuid.uuid4().hex[:8]}.py"
    output_path = os.path.join(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/tmp", unique_filename)

    with open(output_path, 'w') as file:
        file.writelines(best_code_overall + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_test_stdout.txt"
    logging.info(f"Running validation script...: {print_hyperlink(test_script)}")

    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "test", output_path, f'{cfg.object_n}'], stdout=stdout)
    logging.info(f"Validation script finished. Results are saved in {print_hyperlink(test_script_stdout)}.")
    
    # Print the results
    with open(test_script_stdout, 'r') as file:
        for line in file.readlines():
            logging.info(line.strip())

if __name__ == "__main__":
    main()