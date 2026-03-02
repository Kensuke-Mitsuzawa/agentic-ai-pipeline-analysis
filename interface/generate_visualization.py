from pathlib import Path
import json
from agentic_ai_analysis import main


def main(
    path_dir_outcome_pickles: Path,
    path_dir_output_artifacts: Path):
    
    seq_target_files = list(path_dir_outcome_pickles.rglob("*result.pkl"))
    # print(seq_target_files)
    seq_container = main.load_results(seq_target_files)

    graph_dependency = {
        "agent_2_researcher": ["agent_4_judge_docs", "agent_5_final"],
        "agent_3_distractor": ["agent_4_judge_distractor", "agent_5_final"],
        "agent_4_judge_docs": ["agent_5_final"],
        "agent_4_judge_distractor": ["agent_5_final"]
    }

    dependency_start_node = {
        "prompt": ["agent_2_researcher", "agent_3_distractor"]
    }

    res_matrix = main.compute_cka_agent_nodes.compute_and_visualize_cka(
        seq_container, 
        path_dir_output_artifacts, 
        graph_dependency=graph_dependency,
        dependency_start_node=dependency_start_node
        )
    print(path_dir_output_artifacts / "cka_heatmap.png")
    print(path_dir_output_artifacts / "cka_graph.mmd")
    print(res_matrix)
    print(f"N-sample: {len(seq_container)}")

    # ---- output for analysis ----
    _node_key_distractor = "agent_3_distractor"
    _node_key_final = "agent_5_final"

    _pipeline_out: main.PipelineOutcome
    _out_pairs = []
    for _pipeline_out in seq_container:
        _outcome_distractor = _pipeline_out.nodes[_node_key_distractor]
        _outcome_final_answer = _pipeline_out.nodes[_node_key_final]

        _line_obj = dict(
            prompt=_pipeline_out.prompt,
            distractor=_outcome_distractor.outcome,
            final_answer=_outcome_final_answer.outcome
        )
        _out_pairs.append(_line_obj)
    # end for

    with open('./output_text.json', 'w') as f:
        f.write(json.dumps(_out_pairs, indent=4))

if __name__ == '__main__':
    from argparse import ArgumentParser
    _args = ArgumentParser()
    _args.add_argument('-i', '--path_dir_outcome_pickles', type=Path, required=True)
    _args.add_argument('-o', '--path_dir_output_artifacts', type=Path, required=True)

    _opts = _args.parse_args()
    assert _opts.path_dir_outcome_pickles.exists()

    main(
        path_dir_outcome_pickles=_opts.path_dir_outcome_pickles,
        path_dir_output_artifacts=_opts.path_dir_output_artifacts
    )
