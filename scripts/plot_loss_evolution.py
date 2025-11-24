import argparse
import os
import collections
import matplotlib.pyplot as plt
import wandb
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Plot L1 loss evolution from WandB run.")
    parser.add_argument("--run_path", type=str, required=True, help="WandB run path (entity/project/run_id) or run_id")
    parser.add_argument("--output_dir", type=str, default="archive/loss_evolution", help="Directory to save output images")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Connecting to WandB run: {args.run_path}")
    api = wandb.Api()
    try:
        run = api.run(args.run_path)
    except Exception as e:
        print(f"Error fetching run {args.run_path}: {e}")
        return

    print(f"Fetching history for run: {run.name} ({run.id})")
    # Fetch history. We need all keys.
    # scan_history is better for large histories as it returns an iterator
    history = list(run.scan_history())
    
    if not history:
        print("No history found for this run.")
        return

    # Identify scenarios
    # Look for keys: evaluation_rollouts_metrics/{scenario}/mean_t+0
    scenarios = set()
    for row in history:
        for key in row.keys():
            if key.startswith("evaluation_rollouts_metrics/") and "/mean_t+0" in key:
                # Extract scenario name
                # Format: evaluation_rollouts_metrics/SCENARIO/mean_t+0
                parts = key.split("/")
                if len(parts) >= 3:
                    scenario = parts[1]
                    scenarios.add(scenario)
    
    print(f"Found scenarios: {sorted(list(scenarios))}")

    os.makedirs(args.output_dir, exist_ok=True)
    images_to_log = {}
    # Metrics to plot: Name -> key prefix suffix
    metrics_to_plot = {
        "L1 Loss": "mean",
        "Variance": "var"
    }
    
    import colorsys
    import matplotlib.colors as mc

    def adjust_lightness(color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = mc.to_rgb(c)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, max(0, min(1, amount * l)), s)

    for metric_name, metric_suffix in metrics_to_plot.items():
        print(f"Processing metric: {metric_name}")
        
        # Collect data for all scenarios
        all_curves = {}
        all_steps = set()

        for scenario in scenarios:
            curves = collections.defaultdict(dict)
            for row in history:
                step = row.get("evaluation_step")
                if step is None:
                    step = row.get("_step")
                
                # Key format: evaluation_rollouts_metrics/{scenario}/{metric_suffix}_t+{k}
                prefix = f"evaluation_rollouts_metrics/{scenario}/{metric_suffix}_t+"
                
                for key, val in row.items():
                    if key.startswith(prefix):
                        try:
                            horizon = int(key[len(prefix):])
                            curves[step][horizon] = val
                            all_steps.add(step)
                        except ValueError:
                            continue
            all_curves[scenario] = curves

        if not all_curves:
            print(f"No data found for {metric_name}.")
            continue

        # Plotting - Separate plot for each scenario
        sorted_steps = sorted(list(all_steps))
        sorted_scenarios = sorted(list(scenarios))
        
        # Color map for steps (Hue)
        step_cmap = plt.get_cmap('turbo') 
        step_norm = plt.Normalize(vmin=min(sorted_steps), vmax=max(sorted_steps))

        from matplotlib.lines import Line2D

        for scenario in sorted_scenarios:
            print(f"Plotting scenario: {scenario}")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            curves = all_curves[scenario]
            if not curves:
                print(f"No data for scenario {scenario} in metric {metric_name}")
                plt.close(fig)
                continue

            # Plot curves for this scenario
            for step in sorted_steps:
                if step not in curves:
                    continue
                
                data = curves[step]
                if not data:
                    continue
                
                horizons = sorted(data.keys())
                values = [data[h] for h in horizons]
                
                base_color = step_cmap(step_norm(step))
                ax.plot(horizons, values, color=base_color, alpha=0.8)

            ax.set_xlabel("Rollout Horizon (t+k)")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} Evolution - {scenario}")
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for steps
            sm = plt.cm.ScalarMappable(cmap=step_cmap, norm=step_norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Training Step")
            
            filename_suffix = "loss" if metric_name == "L1 Loss" else "variance"
            # Sanitize scenario name for filename
            safe_scenario = scenario.replace("/", "_").replace(" ", "_")
            output_path = os.path.join(args.output_dir, f"{run.id}_{safe_scenario}_{filename_suffix}.png")
            fig.savefig(output_path)
            plt.close(fig)
            print(f"Saved plot to {output_path}")
            
            images_to_log[f"{scenario}_{filename_suffix}"] = output_path

    if images_to_log:
        try:
            print(f"Logging images to run {run.id}...")
            # We need project and entity
            # Note: resume="allow" allows us to log to an existing run
            with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="allow") as wrun:
                log_payload = {}
                for key, image_path in images_to_log.items():
                    # key is like "{scenario}_{suffix}"
                    # We want to log it under evaluation/loss_evolution/{scenario}_{suffix}
                    # But wait, the user might want them grouped by scenario or by metric?
                    # Let's use evaluation/loss_evolution/{key} which includes scenario and metric type
                    log_payload[f"evaluation/loss_evolution/{key}"] = wandb.Image(image_path)
                wrun.log(log_payload)
            print("Logging complete.")
        except Exception as e:
            print(f"Failed to log to wandb: {e}")
    else:
        print("No images generated to log.")

if __name__ == "__main__":
    main()
