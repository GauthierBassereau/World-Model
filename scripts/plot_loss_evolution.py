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

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sorted_steps = sorted(list(all_steps))
        sorted_scenarios = sorted(list(scenarios))
        
        # Color map for steps (Hue)
        # Use a colormap that has distinct colors across the range
        step_cmap = plt.get_cmap('turbo') 
        step_norm = plt.Normalize(vmin=min(sorted_steps), vmax=max(sorted_steps))

        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Add scenario legend entries (using a neutral gray to show lightness difference)
        for i, scenario in enumerate(sorted_scenarios):
            # Calculate lightness factor
            # We want to vary lightness for scenarios. 
            # If 2 scenarios: 1.0 (normal), 0.6 (darker) or 1.3 (lighter).
            # Let's say we map scenario index to lightness multiplier [0.7, 1.3]
            if len(sorted_scenarios) > 1:
                # Map i from 0..N-1 to 1.3..0.7
                lightness = 1.3 - 0.6 * (i / (len(sorted_scenarios) - 1))
            else:
                lightness = 1.0
            
            # Create a proxy artist for legend
            # We use a neutral color to demonstrate the lightness, or just black modified
            base_gray = (0.5, 0.5, 0.5)
            c = adjust_lightness(base_gray, lightness)
            legend_elements.append(Line2D([0], [0], color=c, lw=2, label=f"Scenario: {scenario}"))

        # Plot curves
        for step in sorted_steps:
            base_color = step_cmap(step_norm(step))
            
            for i, scenario in enumerate(sorted_scenarios):
                curves = all_curves[scenario]
                if step not in curves:
                    continue
                
                data = curves[step]
                if not data:
                    continue
                
                horizons = sorted(data.keys())
                values = [data[h] for h in horizons]
                
                # Apply lightness adjustment
                if len(sorted_scenarios) > 1:
                    lightness = 1.3 - 0.6 * (i / (len(sorted_scenarios) - 1))
                else:
                    lightness = 1.0
                
                final_color = adjust_lightness(base_color, lightness)
                
                ax.plot(horizons, values, color=final_color, alpha=0.8)

        ax.set_xlabel("Rollout Horizon (t+k)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Evolution - All Scenarios")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for steps
        sm = plt.cm.ScalarMappable(cmap=step_cmap, norm=step_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Training Step")
        
        # Add legend for scenarios
        ax.legend(handles=legend_elements, loc='upper left')
        
        filename_suffix = "loss" if metric_name == "L1 Loss" else "variance"
        output_path = os.path.join(args.output_dir, f"{run.id}_combined_{filename_suffix}.png")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved plot to {output_path}")
        
        images_to_log[f"combined_{filename_suffix}"] = output_path

    if images_to_log:
        try:
            print(f"Logging images to run {run.id}...")
            # We need project and entity
            # Note: resume="allow" allows us to log to an existing run
            with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="allow") as wrun:
                log_payload = {}
                for scenario, image_path in images_to_log.items():
                    log_payload[f"evaluation/loss_evolution/{scenario}"] = wandb.Image(image_path)
                wrun.log(log_payload)
            print("Logging complete.")
        except Exception as e:
            print(f"Failed to log to wandb: {e}")
    else:
        print("No images generated to log.")

if __name__ == "__main__":
    main()
