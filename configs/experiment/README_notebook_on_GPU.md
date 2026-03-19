# Running Jupyter notebooks block-by-block on GPU (Slurm)

On this cluster the GPU is only on compute nodes. To run a `.ipynb` **cell by cell** with GPU (e.g. for inference in `notebooks/models_inference.ipynb`), use an **interactive GPU session** and start Jupyter on the compute node, then connect from your side.

## Option 1: Interactive session + Jupyter (block-by-block)

### Step 1: Get an interactive GPU node

From the repo root, run:

```bash
bash configs/experiment/Submitscript_jupyter_interactive.sh
```

Or manually (customize partition/GPU type to match your cluster):

```bash
salloc --partition=main --gres=gpu:nvidia_h100_80gb_hbm3:1 --mem=64G --time=2:00:00 --cpus-per-task=4
```

When the job starts, your shell will be **on a compute node with GPU**.

### Step 2: Start Jupyter on the compute node

In that same terminal (now on the GPU node):

```bash
cd /path/to/Physically-conditioned-latent-diffusion-model-for-temperature
export PROJECT_ROOT=$(pwd) && export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
# Use your venv if you have one
.venv/bin/jupyter notebook --no-browser --port=8888
```

(If Jupyter is not in your venv, use `jupyter notebook --no-browser --port=8888`.)

Note the line that looks like:

```text
http://127.0.0.1:8888/lab?token=...
```

Copy that **token** (or the full URL); you’ll need it in the browser.

### Step 3: Connect from your machine (SSH tunnel)

From your **local machine** (or from the login node if you’re on the cluster in a different session), open an SSH tunnel so that `localhost:8888` on your side goes to the compute node’s Jupyter:

```bash
ssh -L 8888:localhost:8888 YOUR_USER@LOGIN_NODE_HOSTNAME
```

- If you already SSH’d to the login node and then ran `salloc` there, the compute node might be a different host. In that case, **from the same machine where the Jupyter process is running** you don’t need a tunnel: just open in the browser the URL that Jupyter printed (e.g. `http://127.0.0.1:8888/lab?token=...`), but that only works if your browser runs on that same machine (e.g. in a VSCode/Cursor port-forward or in a terminal on the login node with a browser).
- If you develop from your laptop: SSH from your **laptop** to the **login node** with the tunnel:  
  `ssh -L 8888:localhost:8888 user@login`. Then in the terminal that is **on the login node**, run `salloc` and then Jupyter. The tunnel forwards your laptop’s 8888 to the login node’s 8888; Jupyter is bound to 127.0.0.1 on the login node, so that works.  
  **Caveat:** With `salloc`, your shell moves to a **compute** node. So Jupyter is actually running on the compute node, not the login node. So you need the tunnel to reach the **compute node**. That usually means either:
  - **ProxyJump**: `ssh -L 8888:COMPUTE_NODE:8888 -J user@login user@login` and then you’d need to know the compute node name (e.g. from `hostname` in the salloc shell), or
  - Many clusters provide a way to run Jupyter “on the compute node” but the URL is given on the login node (e.g. Open On-Demand). If not, the standard approach is: SSH to login with `-L 8888:localhost:8888`, then in that session run `srun --pty ... jupyter notebook --no-browser --port=8888` so that Jupyter runs on the compute node but the port is forwarded back through the login node. So:

**Recommended (tunnel to login node, Jupyter via srun on compute node):**

On your **local machine**:

```bash
ssh -L 8888:localhost:8888 YOUR_USER@LOGIN_NODE
```

On the **login node** (in that SSH session):

```bash
cd /path/to/Physically-conditioned-latent-diffusion-model-for-temperature
srun --partition=main --gres=gpu:nvidia_h100_80gb_hbm3:1 --mem=64G --time=2:00:00 --pty \
  bash -c 'export PROJECT_ROOT=$PWD PYTHONPATH=$PWD:$PYTHONPATH && .venv/bin/jupyter notebook --no-browser --port=8888'
```

Then in your **local** browser open: `http://localhost:8888` and paste the token Jupyter printed. You can now run the notebook block-by-block with GPU.

**If the browser cannot connect:** Jupyter is bound to the **compute** node. On many clusters you must also forward from the login node to the compute node. In the `salloc` shell run `hostname` to get the compute node name (e.g. `node42`). In a **second** terminal on the login node run: `ssh -L 8888:localhost:8888 node42` and leave it open. Then the tunnel laptop → login → compute will work. Some clusters offer Jupyter via a web portal (e.g. Open On-Demand); check your cluster docs.

### Step 4: Use the notebook

In the browser, open `notebooks/models_inference.ipynb` (or any `.ipynb`). Run cells one by one; the kernel runs on the compute node and has GPU access.

When done, in the terminal stop Jupyter (Ctrl+C) and type `exit` to leave the allocation.

---

## Option 2: Run the whole notebook once on GPU (batch, no interaction)

If you don’t need to run cells one-by-one and just want to execute the entire notebook on GPU and save outputs:

1. Convert the notebook to a script and run it with your existing Slurm submit script, or  
2. Use `papermill` in the Slurm job to run the notebook and save the result notebook.

Example with a one-off Slurm job (adjust paths and partition/GPU to match your cluster):

```bash
# From repo root
sbatch --job-name=run_inference_nb --partition=main --gres=gpu:nvidia_h100_80gb_hbm3:1   --mem=64G --time=1:00:00 --output=slurm_logs/inference_nb-%j.out --error=slurm_logs/inference_nb-%j.err   --wrap="cd $PWD && export PYTHONPATH=$PWD:\$PYTHONPATH && .venv/bin/jupyter nbconvert --to notebook --execute notebooks/models_inference.ipynb --output=models_inference_executed.ipynb"
```

Or execute as Python script:

```bash
.venv/bin/jupyter nbconvert --to script notebooks/models_inference.ipynb --output=models_inference
# Then in Submitscript.sh point the --wrap to: .venv/bin/python models_inference.py
```

---

**Summary:** For **block-by-block** inference testing, use **Option 1**: get an interactive GPU session (`salloc` or `srun --pty`), start Jupyter with `--no-browser --port=8888`, and connect via SSH tunnel so your browser talks to that Jupyter. Then run your `.ipynb` cells as usual.
