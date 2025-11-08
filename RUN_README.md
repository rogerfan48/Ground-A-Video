````markdown
# Batch Execution Script

## Usage

Simply run the script to automatically process all o2o, o2p, p2p configs:

```bash
cd ~/code/TRACK/Ground-A-Video
./run.sh
```

## Features

- ✅ Automatically activates conda environment (groundvideo_rtx5090)
- ✅ Automatically sets CUDA memory configuration
- ✅ Runs all config files (o2o, o2p, p2p)
- ✅ Real-time progress display and output
- ✅ Automatically saves logs to `logs/` directory
- ✅ Color-coded output with clear success/failure indicators
- ✅ Displays summary report at the end

## Output

- **Terminal**: Real-time progress display
- **Logs**: Automatically saved to `logs/run_YYYYMMDD_HHMMSS.log`
- **Results**: Saved to corresponding subdirectories in `outputs/`

## Configuration

Default clip_length is 15, but one special case requires 12 (p2p/n_u_c_s). You can adjust at the top of `run.sh`:

```bash
# Default clip_length
CLIP_LENGTH_DEFAULT=15

# Single special case override (category/name)
CLIP_LENGTH_OVERRIDE_CONFIG="p2p/n_u_c_s"
CLIP_LENGTH_OVERRIDE_VALUE=12
```

- To apply the override to a different config: change `CLIP_LENGTH_OVERRIDE_CONFIG` to `o2o/xxx`, `o2p/xxx` or `p2p/xxx`.
- To use the same clip_length for all configs: change `CLIP_LENGTH_DEFAULT` to your desired value, and leave `CLIP_LENGTH_OVERRIDE_CONFIG` empty or set to a non-existent name.

## Notes

- The script automatically uses `conda run` to ensure execution in the correct environment
- Each config displays success or failure upon completion
- Continues to the next config if a failure occurs
- Lists all failed configs at the end

That's it!

````
