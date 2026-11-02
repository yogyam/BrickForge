# The virtualenv lives outside the project because ~/Desktop is iCloud-synced on this machine and the
# sync daemon marks files under dot-directories as hidden, which makes Python skip the editable .pth.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/brickforge

.PHONY: setup test fixtures demo bench sft-data sft-gate sft-train sft-bench rl-gate rl-train rl-bench clean

setup:
	uv sync

test:
	uv run pytest -q

fixtures:
	uv run brickforge fixtures

demo: fixtures
	uv run brickforge generate --mesh fixtures/meshes/sphere.stl --out runs/sphere
	@echo "--- intentional failure: the open box is not watertight and must be rejected (exit 1 is expected) ---"
	-@uv run brickforge generate --mesh fixtures/meshes/open_box_invalid.stl --out runs/open_box_invalid
	@echo "--- done. Open viewer/index.html and load runs/sphere/build.json (and voxels.json for the target) ---"

bench:
	uv run brickforge-bench fetch-thingi --n 40
	uv run brickforge-bench run --suite all --generators greedy_v1 --budgets 50 100 200 --out docs/benchmarks/baseline_v1

sft-data:
	uv run python -m ml.data --synthetic-per-family 200 --out data/sft/synthetic.jsonl

sft-gate:
	uv run --group ml python -m ml.train_sft --data data/sft/synthetic.jsonl --overfit 32 --steps 1500 --eval-every 500 --name sft_overfit32

sft-train:
	cat data/sft/synthetic.jsonl data/sft/thingi.jsonl > data/sft/examples.jsonl
	uv run --group ml python -m ml.train_sft --data data/sft/examples.jsonl --steps 20000 --augment --eval-every 1000 --eval-n 32 --name sft_v1

sft-bench:
	uv run --group ml brickforge-bench run --suite all --generators greedy_v1 sft_v1 sft_v1_nostop --budgets 50 100 200 --out docs/benchmarks/sft_v1
	uv run --group ml brickforge-bench audit docs/benchmarks/sft_v1.jsonl

rl-gate:
	uv run --group ml python -m ml.train_rl --overfit 16 --iters 20 --episodes 48 --workers 8 --eval-every 5 --name rl_gate16

rl-train:
	uv run --group ml python -m ml.train_rl --name rl_v1 --iters 250 --episodes 128 --workers 8 --epochs 3 --minibatch 128 --lr 1e-4 --augment --eval-every 10 --eval-n 64

rl-bench:
	uv run --group ml brickforge-bench run --suite all --generators greedy_v1 sft_v1 sft_v1_nostop sft_v1_bo8 rl_v1 --budgets 50 100 200 --out docs/benchmarks/rl_v1
	uv run --group ml brickforge-bench audit docs/benchmarks/rl_v1.jsonl

clean:
	rm -rf runs
