profile-mem:
	python profile-mem.py
	echo "Converting memory snapshot to HTML"
	python _memory_viz.py trace_plot mem-profiles/mem-snapshot.pickle -o mem-profiles/mem-snapshot.html

download-html-conversion-tool:
	wget https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/cuda/_memory_viz.py


ncu-profile-fused-triton:
	mkdir -p profiles/ncu
	ncu --set full \
		--launch-skip 10 \
		--kernel-name fused_sample_triton_kernel \
		-fo profiles/ncu/fused-triton python ./speed_test.py --name fused-triton

ncu-profile-naive-compiled:
	mkdir -p profiles/ncu
	ncu --nvtx \
		--nvtx-include "kernel/" \
		--launch-skip 10 \
		-fo profiles/ncu/naive-compiled python ./speed_test.py --name naive-compiled


nsight-profile-fused-triton:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/fused-triton python ./speed_test.py --name fused-triton

nsight-profile-naive-compiled:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/naive-compiled python ./speed_test.py --name naive-compiled